from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass
class SpiralMatrixBatch:
    """Bi-directional inward spirals for the latent ODE extrapolation benchmark."""

    full_times: torch.Tensor
    train_times: torch.Tensor
    future_times: torch.Tensor
    full_trajs: torch.Tensor
    train_trajs: torch.Tensor
    future_trajs: torch.Tensor
    noisy_train_trajs: torch.Tensor
    observed_times: torch.Tensor
    observed_trajs: torch.Tensor
    observed_indices: torch.Tensor
    direction: torch.Tensor
    phase_offsets: torch.Tensor
    initial_points: torch.Tensor
    cw_step_matrix: torch.Tensor
    ccw_step_matrix: torch.Tensor
    noise_std: float
    noise_mode: str
    step_angle: float
    contraction: float


def _build_step_matrix(step_angle: float, contraction: float) -> tuple[torch.Tensor, torch.Tensor]:
    cos_val = math.cos(step_angle)
    sin_val = math.sin(step_angle)

    ccw = contraction * torch.tensor(
        [[cos_val, -sin_val], [sin_val, cos_val]],
        dtype=torch.float32,
    )
    cw = contraction * torch.tensor(
        [[cos_val, sin_val], [-sin_val, cos_val]],
        dtype=torch.float32,
    )
    return cw, ccw


def _rollout_with_matrix(
    initial_points: torch.Tensor,
    step_matrix: torch.Tensor,
    n_steps: int,
) -> torch.Tensor:
    trajs = torch.empty((initial_points.shape[0], n_steps, 2), dtype=torch.float32)
    current = initial_points.to(dtype=torch.float32)
    trajs[:, 0] = current

    for step in range(1, n_steps):
        current = current @ step_matrix.T
        trajs[:, step] = current

    return trajs


def _sample_observation_indices(
    n_spirals: int,
    train_steps: int,
    n_obs: int,
    generator: torch.Generator,
    shared_time_grid: bool,
) -> torch.Tensor:
    draws = 1 if shared_time_grid else n_spirals
    scores = torch.rand((draws, train_steps), generator=generator, dtype=torch.float32)
    obs_idx = torch.argsort(scores, dim=-1)[:, :n_obs]
    obs_idx, _ = torch.sort(obs_idx, dim=-1)
    if shared_time_grid:
        obs_idx = obs_idx.expand(n_spirals, -1).clone()
    return obs_idx


def generate_spiral2d_matrix(
    n_spirals: int = 1000,
    train_steps: int = 100,
    future_steps: int = 100,
    n_obs: int = 50,
    noise_std: float = 0.03,
    noise_mode: str = "relative",
    radius_start: float = 1.0,
    radius_end: float = 0.15,
    radius_jitter: float = 0.0,
    seed: int = 42,
    shared_time_grid: bool = False,
    shuffle: bool = True,
) -> SpiralMatrixBatch:
    """Generates inward spirals via repeated matrix multiplication.

    Design choices:
    - follows the Neural ODE / latent ODE benchmark setup with 1000 2D spirals,
      two directions, irregularly sampled observations and future extrapolation;
    - uses 100 train points and a configurable future horizon
      (100 points by default to match the benchmark);
    - the first 100 points correspond to one full revolution, so the model observes
      exactly one turn and extrapolates one more turn into the future;
    - each step applies the same 2x2 rotation-contraction matrix, producing a
      logarithmic inward spiral instead of using an explicit polar-coordinate formula.

    Noise modes:
    - "absolute": iid Gaussian noise with fixed std in x/y;
    - "relative": iid Gaussian noise with std scaled by the current radius;
    - "none": no noise.
    """
    if n_spirals < 2:
        raise ValueError("n_spirals must be at least 2")
    if train_steps < 2:
        raise ValueError("train_steps must be at least 2")
    if future_steps < 1:
        raise ValueError("future_steps must be at least 1")
    if n_obs < 1 or n_obs > train_steps:
        raise ValueError("n_obs must be in [1, train_steps]")
    if radius_start <= 0.0:
        raise ValueError("radius_start must be positive")
    if radius_end <= 0.0 or radius_end >= radius_start:
        raise ValueError("radius_end must be in (0, radius_start)")
    if radius_jitter < 0.0 or radius_jitter >= 1.0:
        raise ValueError("radius_jitter must be in [0, 1)")
    if noise_std < 0.0:
        raise ValueError("noise_std must be non-negative")
    if noise_mode not in {"absolute", "relative", "none"}:
        raise ValueError("noise_mode must be one of: 'absolute', 'relative', 'none'")

    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)

    total_steps = train_steps + future_steps
    step_angle = 2.0 * math.pi / float(train_steps)
    full_times = torch.arange(total_steps, dtype=torch.float32) * step_angle
    train_times = full_times[:train_steps]
    future_times = full_times[train_steps:]

    contraction = (float(radius_end) / float(radius_start)) ** (1.0 / float(total_steps - 1))
    cw_step_matrix, ccw_step_matrix = _build_step_matrix(
        step_angle=step_angle,
        contraction=contraction,
    )

    phase_offsets = torch.linspace(
        0.0,
        2.0 * math.pi,
        steps=n_spirals + 1,
        dtype=torch.float32,
    )[:-1]
    global_phase = 2.0 * math.pi * torch.rand(1, generator=gen, dtype=torch.float32)
    phase_offsets = torch.remainder(phase_offsets + global_phase, 2.0 * math.pi)

    if radius_jitter > 0.0:
        radius_scale = 1.0 + radius_jitter * (2.0 * torch.rand(n_spirals, generator=gen) - 1.0)
    else:
        radius_scale = torch.ones(n_spirals, dtype=torch.float32)
    radii = float(radius_start) * radius_scale

    initial_points = torch.stack(
        [radii * torch.cos(phase_offsets), radii * torch.sin(phase_offsets)],
        dim=-1,
    )

    n_cw = n_spirals // 2
    n_ccw = n_spirals - n_cw
    direction = torch.empty(n_spirals, dtype=torch.long)
    direction[:n_cw] = -1
    direction[n_cw:] = 1

    full_trajs = torch.empty((n_spirals, total_steps, 2), dtype=torch.float32)
    full_trajs[:n_cw] = _rollout_with_matrix(
        initial_points=initial_points[:n_cw],
        step_matrix=cw_step_matrix,
        n_steps=total_steps,
    )
    full_trajs[n_cw:] = _rollout_with_matrix(
        initial_points=initial_points[n_cw:],
        step_matrix=ccw_step_matrix,
        n_steps=total_steps,
    )

    if shuffle:
        perm = torch.randperm(n_spirals, generator=gen)
        full_trajs = full_trajs[perm]
        direction = direction[perm]
        phase_offsets = phase_offsets[perm]
        initial_points = initial_points[perm]

    train_trajs = full_trajs[:, :train_steps]
    future_trajs = full_trajs[:, train_steps:]
    if noise_mode == "none" or noise_std == 0.0:
        noisy_train_trajs = train_trajs.clone()
    else:
        noise = torch.randn(
            train_trajs.shape,
            generator=gen,
            dtype=torch.float32,
        )
        if noise_mode == "absolute":
            noise_scale = torch.full_like(train_trajs, fill_value=float(noise_std))
        else:
            radius = torch.linalg.norm(train_trajs, dim=-1, keepdim=True).clamp_min(1e-6)
            noise_scale = float(noise_std) * radius.expand_as(train_trajs)
        noisy_train_trajs = train_trajs + noise_scale * noise

    observed_indices = _sample_observation_indices(
        n_spirals=n_spirals,
        train_steps=train_steps,
        n_obs=n_obs,
        generator=gen,
        shared_time_grid=shared_time_grid,
    )
    expanded_train_times = train_times.unsqueeze(0).expand(n_spirals, -1)
    observed_times = expanded_train_times.gather(1, observed_indices)
    observed_trajs = noisy_train_trajs.gather(
        1,
        observed_indices.unsqueeze(-1).expand(-1, -1, 2),
    )

    return SpiralMatrixBatch(
        full_times=full_times,
        train_times=train_times,
        future_times=future_times,
        full_trajs=full_trajs,
        train_trajs=train_trajs,
        future_trajs=future_trajs,
        noisy_train_trajs=noisy_train_trajs,
        observed_times=observed_times,
        observed_trajs=observed_trajs,
        observed_indices=observed_indices,
        direction=direction,
        phase_offsets=phase_offsets,
        initial_points=initial_points,
        cw_step_matrix=cw_step_matrix,
        ccw_step_matrix=ccw_step_matrix,
        noise_std=float(noise_std),
        noise_mode=noise_mode,
        step_angle=step_angle,
        contraction=contraction,
    )


def future_rmse(pred_future: torch.Tensor, target_future: torch.Tensor) -> torch.Tensor:
    """Computes RMSE on the future extrapolation horizon."""
    if pred_future.shape != target_future.shape:
        raise ValueError("pred_future and target_future must have the same shape")
    return torch.sqrt(torch.mean((pred_future - target_future).pow(2)))


__all__ = ["SpiralMatrixBatch", "generate_spiral2d_matrix", "future_rmse"]
