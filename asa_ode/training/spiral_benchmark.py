from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from tqdm import tqdm

from asa_ode.data.spiral_latent_ode import SpiralDataBatch
from asa_ode.models.spiral_benchmarks import (
    GRUBaseline,
    LatentODEVAE,
    gaussian_nll,
    kl_standard_normal,
    rmse_metric,
)


@dataclass
class SpiralPreparedBatch:
    """Stores tensors moved to runtime device for spiral training and evaluation."""

    observed_values: torch.Tensor
    observed_times: torch.Tensor
    observed_indices: torch.Tensor
    full_values: torch.Tensor
    full_times: torch.Tensor
    direction: torch.Tensor


def prepare_spiral_batch(data: SpiralDataBatch, device: torch.device) -> SpiralPreparedBatch:
    """Moves spiral tensors to device and validates shared-time training assumptions."""
    if data.samp_ts.ndim != 1:
        raise ValueError("This benchmark utility expects shared_time_grid=True (samp_ts must be 1D).")
    if data.samp_indices.ndim != 1:
        raise ValueError("This benchmark utility expects shared_time_grid=True (samp_indices must be 1D).")

    return SpiralPreparedBatch(
        observed_values=data.samp_trajs.to(device=device, dtype=torch.float32),
        observed_times=data.samp_ts.to(device=device, dtype=torch.float32),
        observed_indices=data.samp_indices.to(device=device, dtype=torch.long),
        full_values=data.orig_trajs.to(device=device, dtype=torch.float32),
        full_times=data.orig_ts.to(device=device, dtype=torch.float32),
        direction=data.direction.to(device=device, dtype=torch.long),
    )


@torch.no_grad()
def evaluate_latent_ode(
    model: LatentODEVAE,
    batch: SpiralPreparedBatch,
    obs_noise_std: float = 0.3,
) -> dict[str, torch.Tensor | float]:
    """Evaluates Latent ODE on full trajectories and returns RMSE and predictions."""
    model.eval()
    out = model(observed_values=batch.observed_values, pred_times=batch.full_times, sample=False)
    pred_full = out["x_pred"].permute(1, 0, 2)
    rmse = rmse_metric(pred_full, batch.full_values)

    logvar_val = math.log(float(obs_noise_std) ** 2)
    recon_logvar = torch.full_like(out["x_pred"], fill_value=logvar_val)
    recon_nll = gaussian_nll(batch.full_values.permute(1, 0, 2), out["x_pred"], recon_logvar)
    kl = kl_standard_normal(out["mu"], out["logvar"])

    return {
        "rmse": float(rmse.item()),
        "pred_full": pred_full.detach().cpu(),
        "mu": out["mu"].detach().cpu(),
        "logvar": out["logvar"].detach().cpu(),
        "z_traj": out["z_traj"].detach().cpu(),
        "recon_nll": float(recon_nll.item()),
        "kl": float(kl.item()),
    }


@torch.no_grad()
def evaluate_rnn(
    model: GRUBaseline,
    batch: SpiralPreparedBatch,
) -> dict[str, torch.Tensor | float]:
    """Evaluates RNN baseline rollout on full trajectories and returns RMSE and predictions."""
    model.eval()
    pred_full = model.rollout_full(
        observed_values=batch.observed_values,
        observed_indices=batch.observed_indices,
        full_times=batch.full_times,
    )
    rmse = rmse_metric(pred_full, batch.full_values)
    return {"rmse": float(rmse.item()), "pred_full": pred_full.detach().cpu()}


def train_latent_ode_fullbatch(
    model: LatentODEVAE,
    batch: SpiralPreparedBatch,
    n_iters: int = 2000,
    lr: float = 1e-2,
    obs_noise_std: float = 0.3,
    eval_every: int = 50,
    kl_anneal_iters: int = 500,
    grad_clip_norm: float = 0.0,
    desc: str = "Latent ODE",
) -> dict[str, object]:
    """Trains Latent ODE VAE on full-batch spiral observations."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    logvar_val = math.log(float(obs_noise_std) ** 2)
    recon_target = batch.observed_values.permute(1, 0, 2)
    recon_logvar = torch.full_like(recon_target, fill_value=logvar_val)

    history: dict[str, list[float]] = {
        "iter": [],
        "loss": [],
        "recon_nll": [],
        "kl": [],
        "beta": [],
        "rmse": [],
    }

    progress = tqdm(range(1, n_iters + 1), desc=desc)
    for iteration in progress:
        model.train()
        optimizer.zero_grad(set_to_none=True)

        out = model(observed_values=batch.observed_values, pred_times=batch.observed_times, sample=True)
        recon_nll = gaussian_nll(recon_target, out["x_pred"], recon_logvar)
        kl = kl_standard_normal(out["mu"], out["logvar"])
        if kl_anneal_iters > 0:
            beta = min(1.0, iteration / float(kl_anneal_iters))
        else:
            beta = 1.0
        loss = recon_nll + beta * kl

        loss.backward()
        if grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()

        if eval_every > 0 and (iteration == 1 or iteration % eval_every == 0 or iteration == n_iters):
            rmse_value = evaluate_latent_ode(model, batch, obs_noise_std=obs_noise_std)["rmse"]
        else:
            rmse_value = history["rmse"][-1] if history["rmse"] else float("nan")

        history["iter"].append(float(iteration))
        history["loss"].append(float(loss.item()))
        history["recon_nll"].append(float(recon_nll.item()))
        history["kl"].append(float(kl.item()))
        history["beta"].append(float(beta))
        history["rmse"].append(float(rmse_value))
        progress.set_postfix(loss=f"{loss.item():.4f}", rmse=f"{rmse_value:.4f}")

    final_eval = evaluate_latent_ode(model, batch, obs_noise_std=obs_noise_std)
    return {"history": history, "eval": final_eval}


def train_rnn_fullbatch(
    model: GRUBaseline,
    batch: SpiralPreparedBatch,
    n_iters: int = 2000,
    lr: float = 1e-2,
    obs_noise_std: float = 0.3,
    eval_every: int = 50,
    use_gaussian_nll: bool = True,
    grad_clip_norm: float = 0.0,
    desc: str = "RNN",
) -> dict[str, object]:
    """Trains GRU baseline on one-step observed prediction objective."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    logvar_val = math.log(float(obs_noise_std) ** 2)

    history: dict[str, list[float]] = {"iter": [], "loss": [], "rmse": []}

    progress = tqdm(range(1, n_iters + 1), desc=desc)
    for iteration in progress:
        model.train()
        optimizer.zero_grad(set_to_none=True)

        pred_next, target_next = model.predict_next_observed(
            observed_values=batch.observed_values,
            observed_times=batch.observed_times,
        )

        if use_gaussian_nll:
            step_logvar = torch.full_like(pred_next, fill_value=logvar_val)
            loss = gaussian_nll(target_next, pred_next, step_logvar)
        else:
            loss = torch.mean((pred_next - target_next).pow(2))

        loss.backward()
        if grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()

        if eval_every > 0 and (iteration == 1 or iteration % eval_every == 0 or iteration == n_iters):
            rmse_value = evaluate_rnn(model, batch)["rmse"]
        else:
            rmse_value = history["rmse"][-1] if history["rmse"] else float("nan")

        history["iter"].append(float(iteration))
        history["loss"].append(float(loss.item()))
        history["rmse"].append(float(rmse_value))
        progress.set_postfix(loss=f"{loss.item():.4f}", rmse=f"{rmse_value:.4f}")

    final_eval = evaluate_rnn(model, batch)
    return {"history": history, "eval": final_eval}
