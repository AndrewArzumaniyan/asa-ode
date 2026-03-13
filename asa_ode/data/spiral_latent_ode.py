from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass
class SpiralDataBatch:
    """Stores clean and sampled noisy spiral trajectories for experiments."""

    orig_ts: torch.Tensor
    orig_trajs: torch.Tensor
    noisy_trajs: torch.Tensor
    samp_ts: torch.Tensor
    samp_trajs: torch.Tensor
    samp_indices: torch.Tensor
    direction: torch.Tensor


def generate_spiral2d(
    nspiral: int = 1000,
    ntotal: int = 100,
    nsample: int = 50,
    start: float = 0.0,
    stop: float = 6.0 * math.pi,
    noise_std: float = 0.1,
    a: float = 0.0,
    b: float = 0.3,
    seed: int = 42,
    shared_time_grid: bool = True,
    shuffle_directions: bool = True,
    include_t0: bool = True,
) -> SpiralDataBatch:
    """Generates clockwise and counter-clockwise Archimedean spirals with noisy irregular observations."""
    if nspiral < 2:
        raise ValueError("nspiral must be at least 2")
    if ntotal < 2:
        raise ValueError("ntotal must be at least 2")
    if nsample < 2 or nsample > ntotal:
        raise ValueError("nsample must be in [2, ntotal]")

    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)

    theta = torch.linspace(float(start), float(stop), steps=ntotal, dtype=torch.float32)
    radius = float(a) + float(b) * theta

    x = radius * torch.cos(theta)
    y = radius * torch.sin(theta)

    cw = torch.stack([x, y], dim=-1)
    ccw = torch.stack([x, -y], dim=-1)

    ncw = nspiral // 2
    nccw = nspiral - ncw

    orig_trajs = torch.empty((nspiral, ntotal, 2), dtype=torch.float32)
    direction = torch.empty((nspiral,), dtype=torch.long)

    orig_trajs[:ncw] = cw.unsqueeze(0).expand(ncw, -1, -1)
    orig_trajs[ncw:] = ccw.unsqueeze(0).expand(nccw, -1, -1)
    direction[:ncw] = 1
    direction[ncw:] = -1

    if shuffle_directions:
        perm = torch.randperm(nspiral, generator=gen)
        orig_trajs = orig_trajs[perm]
        direction = direction[perm]

    noisy_trajs = orig_trajs + float(noise_std) * torch.randn(orig_trajs.shape, generator=gen, dtype=torch.float32)

    if shared_time_grid:
        if include_t0:
            if nsample < 2:
                raise ValueError("nsample must be at least 2 when include_t0 is True")
            tail = torch.randperm(ntotal - 1, generator=gen)[: nsample - 1] + 1
            sample_idx = torch.cat([torch.zeros(1, dtype=torch.long), tail], dim=0)
        else:
            sample_idx = torch.randperm(ntotal, generator=gen)[:nsample]
        sample_idx, _ = torch.sort(sample_idx)
        samp_trajs = noisy_trajs[:, sample_idx]
        samp_ts = theta[sample_idx]
    else:
        all_idx = []
        all_ts = []
        all_trajs = []
        for i in range(nspiral):
            _ = i
            if include_t0:
                if nsample < 2:
                    raise ValueError("nsample must be at least 2 when include_t0 is True")
                tail = torch.randperm(ntotal - 1, generator=gen)[: nsample - 1] + 1
                idx = torch.cat([torch.zeros(1, dtype=torch.long), tail], dim=0)
            else:
                idx = torch.randperm(ntotal, generator=gen)[:nsample]
            idx, _ = torch.sort(idx)
            all_idx.append(idx)
            all_ts.append(theta[idx])
            all_trajs.append(noisy_trajs[i, idx])
        sample_idx = torch.stack(all_idx, dim=0)
        samp_ts = torch.stack(all_ts, dim=0)
        samp_trajs = torch.stack(all_trajs, dim=0)

    return SpiralDataBatch(
        orig_ts=theta,
        orig_trajs=orig_trajs,
        noisy_trajs=noisy_trajs,
        samp_ts=samp_ts,
        samp_trajs=samp_trajs,
        samp_indices=sample_idx,
        direction=direction,
    )
