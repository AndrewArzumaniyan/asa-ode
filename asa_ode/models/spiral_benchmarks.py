from __future__ import annotations

import math

import torch
import torch.nn as nn


def resolve_odeint(use_adjoint: bool):
    """Resolves torchdiffeq integration function based on adjoint setting."""
    try:
        if use_adjoint:
            from torchdiffeq import odeint_adjoint as odeint_fn
        else:
            from torchdiffeq import odeint as odeint_fn
    except ImportError as exc:
        raise ImportError("torchdiffeq is required for Latent ODE experiments.") from exc
    return odeint_fn


def log_normal_pdf(x: torch.Tensor, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Computes elementwise Gaussian log-density with diagonal covariance."""
    log2pi = math.log(2.0 * math.pi)
    return -0.5 * (log2pi + logvar + (x - mean).pow(2) * torch.exp(-logvar))


def gaussian_nll(x: torch.Tensor, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Computes negative Gaussian log-likelihood averaged over batch."""
    ll = log_normal_pdf(x, mean, logvar).sum(dim=-1)
    return -ll.mean()


def trajectory_gaussian_nll(x: torch.Tensor, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Computes trajectory NLL by summing over time and averaging over batch."""
    ll = log_normal_pdf(x, mean, logvar).sum(dim=-1)
    if ll.ndim < 2:
        raise ValueError("trajectory_gaussian_nll expects at least [time, batch] leading dimensions.")
    ll_per_batch = ll.sum(dim=0)
    return -ll_per_batch.mean()


def kl_standard_normal(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Computes KL divergence from N(mu, sigma) to N(0, I), averaged over batch."""
    kl = 0.5 * (torch.exp(logvar) + mu.pow(2) - 1.0 - logvar).sum(dim=-1)
    return kl.mean()


def rmse_metric(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Computes RMSE over all batch, time and coordinate dimensions."""
    return torch.sqrt(torch.mean((pred - target).pow(2)))


class ReverseGRUEncoder(nn.Module):
    """Encodes observed trajectories in reverse time and outputs z0 posterior parameters."""

    def __init__(self, obs_dim: int = 2, rnn_hidden: int = 25, latent_dim: int = 4) -> None:
        super().__init__()
        self.input_proj = nn.Linear(obs_dim, rnn_hidden)
        self.gru_cell = nn.GRUCell(rnn_hidden, rnn_hidden)
        self.to_stats = nn.Linear(rnn_hidden, 2 * latent_dim)

    def forward(self, observed_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns mean and log-variance of q(z0|x) from reversed observations."""
        batch_size = observed_values.shape[0]
        h = torch.zeros(batch_size, self.gru_cell.hidden_size, device=observed_values.device, dtype=observed_values.dtype)

        for idx in range(observed_values.shape[1] - 1, -1, -1):
            x_proj = torch.tanh(self.input_proj(observed_values[:, idx]))
            h = self.gru_cell(x_proj, h)

        stats = self.to_stats(h)
        mu, logvar = torch.chunk(stats, chunks=2, dim=-1)
        return mu, logvar


class LatentODEFunc(nn.Module):
    """Defines latent dynamics dz/dt for the spiral Latent ODE model."""

    def __init__(self, latent_dim: int = 4, hidden_dim: int = 20) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Computes latent derivative for ODE integration."""
        _ = t
        return self.net(z)


class LatentDecoder(nn.Module):
    """Decodes latent trajectories into 2D observation trajectories."""

    def __init__(self, latent_dim: int = 4, hidden_dim: int = 20, obs_dim: int = 2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, obs_dim),
        )

    def forward(self, z_traj: torch.Tensor) -> torch.Tensor:
        """Maps latent tensor [time, batch, latent] to observation tensor [time, batch, obs]."""
        flat = z_traj.reshape(-1, z_traj.shape[-1])
        decoded = self.net(flat)
        return decoded.reshape(z_traj.shape[0], z_traj.shape[1], -1)


class LatentODEVAE(nn.Module):
    """Implements Latent ODE with reverse GRU encoder and VAE objective."""

    def __init__(
        self,
        obs_dim: int = 2,
        latent_dim: int = 4,
        rnn_hidden: int = 25,
        ode_hidden: int = 20,
        use_adjoint: bool = True,
        method: str = "dopri5",
        rtol: float = 1e-3,
        atol: float = 1e-4,
    ) -> None:
        super().__init__()
        self.method = method
        self.rtol = rtol
        self.atol = atol

        self.encoder = ReverseGRUEncoder(obs_dim=obs_dim, rnn_hidden=rnn_hidden, latent_dim=latent_dim)
        self.func = LatentODEFunc(latent_dim=latent_dim, hidden_dim=ode_hidden)
        self.decoder = LatentDecoder(latent_dim=latent_dim, hidden_dim=ode_hidden, obs_dim=obs_dim)
        self.odeint_fn = resolve_odeint(use_adjoint=use_adjoint)

    @staticmethod
    def sample_z0(mu: torch.Tensor, logvar: torch.Tensor, sample: bool) -> torch.Tensor:
        """Samples or returns posterior mean for z0."""
        if not sample:
            return mu
        eps = torch.randn_like(mu)
        return mu + torch.exp(0.5 * logvar) * eps

    def forward(
        self,
        observed_values: torch.Tensor,
        pred_times: torch.Tensor,
        sample: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Runs encoder, ODE solve and decoder for requested prediction times."""
        mu, logvar = self.encoder(observed_values)
        z0 = self.sample_z0(mu, logvar, sample=sample)
        z_traj = self.odeint_fn(self.func, z0, pred_times, method=self.method, rtol=self.rtol, atol=self.atol)
        x_pred = self.decoder(z_traj)
        return {"x_pred": x_pred, "z_traj": z_traj, "z0": z0, "mu": mu, "logvar": logvar}


class GRUBaseline(nn.Module):
    """Implements GRU and GRU+time baselines for spiral prediction."""

    def __init__(self, obs_dim: int = 2, hidden_dim: int = 25, use_time_concat: bool = False) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.use_time_concat = use_time_concat
        input_dim = obs_dim + 1 if use_time_concat else obs_dim
        self.gru_cell = nn.GRUCell(input_dim, hidden_dim)
        self.readout = nn.Linear(hidden_dim, obs_dim)

    def _step_input(self, x: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        """Builds per-step GRU input with optional delta-time concatenation."""
        if not self.use_time_concat:
            return x
        return torch.cat([x, dt.unsqueeze(-1)], dim=-1)

    def predict_next_observed(
        self,
        observed_values: torch.Tensor,
        observed_times: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predicts next observed point at each step for one-step training objective."""
        if observed_values.shape[1] < 2:
            raise ValueError("Observed sequence must contain at least 2 points.")

        batch_size = observed_values.shape[0]
        h = torch.zeros(batch_size, self.hidden_dim, device=observed_values.device, dtype=observed_values.dtype)
        preds = []

        for idx in range(observed_values.shape[1] - 1):
            if observed_times.ndim == 2:
                dt = observed_times[:, idx + 1] - observed_times[:, idx]
            else:
                dt = (observed_times[idx + 1] - observed_times[idx]).expand(batch_size)
            inp = self._step_input(observed_values[:, idx], dt)
            h = self.gru_cell(inp, h)
            preds.append(self.readout(h))

        pred = torch.stack(preds, dim=1)
        target = observed_values[:, 1:]
        return pred, target

    def rollout_full(
        self,
        observed_values: torch.Tensor,
        observed_indices: torch.Tensor,
        full_times: torch.Tensor,
    ) -> torch.Tensor:
        """Rolls out full trajectory on full time grid while anchoring at observed indices."""
        batch_size = observed_values.shape[0]
        total_steps = full_times.shape[0]
        h = torch.zeros(batch_size, self.hidden_dim, device=observed_values.device, dtype=observed_values.dtype)

        preds = torch.zeros(batch_size, total_steps, self.obs_dim, device=observed_values.device, dtype=observed_values.dtype)

        if observed_indices.ndim == 2:
            obs_idx = observed_indices[0]
        else:
            obs_idx = observed_indices
        obs_idx = obs_idx.to(device=full_times.device, dtype=torch.long)

        current_x = observed_values[:, 0]
        prev_t = full_times[0]
        obs_ptr = 1

        for step in range(total_steps):
            current_t = full_times[step]
            dt = (current_t - prev_t).expand(batch_size)
            inp = self._step_input(current_x, dt)
            h = self.gru_cell(inp, h)
            step_pred = self.readout(h)
            preds[:, step] = step_pred

            if obs_ptr < obs_idx.numel() and step == int(obs_idx[obs_ptr].item()):
                current_x = observed_values[:, obs_ptr]
                obs_ptr += 1
            else:
                current_x = step_pred
            prev_t = current_t

        return preds
