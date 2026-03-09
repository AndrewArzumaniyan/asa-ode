from __future__ import annotations

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
        raise ImportError(
            "torchdiffeq is required. Install dependencies from requirements.txt"
        ) from exc
    return odeint_fn


class EncoderODEFunc(nn.Module):
    """Defines hidden state dynamics between observations in the encoder."""

    def __init__(self, hidden_dim: int, ode_hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, ode_hidden_dim),
            nn.Tanh(),
            nn.Linear(ode_hidden_dim, hidden_dim),
        )

    def forward(self, t: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """Computes encoder hidden derivative for ODE integration."""
        _ = t
        return self.net(h)


class FeaturewiseODERNNEncoder(nn.Module):
    """Encodes context trajectory with backward ODE-RNN independently per feature."""

    def __init__(
        self,
        latent_dim: int,
        ode_hidden_dim: int,
        method: str,
        rtol: float,
        atol: float,
        use_adjoint: bool,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.method = method
        self.rtol = rtol
        self.atol = atol

        self.odeint_fn = resolve_odeint(use_adjoint)
        self.ode_func = EncoderODEFunc(latent_dim, ode_hidden_dim)
        self.gru = nn.GRUCell(1, latent_dim)

    def forward(
        self,
        context_times: torch.Tensor,
        context_values: torch.Tensor,
        context_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encodes one patient context into initial latent state z0."""
        context_times = context_times.to(dtype=torch.float32)
        context_values = context_values.to(dtype=torch.float32)
        context_mask = context_mask.to(dtype=torch.float32)

        num_features = context_values.shape[-1]
        h = torch.zeros((num_features, self.latent_dim), dtype=context_values.dtype, device=context_values.device)

        rev_times = torch.flip(context_times, dims=[0])
        rev_values = torch.flip(context_values, dims=[0])
        rev_mask = torch.flip(context_mask, dims=[0])

        for step in range(rev_times.numel()):
            if step > 0:
                dt = rev_times[step - 1] - rev_times[step]
                if float(dt.item()) > 0.0:
                    interval = torch.stack([torch.zeros_like(dt), dt]).to(dtype=h.dtype, device=h.device)
                    h = self.odeint_fn(
                        self.ode_func,
                        h,
                        interval,
                        rtol=self.rtol,
                        atol=self.atol,
                        method=self.method,
                    )[-1]

            obs = rev_values[step].unsqueeze(-1)
            obs_mask = rev_mask[step].unsqueeze(-1)
            h_candidate = self.gru(obs, h)
            h = obs_mask * h_candidate + (1.0 - obs_mask) * h

        return h


class LatentDynamicsFunc(nn.Module):
    """Defines latent dynamics dz/dt for baseline feature-wise Neural ODE."""

    def __init__(self, latent_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Computes latent derivative for each feature independently."""
        _ = t
        return self.net(z)


class FeaturewiseDecoder(nn.Module):
    """Decodes latent feature vectors into scalar observations."""

    def __init__(self, latent_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Maps latent trajectory tensor [..., features, latent] to [..., features]."""
        leading = z.shape[:-1]
        flat = z.reshape(-1, z.shape[-1])
        out = self.net(flat).reshape(*leading)
        return out


class BaselineNeuralODE(nn.Module):
    """Combines encoder, latent dynamics and decoder for interpolation baseline."""

    def __init__(
        self,
        num_features: int,
        latent_dim: int,
        encoder_ode_hidden_dim: int,
        dynamics_hidden_dim: int,
        decoder_hidden_dim: int,
        method: str,
        rtol: float,
        atol: float,
        use_adjoint: bool,
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.latent_dim = latent_dim
        self.method = method
        self.rtol = rtol
        self.atol = atol

        self.encoder = FeaturewiseODERNNEncoder(
            latent_dim=latent_dim,
            ode_hidden_dim=encoder_ode_hidden_dim,
            method=method,
            rtol=rtol,
            atol=atol,
            use_adjoint=use_adjoint,
        )
        self.dynamics = LatentDynamicsFunc(latent_dim=latent_dim, hidden_dim=dynamics_hidden_dim)
        self.decoder = FeaturewiseDecoder(latent_dim=latent_dim, hidden_dim=decoder_hidden_dim)
        self.latent_odeint_fn = resolve_odeint(use_adjoint)

    def _ode_dtype(self) -> torch.dtype:
        """Returns float32 dtype used for ODE solves across devices."""
        return torch.float32

    def forward_single(
        self,
        context_times: torch.Tensor,
        context_values: torch.Tensor,
        context_mask: torch.Tensor,
        target_times: torch.Tensor,
    ) -> torch.Tensor:
        """Predicts target feature values for one patient trajectory."""
        ode_dtype = self._ode_dtype()
        context_times = context_times.to(dtype=ode_dtype)
        context_values = context_values.to(dtype=ode_dtype)
        context_mask = context_mask.to(dtype=ode_dtype)
        target_times = target_times.to(dtype=ode_dtype)

        if target_times.numel() == 0:
            return torch.zeros((0, self.num_features), dtype=context_values.dtype, device=context_values.device)

        z0 = self.encoder(context_times, context_values, context_mask)
        z0 = z0.to(dtype=ode_dtype)
        target_times = target_times.to(dtype=z0.dtype, device=z0.device)
        z_traj = self.latent_odeint_fn(
            self.dynamics,
            z0,
            target_times,
            rtol=self.rtol,
            atol=self.atol,
            method=self.method,
        )
        preds = self.decoder(z_traj)
        return preds

    def forward_batch(self, batch: dict[str, torch.Tensor], device: torch.device) -> torch.Tensor:
        """Runs model on padded batch and returns padded target predictions."""
        ode_dtype = self._ode_dtype()
        context_times = batch["context_times"].to(device=device, dtype=ode_dtype)
        context_values = batch["context_values"].to(device=device, dtype=ode_dtype)
        context_mask = batch["context_mask"].to(device=device, dtype=ode_dtype)
        context_lengths = batch["context_lengths"].to(device)

        target_times = batch["target_times"].to(device=device, dtype=ode_dtype)
        target_lengths = batch["target_lengths"].to(device)

        bsz, max_target_len, _ = batch["target_values"].shape
        preds = torch.zeros((bsz, max_target_len, self.num_features), dtype=context_values.dtype, device=device)

        for i in range(bsz):
            c_len = int(context_lengths[i].item())
            t_len = int(target_lengths[i].item())
            if c_len <= 0 or t_len <= 0:
                continue
            pred_i = self.forward_single(
                context_times=context_times[i, :c_len],
                context_values=context_values[i, :c_len],
                context_mask=context_mask[i, :c_len],
                target_times=target_times[i, :t_len],
            )
            preds[i, :t_len] = pred_i

        return preds
