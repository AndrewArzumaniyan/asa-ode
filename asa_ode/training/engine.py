from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from asa_ode.training.losses import masked_mse
from asa_ode.utils.runtime import get_peak_memory_mb, reset_peak_memory


@dataclass
class LoopMetrics:
    """Stores aggregate metrics for one pass over a dataloader."""

    loss: float
    step_time_sec: float
    peak_memory_mb: float | None


@dataclass
class FitSummary:
    """Stores global training summary and best checkpoint path."""

    best_val_loss: float
    best_epoch: int
    checkpoint_path: str


def _move_targets(batch: dict[str, torch.Tensor], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Moves target tensors to selected device."""
    target_values = batch["target_values"].to(device=device, dtype=torch.float32)
    target_mask = batch["target_mask"].to(device=device, dtype=torch.float32)
    return target_values, target_mask


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip_norm: float,
    epoch_index: int,
    total_epochs: int,
) -> LoopMetrics:
    """Runs one optimization epoch and returns aggregate metrics."""
    model.train()
    reset_peak_memory(device)

    start = time.perf_counter()
    running_loss = 0.0
    steps = 0

    progress = tqdm(loader, desc=f"Train {epoch_index}/{total_epochs}", leave=False)
    for batch in progress:
        optimizer.zero_grad(set_to_none=True)

        pred = model.forward_batch(batch, device=device)
        target_values, target_mask = _move_targets(batch, device)
        loss = masked_mse(pred, target_values, target_mask)

        loss.backward()
        if grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()

        running_loss += float(loss.item())
        steps += 1
        progress.set_postfix(loss=f"{loss.item():.5f}")

    duration = time.perf_counter() - start
    avg_step_time = duration / max(steps, 1)
    mean_loss = running_loss / max(steps, 1)

    return LoopMetrics(loss=mean_loss, step_time_sec=avg_step_time, peak_memory_mb=get_peak_memory_mb(device))


@torch.no_grad()
def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device, desc: str) -> LoopMetrics:
    """Evaluates model without gradient updates and returns aggregate metrics."""
    model.eval()
    reset_peak_memory(device)

    start = time.perf_counter()
    running_loss = 0.0
    steps = 0

    progress = tqdm(loader, desc=desc, leave=False)
    for batch in progress:
        pred = model.forward_batch(batch, device=device)
        target_values, target_mask = _move_targets(batch, device)
        loss = masked_mse(pred, target_values, target_mask)

        running_loss += float(loss.item())
        steps += 1
        progress.set_postfix(loss=f"{loss.item():.5f}")

    duration = time.perf_counter() - start
    avg_step_time = duration / max(steps, 1)
    mean_loss = running_loss / max(steps, 1)

    return LoopMetrics(loss=mean_loss, step_time_sec=avg_step_time, peak_memory_mb=get_peak_memory_mb(device))


def fit_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int,
    grad_clip_norm: float,
    early_stopping_patience: int,
    output_dir: str | Path,
) -> FitSummary:
    """Trains model with validation, checkpointing, and early stopping."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "best_model.pt"
    history_path = out_dir / "history.json"

    best_val_loss = float("inf")
    best_epoch = -1
    patience_left = early_stopping_patience

    history: list[dict[str, float | int | None]] = []

    for epoch in tqdm(range(1, epochs + 1), desc="Epochs"):
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            grad_clip_norm=grad_clip_norm,
            epoch_index=epoch,
            total_epochs=epochs,
        )
        val_metrics = evaluate_model(model, val_loader, device=device, desc=f"Val {epoch}/{epochs}")

        record = {
            "epoch": epoch,
            "train_loss": train_metrics.loss,
            "val_loss": val_metrics.loss,
            "train_step_time_sec": train_metrics.step_time_sec,
            "val_step_time_sec": val_metrics.step_time_sec,
            "peak_memory_mb": train_metrics.peak_memory_mb,
        }
        history.append(record)

        is_best = val_metrics.loss < best_val_loss
        if is_best:
            best_val_loss = val_metrics.loss
            best_epoch = epoch
            patience_left = early_stopping_patience
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_metrics.loss,
                },
                ckpt_path,
            )
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    with history_path.open("w", encoding="utf-8") as fp:
        json.dump(history, fp, indent=2)

    if best_epoch < 0:
        raise RuntimeError("Training did not produce a valid checkpoint.")

    return FitSummary(best_val_loss=best_val_loss, best_epoch=best_epoch, checkpoint_path=str(ckpt_path))
