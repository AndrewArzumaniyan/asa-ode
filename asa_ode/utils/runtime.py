from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class RuntimeContext:
    """Stores selected device and runtime flags."""

    device: torch.device
    pin_memory: bool


def select_device(requested: str = "auto") -> torch.device:
    """Selects the best available torch device for current machine."""
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    return torch.device("cpu")


def setup_runtime(device_name: str = "auto", pin_memory: bool = True) -> RuntimeContext:
    """Builds runtime context based on selected device capabilities."""
    device = select_device(device_name)
    effective_pin_memory = pin_memory and device.type == "cuda"
    return RuntimeContext(device=device, pin_memory=effective_pin_memory)


def seed_everything(seed: int) -> None:
    """Seeds all relevant libraries for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def reset_peak_memory(device: torch.device) -> None:
    """Resets peak memory counters for supported accelerators."""
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def get_peak_memory_mb(device: torch.device) -> float | None:
    """Returns peak allocated memory in MB when backend supports it."""
    if device.type == "cuda":
        return float(torch.cuda.max_memory_allocated(device) / (1024 * 1024))
    if device.type == "mps" and hasattr(torch.mps, "current_allocated_memory"):
        return float(torch.mps.current_allocated_memory() / (1024 * 1024))
    return None
