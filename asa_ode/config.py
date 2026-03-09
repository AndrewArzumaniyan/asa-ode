from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class PathsConfig:
    """Stores file system paths used by training and evaluation."""

    data_root: str
    cache_path: str
    output_dir: str


@dataclass
class DataConfig:
    """Stores dataset and split settings."""

    train_ratio: float
    val_ratio: float
    test_ratio: float
    feature_names: list[str] | None
    min_timepoints: int


@dataclass
class ModelConfig:
    """Stores neural architecture hyperparameters."""

    latent_dim: int
    encoder_ode_hidden_dim: int
    dynamics_hidden_dim: int
    decoder_hidden_dim: int


@dataclass
class SolverConfig:
    """Stores ODE solver setup."""

    method: str
    rtol: float
    atol: float
    use_adjoint: bool


@dataclass
class TrainConfig:
    """Stores optimization and training loop settings."""

    batch_size: int
    epochs: int
    lr: float
    weight_decay: float
    grad_clip_norm: float
    early_stopping_patience: int
    log_every_steps: int


@dataclass
class EvalConfig:
    """Stores evaluation dataloader settings."""

    batch_size: int


@dataclass
class ExperimentConfig:
    """Stores the full experiment configuration."""

    seed: int
    device: str
    num_workers: int
    pin_memory: bool
    paths: PathsConfig
    data: DataConfig
    model: ModelConfig
    solver: SolverConfig
    train: TrainConfig
    eval: EvalConfig



def _read_json(path: str | Path) -> dict[str, Any]:
    """Reads JSON config file into a python dictionary."""
    with Path(path).expanduser().open("r", encoding="utf-8") as fp:
        return json.load(fp)


def load_config(path: str | Path) -> ExperimentConfig:
    """Loads full experiment config from JSON file."""
    raw = _read_json(path)
    return ExperimentConfig(
        seed=raw["seed"],
        device=raw["device"],
        num_workers=raw["num_workers"],
        pin_memory=raw["pin_memory"],
        paths=PathsConfig(**raw["paths"]),
        data=DataConfig(**raw["data"]),
        model=ModelConfig(**raw["model"]),
        solver=SolverConfig(**raw["solver"]),
        train=TrainConfig(**raw["train"]),
        eval=EvalConfig(**raw["eval"]),
    )
