from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from asa_ode.config import load_config
from asa_ode.data import build_dataloaders
from asa_ode.models import BaselineNeuralODE
from asa_ode.training import evaluate_model
from asa_ode.utils.runtime import seed_everything, setup_runtime



def parse_args() -> argparse.Namespace:
    """Parses command line arguments for model evaluation."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/baseline.json")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()



def main() -> None:
    """Loads checkpoint and evaluates baseline model on test split."""
    args = parse_args()
    cfg = load_config(args.config)

    if args.device is not None:
        cfg.device = args.device

    seed_everything(cfg.seed)
    runtime = setup_runtime(device_name=cfg.device, pin_memory=cfg.pin_memory)

    data_bundle = build_dataloaders(
        data_root=cfg.paths.data_root,
        cache_path=cfg.paths.cache_path,
        batch_size_train=cfg.train.batch_size,
        batch_size_eval=cfg.eval.batch_size,
        train_ratio=cfg.data.train_ratio,
        val_ratio=cfg.data.val_ratio,
        test_ratio=cfg.data.test_ratio,
        seed=cfg.seed,
        num_workers=cfg.num_workers,
        pin_memory=runtime.pin_memory,
        feature_names=cfg.data.feature_names,
        min_timepoints=cfg.data.min_timepoints,
        rebuild_cache=False,
    )

    use_adjoint = cfg.solver.use_adjoint
    if runtime.device.type == "mps" and use_adjoint:
        use_adjoint = False

    model = BaselineNeuralODE(
        num_features=len(data_bundle.feature_names),
        latent_dim=cfg.model.latent_dim,
        encoder_ode_hidden_dim=cfg.model.encoder_ode_hidden_dim,
        dynamics_hidden_dim=cfg.model.dynamics_hidden_dim,
        decoder_hidden_dim=cfg.model.decoder_hidden_dim,
        method=cfg.solver.method,
        rtol=cfg.solver.rtol,
        atol=cfg.solver.atol,
        use_adjoint=use_adjoint,
    ).to(runtime.device)

    ckpt_path = Path(args.checkpoint)
    checkpoint = torch.load(ckpt_path, map_location=runtime.device)
    model.load_state_dict(checkpoint["model_state_dict"])

    metrics = evaluate_model(model=model, loader=data_bundle.test_loader, device=runtime.device, desc="Test")

    output = {
        "device": str(runtime.device),
        "loss": metrics.loss,
        "step_time_sec": metrics.step_time_sec,
        "peak_memory_mb": metrics.peak_memory_mb,
        "checkpoint": str(ckpt_path),
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
