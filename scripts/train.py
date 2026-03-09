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
from asa_ode.training import evaluate_model, fit_model
from asa_ode.utils.runtime import seed_everything, setup_runtime



def parse_args() -> argparse.Namespace:
    """Parses command line arguments for training run."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/baseline.json")
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()



def main() -> None:
    """Runs full baseline training pipeline and saves artifacts."""
    args = parse_args()
    cfg = load_config(args.config)

    if args.device is not None:
        cfg.device = args.device
    if args.epochs is not None:
        cfg.train.epochs = args.epochs
    if args.batch_size is not None:
        cfg.train.batch_size = args.batch_size
        cfg.eval.batch_size = args.batch_size

    seed_everything(cfg.seed)
    runtime = setup_runtime(device_name=cfg.device, pin_memory=cfg.pin_memory)

    output_dir = Path(cfg.paths.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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
        rebuild_cache=args.rebuild_cache,
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

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
    )

    fit = fit_model(
        model=model,
        train_loader=data_bundle.train_loader,
        val_loader=data_bundle.val_loader,
        optimizer=optimizer,
        device=runtime.device,
        epochs=cfg.train.epochs,
        grad_clip_norm=cfg.train.grad_clip_norm,
        early_stopping_patience=cfg.train.early_stopping_patience,
        output_dir=output_dir,
    )

    checkpoint = torch.load(fit.checkpoint_path, map_location=runtime.device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_metrics = evaluate_model(
        model=model,
        loader=data_bundle.test_loader,
        device=runtime.device,
        desc="Test",
    )

    summary = {
        "device": str(runtime.device),
        "num_features": len(data_bundle.feature_names),
        "best_epoch": fit.best_epoch,
        "best_val_loss": fit.best_val_loss,
        "test_loss": test_metrics.loss,
        "test_step_time_sec": test_metrics.step_time_sec,
        "test_peak_memory_mb": test_metrics.peak_memory_mb,
        "use_adjoint": use_adjoint,
    }

    with (output_dir / "summary.json").open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
