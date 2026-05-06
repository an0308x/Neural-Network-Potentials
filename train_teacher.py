"""
Phase 1: Train MACE Teacher on MD17

Usage:
    python training/train_teacher.py --config configs/ethanol.yaml
    python training/train_teacher.py --config configs/ethanol.yaml --no-wandb
"""

import os
import sys
import argparse
import yaml
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.download_md17 import MD17DataModule
from models.mace_teacher import MACETeacher
from nnp_utils.metrics import evaluate_model, print_metrics


def train_one_epoch(model, optimizer, train_loader, device, config):
    model.train()
    total_loss = 0
    total_force_loss = 0
    total_energy_loss = 0
    n_batches = 0

    force_weight = config["teacher"]["force_weight"]
    energy_weight = config["teacher"]["energy_weight"]

    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch, compute_force=True)
        energy_loss = nn.functional.mse_loss(out["energy"], batch.energy)
        force_loss = nn.functional.mse_loss(out["force"], batch.force)
        loss = energy_weight * energy_loss + force_weight * force_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()
        total_loss += loss.item()
        total_force_loss += force_loss.item()
        total_energy_loss += energy_loss.item()
        n_batches += 1

    return {
        "loss": total_loss / n_batches,
        "force_loss": total_force_loss / n_batches,
        "energy_loss": total_energy_loss / n_batches,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/ethanol.yaml")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="nnp-distillation")
    parser.add_argument("--wandb-name", type=str, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device(config.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Wandb
    use_wandb = not args.no_wandb
    if use_wandb:
        try:
            import wandb
            run_name = args.wandb_name or f"teacher_mace_{config['data']['molecule']}"
            wandb.init(project=args.wandb_project, name=run_name,
                       config={"phase": "teacher", "model": "MACE", **config},
                       tags=["teacher", "mace", config["data"]["molecule"]])
            print(f"Wandb: {wandb.run.url}")
        except Exception as e:
            print(f"wandb failed: {e}, continuing without")
            use_wandb = False

    # Data
    dm = MD17DataModule(
        molecule=config["data"]["molecule"], data_dir=config["data"]["data_dir"],
        use_revised=config["data"].get("use_revised", False),
        n_train=config["data"]["n_train"], n_val=config["data"]["n_val"],
        cutoff=config["data"]["cutoff"], batch_size=config["teacher"]["batch_size"],
        seed=config["data"]["seed"],
    )
    train_loader, val_loader, test_loader, stats = dm.get_dataloaders()
    print(f"Data loaded: {config['data']['molecule']}")
    print(f"  Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Model
    model = MACETeacher(
        num_interactions=config["teacher"]["num_interactions"],
        hidden_dim=config["teacher"]["hidden_dim"],
        max_ell=config["teacher"]["max_ell"],
        num_basis=config["teacher"]["num_basis"],
        cutoff=config["data"]["cutoff"],
        avg_num_neighbors=stats.get("avg_neighbors", 8.0),
    ).to(device)
    model.set_energy_stats(stats["energy_mean"], stats["energy_std"], stats["n_atoms"])

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"MACE Teacher: {n_params:,} parameters")
    if use_wandb:
        wandb.log({"model/n_params": n_params})

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["teacher"]["lr"],
                                   weight_decay=config["teacher"]["weight_decay"], amsgrad=True)

    if config["teacher"]["scheduler"] == "cosine":
        scheduler = CosineAnnealingLR(optimizer,
            T_max=config["teacher"]["max_epochs"] - config["teacher"]["warmup_epochs"],
            eta_min=config["teacher"]["min_lr"])
    else:
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=20,
                                       min_lr=config["teacher"]["min_lr"])

    warmup_epochs = config["teacher"]["warmup_epochs"]
    ckpt_dir = Path(config["teacher"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_val_force_mae = float("inf")
    patience_counter = 0

    print(f"\n{'='*60}\nStarting MACE teacher training\n{'='*60}\n")

    for epoch in range(1, config["teacher"]["max_epochs"] + 1):
        t0 = time.time()

        if epoch <= warmup_epochs:
            for pg in optimizer.param_groups:
                pg["lr"] = config["teacher"]["lr"] * epoch / warmup_epochs

        train_metrics = train_one_epoch(model, optimizer, train_loader, device, config)
        val_metrics = evaluate_model(model, val_loader, device)

        if epoch > warmup_epochs:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_metrics["force_mae"])
            else:
                scheduler.step()

        dt = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]

        if use_wandb:
            wandb.log({
                "epoch": epoch, "lr": lr,
                "train/loss": train_metrics["loss"],
                "train/force_loss": train_metrics["force_loss"],
                "train/energy_loss": train_metrics["energy_loss"],
                "val/force_mae_meV": val_metrics["force_mae"] * 1000,
                "val/force_rmse_meV": val_metrics["force_rmse"] * 1000,
                "val/energy_mae_meV": val_metrics["energy_mae"] * 1000,
                "val/energy_rmse_meV": val_metrics["energy_rmse"] * 1000,
                "val/force_cosine": val_metrics["force_cosine"],
                "time/epoch_seconds": dt,
            })

        if epoch % config.get("log_every", 10) == 0 or epoch == 1:
            print(f"Epoch {epoch:4d} | Loss {train_metrics['loss']:.6f} | "
                  f"Val F-MAE {val_metrics['force_mae']*1000:.2f} meV/Ang | "
                  f"Val E-MAE {val_metrics['energy_mae']*1000:.2f} meV | "
                  f"LR {lr:.2e} | {dt:.1f}s")

        if val_metrics["force_mae"] < best_val_force_mae:
            best_val_force_mae = val_metrics["force_mae"]
            patience_counter = 0
            torch.save({
                "epoch": epoch, "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_metrics": val_metrics, "config": config, "stats": stats,
            }, ckpt_dir / "best_model.pt")
            if use_wandb:
                wandb.run.summary["best_val_force_mae_meV"] = best_val_force_mae * 1000
                wandb.run.summary["best_epoch"] = epoch
        else:
            patience_counter += 1

        if patience_counter >= config["teacher"]["patience"]:
            print(f"\nEarly stopping at epoch {epoch}")
            break

    print(f"\n{'='*60}\nTraining complete! Best val force MAE: {best_val_force_mae*1000:.2f} meV/Ang\n{'='*60}\n")

    ckpt = torch.load(ckpt_dir / "best_model.pt", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    print("Test set evaluation:")
    test_metrics = evaluate_model(model, test_loader, device)
    print_metrics(test_metrics, prefix="  ")

    if use_wandb:
        wandb.log({f"test/{k}": v * 1000 for k, v in test_metrics.items() if "mae" in k or "rmse" in k})
        wandb.run.summary.update({"test_force_mae_meV": test_metrics["force_mae"] * 1000,
                                   "test_energy_mae_meV": test_metrics["energy_mae"] * 1000})
        wandb.finish()
    torch.save(test_metrics, ckpt_dir / "test_metrics.pt")


if __name__ == "__main__":
    main()