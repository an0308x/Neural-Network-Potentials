"""
Phase 3b: Train small MACE student via distillation from MACE teacher.

Same-family distillation: MACE-large → MACE-small.
Cleanest setting for REPA since both models produce equivariant irreps.

Usage:
    python training/train_mace_student.py --config configs/ethanol_mace2mace.yaml
    python training/train_mace_student.py --config configs/ethanol_mace2mace.yaml --use_original_data
    python training/train_mace_student.py --config configs/ethanol_mace2mace.yaml --no-wandb
"""

import os
import sys
import argparse
import yaml
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.loader import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.mace_small_student import MACESmallStudent, count_parameters
from nnp_utils.metrics import evaluate_model, print_metrics


def train_one_epoch(model, optimizer, train_loader, device, force_weight, energy_weight):
    model.train()
    total_loss = 0
    total_force_loss = 0
    total_energy_loss = 0
    n_batches = 0

    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch, compute_force=True)
        force_loss = nn.functional.mse_loss(out["force"], batch.force)
        energy_loss = nn.functional.mse_loss(out["energy"], batch.energy)
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
    parser.add_argument("--config", type=str, default="configs/ethanol_mace2mace.yaml")
    parser.add_argument("--use_original_data", action="store_true",
                        help="Train on original DFT data instead of distillation data")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="nnp-distillation")
    parser.add_argument("--wandb-name", type=str, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device(config.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Wandb ---
    use_wandb = not args.no_wandb
    if use_wandb:
        try:
            import wandb
            data_source = "original" if args.use_original_data else "distillation"
            run_name = args.wandb_name or f"mace_small_{config['data']['molecule']}_{data_source}"
            wandb.init(project=args.wandb_project, name=run_name,
                       config={"phase": "mace2mace_student", "model": "MACE-small",
                               "data_source": data_source, **config},
                       tags=["student", "mace-small", config["data"]["molecule"],
                             "mace2mace", data_source])
            print(f"Wandb: {wandb.run.url}")
        except Exception as e:
            print(f"wandb failed: {e}")
            use_wandb = False

    # --- Data ---
    sc = config["mace_student"]

    if args.use_original_data:
        from data.download_md17 import MD17DataModule
        dm = MD17DataModule(
            molecule=config["data"]["molecule"], data_dir=config["data"]["data_dir"],
            use_revised=config["data"].get("use_revised", False),
            n_train=config["data"]["n_train"], n_val=config["data"]["n_val"],
            cutoff=config["data"]["cutoff"], batch_size=sc["batch_size"],
            seed=config["data"]["seed"],
        )
        train_loader, val_loader, test_loader, stats = dm.get_dataloaders()
        print("Training on original MD17 data")
    else:
        distill_dir = Path(config["distillation"]["output_dir"])
        assert (distill_dir / "distillation_train.pt").exists(), (
            f"Distillation dataset not found at {distill_dir}. Run data/distillation_dataset.py first.")

        train_data = torch.load(distill_dir / "distillation_train.pt", weights_only=False)
        val_data = torch.load(distill_dir / "distillation_val.pt", weights_only=False)
        stats = torch.load(distill_dir / "stats.pt", weights_only=False)

        train_loader = DataLoader(train_data, batch_size=sc["batch_size"],
                                  shuffle=True, num_workers=0, pin_memory=True)
        val_loader = DataLoader(val_data, batch_size=sc["batch_size"],
                                shuffle=False, num_workers=0, pin_memory=True)

        # Load original test set for DFT evaluation
        from data.download_md17 import MD17DataModule
        dm = MD17DataModule(
            molecule=config["data"]["molecule"], data_dir=config["data"]["data_dir"],
            use_revised=config["data"].get("use_revised", False),
            n_train=config["data"]["n_train"], n_val=config["data"]["n_val"],
            cutoff=config["data"]["cutoff"], batch_size=sc["batch_size"],
            seed=config["data"]["seed"],
        )
        _, _, test_data, _ = dm.load_splits()
        test_loader = DataLoader(test_data, batch_size=sc["batch_size"], shuffle=False)
        print(f"Training on distillation dataset ({len(train_data)} configs)")

    # --- Model ---
    model = MACESmallStudent(
        num_interactions=sc["num_interactions"],
        hidden_dim=sc["hidden_dim"],
        max_ell=sc["max_ell"],
        num_basis=sc["num_basis"],
        cutoff=sc.get("cutoff", config["data"]["cutoff"]),
    ).to(device)
    model.set_energy_stats(stats["energy_mean"], stats["energy_std"], stats["n_atoms"])

    n_params = count_parameters(model)
    print(f"MACE-small Student: {n_params:,} parameters")
    if use_wandb:
        import wandb
        wandb.log({"model/n_params": n_params})

    # Resume
    start_epoch = 1
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, weights_only=False, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        print(f"Resumed from epoch {ckpt['epoch']}")

    # --- Optimizer ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=sc["lr"],
                                   weight_decay=sc["weight_decay"], amsgrad=True)
    scheduler = CosineAnnealingLR(optimizer,
        T_max=sc["max_epochs"] - sc.get("warmup_epochs", 10),
        eta_min=sc["min_lr"])

    warmup_epochs = sc.get("warmup_epochs", 10)
    force_weight = sc["force_weight"]
    energy_weight = sc["energy_weight"]

    ckpt_dir = Path(sc["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_val_force_mae = float("inf")
    patience_counter = 0

    print(f"\n{'='*60}")
    print(f"Starting MACE-small student training (MACE→MACE distillation)")
    print(f"  Teacher: {config['teacher']['hidden_dim']}d, {config['teacher']['num_interactions']}L, L_max={config['teacher']['max_ell']}")
    print(f"  Student: {sc['hidden_dim']}d, {sc['num_interactions']}L, L_max={sc['max_ell']}")
    print(f"  Force weight: {force_weight}, Energy weight: {energy_weight}")
    print(f"{'='*60}\n")

    for epoch in range(start_epoch, sc["max_epochs"] + 1):
        t0 = time.time()

        if epoch <= warmup_epochs:
            for pg in optimizer.param_groups:
                pg["lr"] = sc["lr"] * epoch / warmup_epochs

        train_metrics = train_one_epoch(model, optimizer, train_loader, device, force_weight, energy_weight)
        val_metrics = evaluate_model(model, val_loader, device)

        if epoch > warmup_epochs:
            scheduler.step()

        dt = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]

        if use_wandb:
            import wandb
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
                  f"F-loss {train_metrics['force_loss']:.6f} | "
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
                import wandb
                wandb.run.summary["best_val_force_mae_meV"] = best_val_force_mae * 1000
                wandb.run.summary["best_epoch"] = epoch
        else:
            patience_counter += 1

        if epoch % 100 == 0:
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                         "optimizer_state_dict": optimizer.state_dict()},
                        ckpt_dir / f"checkpoint_epoch{epoch}.pt")

        if patience_counter >= sc["patience"]:
            print(f"\nEarly stopping at epoch {epoch}")
            break

    print(f"\n{'='*60}")
    print(f"Training complete! Best val force MAE: {best_val_force_mae*1000:.2f} meV/Ang")
    print(f"{'='*60}\n")

    ckpt = torch.load(ckpt_dir / "best_model.pt", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    print("Test set evaluation (DFT labels):")
    test_metrics = evaluate_model(model, test_loader, device)
    print_metrics(test_metrics, prefix="  ")

    if use_wandb:
        import wandb
        wandb.log({f"test/{k}": v * 1000 for k, v in test_metrics.items() if "mae" in k or "rmse" in k})
        wandb.run.summary.update({"test_force_mae_meV": test_metrics["force_mae"] * 1000,
                                   "test_energy_mae_meV": test_metrics["energy_mae"] * 1000})
        wandb.finish()
    torch.save(test_metrics, ckpt_dir / "test_metrics.pt")


if __name__ == "__main__":
    main()