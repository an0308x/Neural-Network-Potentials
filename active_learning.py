"""
Phase 4: Active Learning Loop

1. Run short MD with student model
2. Relabel visited states with teacher
3. Augment training set
4. Retrain student

Usage:
    python training/active_learning.py --config configs/ethanol.yaml --n_loops 2
"""

import os
import sys
import argparse
import yaml
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.mace_teacher import MACETeacher
from models.painn_student import PaiNNStudent, count_parameters
from data.distillation_dataset import label_with_teacher, load_teacher
from nnp_utils.md_simulation import run_md
from nnp_utils.metrics import evaluate_model, print_metrics
from training.train_student import train_one_epoch


def load_student(config, device):
    """Load trained PaiNN student."""
    ckpt_path = Path(config["student"]["checkpoint_dir"]) / "best_model.pt"
    ckpt = torch.load(ckpt_path, weights_only=False, map_location=device)
    stats = ckpt["stats"]

    model = PaiNNStudent(
        num_interactions=config["student"]["num_interactions"],
        hidden_dim=config["student"]["hidden_dim"],
        num_basis=config["student"]["num_basis"],
        cutoff=config["student"]["cutoff"],
    ).to(device)

    model.set_energy_stats(stats["energy_mean"], stats["energy_std"], stats["n_atoms"])
    model.load_state_dict(ckpt["model_state_dict"])

    return model, stats


def active_learning_loop(config, device, n_loops=2):
    """Run the active learning loop."""

    # Load teacher and student
    teacher, stats = load_teacher(config, device)
    student, _ = load_student(config, device)

    # Load current training data
    distill_dir = Path(config["distillation"]["output_dir"])
    train_data = torch.load(distill_dir / "distillation_train.pt", weights_only=False)
    val_data = torch.load(distill_dir / "distillation_val.pt", weights_only=False)

    al_config = config["active_learning"]

    for loop in range(1, n_loops + 1):
        print(f"\n{'='*60}")
        print(f"Active Learning Loop {loop}/{n_loops}")
        print(f"{'='*60}")

        # --- Step 1: Run student MD ---
        print(f"\n--- Running student MD rollout ---")
        init_data = train_data[0]  # Use a training config as starting point

        snapshots = run_md(
            model=student,
            init_data=init_data,
            temperature=al_config["md_temperature"],
            timestep=al_config["md_timestep"],
            n_steps=al_config["md_n_steps"],
            save_every=al_config["md_save_every"],
            device=device,
            cutoff=config["data"]["cutoff"],
            seed=42 + loop,
        )

        # Subsample if too many
        max_new = al_config["max_new_samples"]
        if len(snapshots) > max_new:
            import random
            random.seed(42 + loop)
            snapshots = random.sample(snapshots, max_new)

        print(f"  Collected {len(snapshots)} new configurations")

        # --- Step 2: Relabel with teacher ---
        print(f"\n--- Relabeling with teacher ---")
        labeled_new = label_with_teacher(
            teacher, snapshots, device, config["data"]["cutoff"]
        )

        # --- Step 3: Augment training set ---
        train_data.extend(labeled_new)
        print(f"  Training set size: {len(train_data)}")

        # Save augmented dataset
        torch.save(train_data, distill_dir / f"distillation_train_loop{loop}.pt")

        # --- Step 4: Retrain student ---
        print(f"\n--- Retraining student ---")

        if not al_config.get("retrain_from_scratch", False):
            # Fine-tune from current weights
            print("  Fine-tuning from current checkpoint")
        else:
            # Reinitialize
            print("  Training from scratch")
            student = PaiNNStudent(
                num_interactions=config["student"]["num_interactions"],
                hidden_dim=config["student"]["hidden_dim"],
                num_basis=config["student"]["num_basis"],
                cutoff=config["student"]["cutoff"],
            ).to(device)
            student.set_energy_stats(
                stats["energy_mean"], stats["energy_std"], stats["n_atoms"]
            )

        # Create dataloaders
        train_loader = DataLoader(
            train_data,
            batch_size=config["student"]["batch_size"],
            shuffle=True,
            num_workers=0,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_data,
            batch_size=config["student"]["batch_size"],
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )

        # Quick retrain (fewer epochs for fine-tuning)
        retrain_epochs = 200 if not al_config.get("retrain_from_scratch", False) else config["student"]["max_epochs"]

        optimizer = torch.optim.AdamW(
            student.parameters(),
            lr=config["student"]["lr"] * 0.1,  # Lower LR for fine-tuning
            weight_decay=config["student"]["weight_decay"],
        )

        best_val_fmae = float("inf")
        patience = 0

        for epoch in range(1, retrain_epochs + 1):
            train_metrics = train_one_epoch(
                student, optimizer, train_loader, device,
                config["student"]["force_weight"],
                config["student"]["energy_weight"],
            )

            if epoch % 20 == 0:
                val_metrics = evaluate_model(student, val_loader, device)
                print(
                    f"  Epoch {epoch:4d} | "
                    f"F-MAE {val_metrics['force_mae']*1000:.2f} meV/Å | "
                    f"E-MAE {val_metrics['energy_mae']*1000:.2f} meV"
                )

                if val_metrics["force_mae"] < best_val_fmae:
                    best_val_fmae = val_metrics["force_mae"]
                    patience = 0
                    # Save
                    ckpt_dir = Path(config["student"]["checkpoint_dir"])
                    torch.save(
                        {
                            "epoch": epoch,
                            "loop": loop,
                            "model_state_dict": student.state_dict(),
                            "val_metrics": val_metrics,
                            "config": config,
                            "stats": stats,
                        },
                        ckpt_dir / f"best_model_loop{loop}.pt",
                    )
                else:
                    patience += 1
                    if patience >= 5:  # 5 * 20 = 100 epochs patience
                        print(f"  Early stopping at epoch {epoch}")
                        break

        print(f"\n  Loop {loop} best val F-MAE: {best_val_fmae*1000:.2f} meV/Å")

    # Final evaluation
    print(f"\n{'='*60}")
    print("Final evaluation after active learning")
    print(f"{'='*60}")

    # Load best model from last loop
    ckpt_dir = Path(config["student"]["checkpoint_dir"])
    ckpt = torch.load(ckpt_dir / f"best_model_loop{n_loops}.pt", weights_only=False)
    student.load_state_dict(ckpt["model_state_dict"])

    val_metrics = evaluate_model(student, val_loader, device)
    print("\nValidation metrics:")
    print_metrics(val_metrics, prefix="  ")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/ethanol.yaml")
    parser.add_argument("--n_loops", type=int, default=2)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device(config.get("device", "cuda") if torch.cuda.is_available() else "cpu")

    active_learning_loop(config, device, n_loops=args.n_loops)


if __name__ == "__main__":
    main()