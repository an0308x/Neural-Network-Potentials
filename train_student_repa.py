"""
Phase 3+REPA: Train PaiNN Student with Representation Alignment

Loss: w_energy * L_energy + w_force * L_force + lambda(t) * L_repa
Lambda: linear warmup from 0 to repa_weight over repa_warmup epochs.

Usage:
    python training/train_student_repa.py --config configs/ethanol.yaml --repa_weight 0.1
    python training/train_student_repa.py --config configs/ethanol.yaml --repa_weight 0.2 --repa_warmup 50
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

from models.mace_teacher import MACETeacher
from models.painn_student import PaiNNStudent, count_parameters
from models.repa import REPALoss
from nnp_utils.metrics import evaluate_model, print_metrics


def get_repa_lambda(epoch, repa_weight, repa_warmup):
    if repa_warmup <= 0:
        return repa_weight
    return repa_weight * min(1.0, epoch / repa_warmup)


def train_one_epoch_repa(student, teacher, repa_loss_fn, optimizer, train_loader,
                          device, force_weight, energy_weight, repa_lambda):
    student.train()
    teacher.eval()

    total_loss = 0
    total_force_loss = 0
    total_energy_loss = 0
    total_repa_loss = 0
    total_diag = {"std_student": 0.0, "std_teacher": 0.0, "cos_sim": 0.0}
    n_batches = 0

    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        student_out = student(batch, compute_force=True)
        force_loss = nn.functional.mse_loss(student_out["force"], batch.force)
        energy_loss = nn.functional.mse_loss(student_out["energy"], batch.energy)

        with torch.no_grad():
            teacher_reps = teacher.get_representations(batch)
        student_reps = student.get_representations(batch)

        repa_loss, diag = repa_loss_fn(teacher_reps, student_reps)

        loss = (energy_weight * energy_loss
                + force_weight * force_loss
                + repa_lambda * repa_loss)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(student.parameters()) + list(repa_loss_fn.parameters()),
            max_norm=10.0,
        )
        optimizer.step()

        total_loss += loss.item()
        total_force_loss += force_loss.item()
        total_energy_loss += energy_loss.item()
        total_repa_loss += repa_loss.item()
        for k in total_diag:
            total_diag[k] += diag[k]
        n_batches += 1

    return {
        "loss": total_loss / n_batches,
        "force_loss": total_force_loss / n_batches,
        "energy_loss": total_energy_loss / n_batches,
        "repa_loss": total_repa_loss / n_batches,
        "std_student": total_diag["std_student"] / n_batches,
        "std_teacher": total_diag["std_teacher"] / n_batches,
        "cos_sim": total_diag["cos_sim"] / n_batches,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/ethanol.yaml")
    parser.add_argument("--repa_weight", type=float, default=0.1,
                        help="Max REPA lambda (after warmup)")
    parser.add_argument("--repa_warmup", type=int, default=50,
                        help="Epochs to linearly warm up lambda from 0")
    parser.add_argument("--align_dim", type=int, default=64,
                        help="Unused (kept for compat), projection goes to teacher_scalar_dim")
    parser.add_argument("--repa_loss_type", type=str, default="cosine",
                        choices=["cosine", "mse"])
    parser.add_argument("--use_original_data", action="store_true")
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
            run_name = (args.wandb_name or
                        f"painn_repa_lam{args.repa_weight}_warm{args.repa_warmup}_{config['data']['molecule']}")
            wandb.init(project=args.wandb_project, name=run_name,
                       config={
                           "phase": "student_repa", "model": "PaiNN+REPA",
                           "data_source": data_source,
                           "repa_weight": args.repa_weight,
                           "repa_warmup": args.repa_warmup,
                           "repa_loss_type": args.repa_loss_type,
                           **config,
                       },
                       tags=["student", "painn", "repa", config["data"]["molecule"], data_source])
            print(f"Wandb: {wandb.run.url}")
        except Exception as e:
            print(f"wandb failed: {e}")
            use_wandb = False

    # --- Data ---
    sc = config["student"]

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
        assert (distill_dir / "distillation_train.pt").exists(), \
            "Distillation dataset not found. Run data/distillation_dataset.py first."

        train_data = torch.load(distill_dir / "distillation_train.pt", weights_only=False)
        val_data = torch.load(distill_dir / "distillation_val.pt", weights_only=False)
        stats = torch.load(distill_dir / "stats.pt", weights_only=False)

        train_loader = DataLoader(train_data, batch_size=sc["batch_size"],
                                  shuffle=True, num_workers=0, pin_memory=True)
        val_loader = DataLoader(val_data, batch_size=sc["batch_size"],
                                shuffle=False, num_workers=0, pin_memory=True)

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

    # --- Teacher (frozen) ---
    print("\nLoading MACE teacher (frozen)...")
    teacher_ckpt = torch.load(
        Path(config["teacher"]["checkpoint_dir"]) / "best_model.pt",
        weights_only=False, map_location=device,
    )
    teacher = MACETeacher(
        num_interactions=config["teacher"]["num_interactions"],
        hidden_dim=config["teacher"]["hidden_dim"],
        max_ell=config["teacher"]["max_ell"],
        num_basis=config["teacher"]["num_basis"],
        cutoff=config["data"]["cutoff"],
    ).to(device)
    teacher.set_energy_stats(stats["energy_mean"], stats["energy_std"], stats["n_atoms"])
    teacher.load_state_dict(teacher_ckpt["model_state_dict"])
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    print(f"  Teacher: {sum(p.numel() for p in teacher.parameters()):,} params (frozen)")

    # --- Student ---
    student = PaiNNStudent(
        num_interactions=sc["num_interactions"],
        hidden_dim=sc["hidden_dim"],
        num_basis=sc["num_basis"],
        cutoff=sc["cutoff"],
    ).to(device)
    student.set_energy_stats(stats["energy_mean"], stats["energy_std"], stats["n_atoms"])
    n_student_params = count_parameters(student)
    print(f"  Student: {n_student_params:,} params")

    # --- REPA ---
    n_teacher_reps = config["teacher"]["num_interactions"] + 1
    n_student_reps = sc["num_interactions"] + 1

    repa_loss_fn = REPALoss(
        teacher_irreps=str(teacher.irreps_node),
        student_hidden_dim=sc["hidden_dim"],
        n_teacher_layers=n_teacher_reps,
        n_student_layers=n_student_reps,
        loss_type=args.repa_loss_type,
    ).to(device)

    n_repa_params = sum(p.numel() for p in repa_loss_fn.parameters() if p.requires_grad)
    print(f"  REPA projectors: {n_repa_params:,} params (linear, no bias)")
    print(f"  REPA lambda_max: {args.repa_weight}, warmup: {args.repa_warmup} epochs")

    if use_wandb:
        import wandb
        wandb.log({"model/student_params": n_student_params, "model/repa_params": n_repa_params})

    # --- Optimizer ---
    optimizer = torch.optim.AdamW(
        list(student.parameters()) + list(repa_loss_fn.parameters()),
        lr=sc["lr"], weight_decay=sc["weight_decay"], amsgrad=True,
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=sc["max_epochs"] - sc.get("warmup_epochs", 10),
        eta_min=sc["min_lr"],
    )

    warmup_epochs = sc.get("warmup_epochs", 10)
    force_weight = sc["force_weight"]
    energy_weight = sc["energy_weight"]

    ckpt_dir = Path(sc["checkpoint_dir"].replace("student", "student_repa"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_val_force_mae = float("inf")
    patience_counter = 0

    print(f"\n{'='*60}")
    print(f"Starting PaiNN+REPA training")
    print(f"  Loss = {energy_weight}*E + {force_weight}*F + lam(t)*REPA")
    print(f"  lam: 0 -> {args.repa_weight} over {args.repa_warmup} epochs")
    print(f"{'='*60}\n")

    for epoch in range(1, sc["max_epochs"] + 1):
        t0 = time.time()

        if epoch <= warmup_epochs:
            for pg in optimizer.param_groups:
                pg["lr"] = sc["lr"] * epoch / warmup_epochs

        repa_lambda = get_repa_lambda(epoch, args.repa_weight, args.repa_warmup)

        train_metrics = train_one_epoch_repa(
            student, teacher, repa_loss_fn, optimizer, train_loader,
            device, force_weight, energy_weight, repa_lambda,
        )
        val_metrics = evaluate_model(student, val_loader, device)

        if epoch > warmup_epochs:
            scheduler.step()

        dt = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]

        if use_wandb:
            import wandb
            wandb.log({
                "epoch": epoch, "lr": lr, "repa_lambda": repa_lambda,
                "train/loss": train_metrics["loss"],
                "train/force_loss": train_metrics["force_loss"],
                "train/energy_loss": train_metrics["energy_loss"],
                "train/repa_loss": train_metrics["repa_loss"],
                "train/repa_cos_sim": train_metrics["cos_sim"],
                "train/std_student": train_metrics["std_student"],
                "train/std_teacher": train_metrics["std_teacher"],
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
                  f"REPA {train_metrics['repa_loss']:.4f} | "
                  f"cos {train_metrics['cos_sim']:.4f} | "
                  f"std_s {train_metrics['std_student']:.4f} std_t {train_metrics['std_teacher']:.4f} | "
                  f"lam {repa_lambda:.4f} | "
                  f"Val F-MAE {val_metrics['force_mae']*1000:.2f} | "
                  f"LR {lr:.2e} | {dt:.1f}s")

        if val_metrics["force_mae"] < best_val_force_mae:
            best_val_force_mae = val_metrics["force_mae"]
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": student.state_dict(),
                "repa_state_dict": repa_loss_fn.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_metrics": val_metrics, "config": config, "stats": stats,
                "repa_weight": args.repa_weight,
            }, ckpt_dir / "best_model.pt")
            if use_wandb:
                import wandb
                wandb.run.summary["best_val_force_mae_meV"] = best_val_force_mae * 1000
                wandb.run.summary["best_epoch"] = epoch
        else:
            patience_counter += 1

        if epoch % 100 == 0:
            torch.save({"epoch": epoch, "model_state_dict": student.state_dict(),
                         "repa_state_dict": repa_loss_fn.state_dict(),
                         "optimizer_state_dict": optimizer.state_dict()},
                        ckpt_dir / f"checkpoint_epoch{epoch}.pt")

        if patience_counter >= sc["patience"]:
            print(f"\nEarly stopping at epoch {epoch}")
            break

    print(f"\n{'='*60}")
    print(f"Training complete! Best val force MAE: {best_val_force_mae*1000:.2f} meV/Ang")
    print(f"{'='*60}\n")

    ckpt = torch.load(ckpt_dir / "best_model.pt", weights_only=False)
    student.load_state_dict(ckpt["model_state_dict"])

    print("Test set evaluation:")
    test_metrics = evaluate_model(student, test_loader, device)
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