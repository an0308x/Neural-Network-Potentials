"""
Train NequIP student via distillation, with optional REPA.

Usage:
    # Baseline (no REPA)
    python training/train_nequip_student.py --config configs/ethanol_nequip.yaml

    # With REPA
    python training/train_nequip_student.py --config configs/ethanol_nequip.yaml \
        --repa --repa_weight 0.1 --repa_warmup 50
"""

import os
import sys
import argparse
import yaml
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.loader import DataLoader
from e3nn import o3

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.mace_teacher import MACETeacher
from models.nequip_student import NequIPStudent, count_parameters
from nnp_utils.metrics import evaluate_model, print_metrics


# ──────────────────────────────────────────────────────────
# REPA for NequIP (equivariant irreps -> align L=0 scalars)
# ──────────────────────────────────────────────────────────

class NequIPREPA(nn.Module):
    """REPA for MACE teacher -> NequIP student. Both produce equivariant irreps."""

    def __init__(self, teacher_irreps, student_irreps, n_teacher_layers, n_student_layers):
        super().__init__()
        self.teacher_irreps = o3.Irreps(teacher_irreps)
        self.student_irreps = o3.Irreps(student_irreps)

        self.teacher_scalar_dim = sum(mul * ir.dim for mul, ir in self.teacher_irreps if ir.l == 0)
        self.student_scalar_dim = sum(mul * ir.dim for mul, ir in self.student_irreps if ir.l == 0)

        self._teacher_slices = self._get_scalar_slices(self.teacher_irreps)
        self._student_slices = self._get_scalar_slices(self.student_irreps)

        n_align = min(n_teacher_layers - 1, n_student_layers - 1)
        self.teacher_layers = list(range(n_teacher_layers - n_align, n_teacher_layers))
        self.student_layers = list(range(n_student_layers - n_align, n_student_layers))
        self.n_pairs = n_align

        print(f"  REPA (MACE→NequIP): aligning {n_align} layer pairs")
        print(f"    teacher {self.teacher_layers} <-> student {self.student_layers}")
        print(f"    teacher scalar dim={self.teacher_scalar_dim}, student scalar dim={self.student_scalar_dim}")

        self.projections = nn.ModuleList([
            nn.Linear(self.student_scalar_dim, self.teacher_scalar_dim, bias=False)
            for _ in range(n_align)
        ])

    @staticmethod
    def _get_scalar_slices(irreps):
        slices = []
        offset = 0
        for mul, ir in irreps:
            dim = mul * ir.dim
            if ir.l == 0:
                slices.append((offset, offset + dim))
            offset += dim
        return slices

    def _extract_scalars(self, rep, slices):
        parts = [rep[:, s0:s1] for s0, s1 in slices]
        return torch.cat(parts, dim=-1)

    def forward(self, teacher_reps, student_reps):
        total_loss = 0.0
        diag = {"std_student": 0.0, "std_teacher": 0.0, "cos_sim": 0.0}

        for i, (t_idx, s_idx) in enumerate(zip(self.teacher_layers, self.student_layers)):
            t_scalar = self._extract_scalars(teacher_reps[t_idx], self._teacher_slices).detach()
            s_scalar = self._extract_scalars(student_reps[s_idx], self._student_slices)
            s_proj = self.projections[i](s_scalar)

            diag["std_student"] += s_proj.std(dim=0).mean().item()
            diag["std_teacher"] += t_scalar.std(dim=0).mean().item()

            t_norm = F.normalize(t_scalar, dim=-1)
            s_norm = F.normalize(s_proj, dim=-1)

            cos_sim = (t_norm * s_norm).sum(dim=-1)
            diag["cos_sim"] += cos_sim.mean().item()
            total_loss = total_loss + (1.0 - cos_sim).mean()

        n = max(self.n_pairs, 1)
        diag = {k: v / n for k, v in diag.items()}
        return total_loss / n, diag


# ──────────────────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────────────────

def get_repa_lambda(epoch, repa_weight, repa_warmup):
    if repa_warmup <= 0:
        return repa_weight
    return repa_weight * min(1.0, epoch / repa_warmup)


def train_one_epoch(student, optimizer, train_loader, device, force_weight, energy_weight,
                    teacher=None, repa_loss_fn=None, repa_lambda=0.0):
    student.train()
    if teacher:
        teacher.eval()

    totals = {"loss": 0, "force_loss": 0, "energy_loss": 0, "repa_loss": 0}
    diag_totals = {"std_student": 0, "std_teacher": 0, "cos_sim": 0}
    n_batches = 0

    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        out = student(batch, compute_force=True)
        force_loss = nn.functional.mse_loss(out["force"], batch.force)
        energy_loss = nn.functional.mse_loss(out["energy"], batch.energy)
        loss = energy_weight * energy_loss + force_weight * force_loss

        repa_loss_val = 0.0
        if teacher and repa_loss_fn and repa_lambda > 0:
            with torch.no_grad():
                teacher_reps = teacher.get_representations(batch)
            student_reps = student.get_representations(batch)
            repa_loss, diag = repa_loss_fn(teacher_reps, student_reps)
            loss = loss + repa_lambda * repa_loss
            repa_loss_val = repa_loss.item()
            for k in diag_totals:
                diag_totals[k] += diag[k]

        loss.backward()

        params = list(student.parameters())
        if repa_loss_fn:
            params += list(repa_loss_fn.parameters())
        torch.nn.utils.clip_grad_norm_(params, max_norm=10.0)
        optimizer.step()

        totals["loss"] += loss.item()
        totals["force_loss"] += force_loss.item()
        totals["energy_loss"] += energy_loss.item()
        totals["repa_loss"] += repa_loss_val
        n_batches += 1

    result = {k: v / n_batches for k, v in totals.items()}
    result.update({k: v / n_batches for k, v in diag_totals.items()})
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/ethanol_nequip.yaml")
    parser.add_argument("--repa", action="store_true", help="Enable REPA")
    parser.add_argument("--repa_weight", type=float, default=0.1)
    parser.add_argument("--repa_warmup", type=int, default=50)
    parser.add_argument("--use_original_data", action="store_true")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="nnp-distillation")
    parser.add_argument("--wandb-name", type=str, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device(config.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    mode = "repa" if args.repa else "baseline"

    # --- Wandb ---
    use_wandb = not args.no_wandb
    if use_wandb:
        try:
            import wandb
            data_source = "original" if args.use_original_data else "distillation"
            if args.wandb_name:
                run_name = args.wandb_name
            elif args.repa:
                run_name = f"nequip_repa_lam{args.repa_weight}_{config['data']['molecule']}"
            else:
                run_name = f"nequip_baseline_{config['data']['molecule']}"
            tags = ["student", "nequip", config["data"]["molecule"], data_source, mode]
            if args.repa:
                tags.append("repa")
            wandb.init(project=args.wandb_project, name=run_name,
                       config={"model": f"NequIP+{mode}", "repa": args.repa,
                               "repa_weight": args.repa_weight if args.repa else 0,
                               **config},
                       tags=tags)
            print(f"Wandb: {wandb.run.url}")
        except Exception as e:
            print(f"wandb failed: {e}")
            use_wandb = False

    # --- Data ---
    sc = config["nequip_student"]

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
    else:
        distill_dir = Path(config["distillation"]["output_dir"])
        assert (distill_dir / "distillation_train.pt").exists()

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

    # --- Teacher (only needed for REPA) ---
    teacher = None
    repa_loss_fn = None

    if args.repa:
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
    student = NequIPStudent(
        num_layers=sc["num_layers"],
        hidden_dim=sc["hidden_dim"],
        max_ell=sc["max_ell"],
        num_basis=sc["num_basis"],
        cutoff=sc.get("cutoff", config["data"]["cutoff"]),
    ).to(device)
    student.set_energy_stats(stats["energy_mean"], stats["energy_std"], stats["n_atoms"])
    n_params = count_parameters(student)
    print(f"  NequIP Student: {n_params:,} params")

    # --- REPA ---
    if args.repa:
        n_teacher_reps = config["teacher"]["num_interactions"] + 1
        n_student_reps = sc["num_layers"] + 1

        repa_loss_fn = NequIPREPA(
            teacher_irreps=str(teacher.irreps_node),
            student_irreps=str(student.irreps_node),
            n_teacher_layers=n_teacher_reps,
            n_student_layers=n_student_reps,
        ).to(device)

        n_repa = sum(p.numel() for p in repa_loss_fn.parameters() if p.requires_grad)
        print(f"  REPA: {n_repa:,} params, lam_max={args.repa_weight}, warmup={args.repa_warmup}")

    # --- Optimizer ---
    opt_params = list(student.parameters())
    if repa_loss_fn:
        opt_params += list(repa_loss_fn.parameters())
    optimizer = torch.optim.AdamW(opt_params, lr=sc["lr"],
                                   weight_decay=sc["weight_decay"], amsgrad=True)
    scheduler = CosineAnnealingLR(optimizer,
        T_max=sc["max_epochs"] - sc.get("warmup_epochs", 10), eta_min=sc["min_lr"])

    warmup_epochs = sc.get("warmup_epochs", 10)
    force_weight = sc["force_weight"]
    energy_weight = sc["energy_weight"]

    suffix = "_repa" if args.repa else ""
    ckpt_dir = Path(sc["checkpoint_dir"] + suffix)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_val_force_mae = float("inf")
    patience_counter = 0

    print(f"\n{'='*60}")
    print(f"Starting NequIP {mode} training")
    if args.repa:
        print(f"  Loss = {energy_weight}*E + {force_weight}*F + lam(t)*REPA")
        print(f"  lam: 0 -> {args.repa_weight} over {args.repa_warmup} epochs")
    else:
        print(f"  Loss = {energy_weight}*E + {force_weight}*F")
    print(f"{'='*60}\n")

    for epoch in range(1, sc["max_epochs"] + 1):
        t0 = time.time()

        if epoch <= warmup_epochs:
            for pg in optimizer.param_groups:
                pg["lr"] = sc["lr"] * epoch / warmup_epochs

        repa_lambda = get_repa_lambda(epoch, args.repa_weight, args.repa_warmup) if args.repa else 0.0

        metrics = train_one_epoch(
            student, optimizer, train_loader, device, force_weight, energy_weight,
            teacher=teacher, repa_loss_fn=repa_loss_fn, repa_lambda=repa_lambda,
        )
        val_metrics = evaluate_model(student, val_loader, device)

        if epoch > warmup_epochs:
            scheduler.step()

        dt = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]

        if use_wandb:
            import wandb
            log = {
                "epoch": epoch, "lr": lr,
                "train/loss": metrics["loss"],
                "train/force_loss": metrics["force_loss"],
                "train/energy_loss": metrics["energy_loss"],
                "val/force_mae_meV": val_metrics["force_mae"] * 1000,
                "val/energy_mae_meV": val_metrics["energy_mae"] * 1000,
                "val/force_cosine": val_metrics["force_cosine"],
                "time/epoch_seconds": dt,
            }
            if args.repa:
                log.update({
                    "repa_lambda": repa_lambda,
                    "train/repa_loss": metrics["repa_loss"],
                    "train/repa_cos_sim": metrics["cos_sim"],
                    "train/std_student": metrics["std_student"],
                    "train/std_teacher": metrics["std_teacher"],
                })
            wandb.log(log)

        if epoch % config.get("log_every", 10) == 0 or epoch == 1:
            repa_str = ""
            if args.repa:
                repa_str = (f"REPA {metrics['repa_loss']:.4f} | "
                           f"cos {metrics['cos_sim']:.4f} | "
                           f"std_s {metrics['std_student']:.4f} | "
                           f"lam {repa_lambda:.4f} | ")
            print(f"Epoch {epoch:4d} | Loss {metrics['loss']:.6f} | "
                  f"F-loss {metrics['force_loss']:.6f} | "
                  f"{repa_str}"
                  f"Val F-MAE {val_metrics['force_mae']*1000:.2f} | "
                  f"LR {lr:.2e} | {dt:.1f}s")

        if val_metrics["force_mae"] < best_val_force_mae:
            best_val_force_mae = val_metrics["force_mae"]
            patience_counter = 0
            save_dict = {
                "epoch": epoch,
                "model_state_dict": student.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_metrics": val_metrics, "config": config, "stats": stats,
            }
            if repa_loss_fn:
                save_dict["repa_state_dict"] = repa_loss_fn.state_dict()
                save_dict["repa_weight"] = args.repa_weight
            torch.save(save_dict, ckpt_dir / "best_model.pt")
            if use_wandb:
                import wandb
                wandb.run.summary["best_val_force_mae_meV"] = best_val_force_mae * 1000
                wandb.run.summary["best_epoch"] = epoch
        else:
            patience_counter += 1

        if epoch % 100 == 0:
            torch.save({"epoch": epoch, "model_state_dict": student.state_dict()},
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
        wandb.run.summary.update({"test_force_mae_meV": test_metrics["force_mae"] * 1000})
        wandb.finish()
    torch.save(test_metrics, ckpt_dir / "test_metrics.pt")


if __name__ == "__main__":
    main()