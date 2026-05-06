"""
Evaluate distilled students properly.

Three evaluations:
1. Student vs Teacher labels (distillation quality) - how well did the student learn?
2. Teacher vs DFT labels (teacher quality) - how good is the teacher?
3. Student vs DFT labels (end-to-end quality) - how good is the student at real physics?

Usage:
    python training/evaluate.py --config configs/ethanol.yaml --student painn
    python training/evaluate.py --config configs/ethanol_mace2mace.yaml --student mace_small
"""

import os
import sys
import argparse
import yaml
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.download_md17 import MD17DataModule
from models.mace_teacher import MACETeacher
from nnp_utils.metrics import evaluate_model, print_metrics


def label_dataset_with_model(model, dataloader, device):
    """Run model on dataset and collect predictions."""
    model.eval()
    all_energy, all_force = [], []
    for batch in dataloader:
        batch = batch.to(device)
        with torch.enable_grad():
            out = model(batch, compute_force=True)
        all_energy.append(out["energy"].detach().cpu())
        all_force.append(out["force"].detach().cpu())
    return torch.cat(all_energy), torch.cat(all_force)


def compute_metrics(pred_e, true_e, pred_f, true_f):
    return {
        "energy_mae": (pred_e - true_e).abs().mean().item(),
        "energy_rmse": (pred_e - true_e).pow(2).mean().sqrt().item(),
        "force_mae": (pred_f - true_f).abs().mean().item(),
        "force_rmse": (pred_f - true_f).pow(2).mean().sqrt().item(),
        "force_cosine": torch.nn.functional.cosine_similarity(
            pred_f, true_f, dim=-1
        ).mean().item(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--student", type=str, required=True,
                        choices=["painn", "mace_small"],
                        help="Which student to evaluate")
    parser.add_argument("--n_test", type=int, default=5000,
                        help="Max test samples (full test set can be huge)")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # --- Load test data (DFT labels) ---
    dm = MD17DataModule(
        molecule=config["data"]["molecule"],
        data_dir=config["data"]["data_dir"],
        use_revised=config["data"].get("use_revised", False),
        n_train=config["data"]["n_train"],
        n_val=config["data"]["n_val"],
        cutoff=config["data"]["cutoff"],
        batch_size=64,
        seed=config["data"]["seed"],
    )
    _, val_data, test_data, stats = dm.load_splits()

    # Limit test size for speed
    if len(test_data) > args.n_test:
        test_data = test_data[:args.n_test]
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False)
    val_loader = DataLoader(val_data, batch_size=64, shuffle=False)
    print(f"Test set: {len(test_data)} configs")
    print(f"Val set: {len(val_data)} configs\n")

    # --- Load teacher ---
    print("Loading MACE teacher...")
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
    teacher_params = sum(p.numel() for p in teacher.parameters())
    print(f"  Teacher params: {teacher_params:,}\n")

    # --- Load student ---
    print(f"Loading {args.student} student...")
    if args.student == "painn":
        from models.painn_student import PaiNNStudent
        sc = config["student"]
        student = PaiNNStudent(
            num_interactions=sc["num_interactions"],
            hidden_dim=sc["hidden_dim"],
            num_basis=sc["num_basis"],
            cutoff=sc["cutoff"],
        ).to(device)
        student_ckpt = torch.load(
            Path(sc["checkpoint_dir"]) / "best_model.pt",
            weights_only=False, map_location=device,
        )
    elif args.student == "mace_small":
        from models.mace_small_student import MACESmallStudent
        sc = config["mace_student"]
        student = MACESmallStudent(
            num_interactions=sc["num_interactions"],
            hidden_dim=sc["hidden_dim"],
            max_ell=sc["max_ell"],
            num_basis=sc["num_basis"],
            cutoff=sc.get("cutoff", config["data"]["cutoff"]),
        ).to(device)
        student_ckpt = torch.load(
            Path(sc["checkpoint_dir"]) / "best_model.pt",
            weights_only=False, map_location=device,
        )

    student.set_energy_stats(stats["energy_mean"], stats["energy_std"], stats["n_atoms"])
    student.load_state_dict(student_ckpt["model_state_dict"])
    student.eval()
    student_params = sum(p.numel() for p in student.parameters())
    print(f"  Student params: {student_params:,}")
    print(f"  Compression ratio: {teacher_params / student_params:.1f}x\n")

    # --- Get predictions ---
    print("Running teacher on test set...")
    teacher_test_e, teacher_test_f = label_dataset_with_model(teacher, test_loader, device)

    print("Running student on test set...")
    student_test_e, student_test_f = label_dataset_with_model(student, test_loader, device)

    # Collect DFT ground truth
    dft_test_e, dft_test_f = [], []
    for batch in test_loader:
        dft_test_e.append(batch.energy)
        dft_test_f.append(batch.force)
    dft_test_e = torch.cat(dft_test_e)
    dft_test_f = torch.cat(dft_test_f)

    # --- Evaluation 1: Teacher vs DFT ---
    print(f"\n{'='*60}")
    print(f"1. TEACHER vs DFT (teacher quality)")
    print(f"{'='*60}")
    teacher_vs_dft = compute_metrics(teacher_test_e, dft_test_e, teacher_test_f, dft_test_f)
    print_metrics(teacher_vs_dft, prefix="  ")

    # --- Evaluation 2: Student vs Teacher ---
    print(f"\n{'='*60}")
    print(f"2. STUDENT vs TEACHER (distillation fidelity)")
    print(f"{'='*60}")
    student_vs_teacher = compute_metrics(student_test_e, teacher_test_e, student_test_f, teacher_test_f)
    print_metrics(student_vs_teacher, prefix="  ")

    # --- Evaluation 3: Student vs DFT ---
    print(f"\n{'='*60}")
    print(f"3. STUDENT vs DFT (end-to-end quality)")
    print(f"{'='*60}")
    student_vs_dft = compute_metrics(student_test_e, dft_test_e, student_test_f, dft_test_f)
    print_metrics(student_vs_dft, prefix="  ")

    # --- Summary table ---
    print(f"\n{'='*60}")
    print(f"SUMMARY: {config['data']['molecule']}")
    print(f"{'='*60}")
    print(f"{'Model':<20} {'Params':>10} {'vs DFT F-MAE':>15} {'vs DFT E-MAE':>15} {'vs Teacher F-MAE':>18}")
    print(f"{'-'*80}")
    print(f"{'MACE teacher':<20} {teacher_params:>10,} "
          f"{teacher_vs_dft['force_mae']*1000:>14.2f}  "
          f"{teacher_vs_dft['energy_mae']*1000:>14.2f}  "
          f"{'--':>17}")
    print(f"{args.student + ' student':<20} {student_params:>10,} "
          f"{student_vs_dft['force_mae']*1000:>14.2f}  "
          f"{student_vs_dft['energy_mae']*1000:>14.2f}  "
          f"{student_vs_teacher['force_mae']*1000:>17.2f}")

    # --- Speedup estimate ---
    print(f"\n  Compression: {teacher_params/student_params:.1f}x fewer parameters")
    print(f"  Distillation fidelity: {student_vs_teacher['force_mae']*1000:.2f} meV/Ang (student deviation from teacher)")
    print(f"  Teacher error floor: {teacher_vs_dft['force_mae']*1000:.2f} meV/Ang (teacher deviation from DFT)")

    # Save
    results = {
        "teacher_vs_dft": teacher_vs_dft,
        "student_vs_teacher": student_vs_teacher,
        "student_vs_dft": student_vs_dft,
        "teacher_params": teacher_params,
        "student_params": student_params,
        "student_type": args.student,
    }
    save_path = Path(sc["checkpoint_dir"]) / "evaluation_results.pt"
    torch.save(results, save_path)
    print(f"\nResults saved to {save_path}")


if __name__ == "__main__":
    main()