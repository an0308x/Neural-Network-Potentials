#!/bin/bash
#SBATCH --job-name=nnp_repa
#SBATCH --account=system
#SBATCH --partition=gl40s_short
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=100G
#SBATCH --time=03-00:00:00
#SBATCH --mem=32G
#SBATCH --output=logs/teacher_%j.out
#SBATCH --error=logs/teacher_%j.err

# === Environment setup ===
export PYTHONNOUSERSITE=1
module load anaconda3/gpu/new
source /gpfs/share/apps/anaconda3/gpu/2023.09/etc/profile.d/conda.sh
export CONDA_ENVS_PATH=/gpfs/scratch/$USER/conda_envs
conda activate nnp_distill

# === Project setup ===
cd /gpfs/scratch/$USER/projects/NNP
export PYTHONPATH=$(pwd):$PYTHONPATH
mkdir -p logs checkpoints/teacher checkpoints/student

# === Info ===
echo "=========================================="
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $SLURM_NODELIST"
echo "GPU:       $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Python:    $(which python)"
echo "Torch:     $(python -c 'import torch; print(torch.__version__, "cuda:", torch.cuda.is_available())')"
echo "Start:     $(date)"
echo "=========================================="

# === Phase 1: Train MACE teacher ===
echo ""
echo ">>> Phase 1: Training MACE teacher..."
python -u training/train_teacher.py --config configs/ethanol.yaml

echo ""
echo ">>> Done at $(date)"
echo "=========================================="