#!/bin/bash
#SBATCH --job-name=nnp_mace2mace
#SBATCH --account=system
#SBATCH --partition=gpu4_medium
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/mace2mace_%j.out
#SBATCH --error=logs/mace2mace_%j.err

export PYTHONNOUSERSITE=1
module load anaconda3/gpu/new
source /gpfs/share/apps/anaconda3/gpu/new/etc/profile.d/conda.sh
export CONDA_ENVS_PATH=/gpfs/scratch/$USER/conda_envs
conda activate nnp_distill

cd /gpfs/scratch/$USER/projects/NNP
export PYTHONPATH=$(pwd):$PYTHONPATH
mkdir -p logs checkpoints/mace_student

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID | Node: $SLURM_NODELIST"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Start: $(date)"
echo "=========================================="

echo ""
echo ">>> MACE-to-MACE distillation: training small MACE student..."
echo ">>> Using existing distillation dataset from data/distillation/"
python -u training/train_mace_student.py --config configs/ethanol_mace2mace.yaml
if [ $? -ne 0 ]; then
    echo "MACE student training FAILED"
    exit 1
fi

echo ""
echo ">>> Done at $(date)"
echo "=========================================="