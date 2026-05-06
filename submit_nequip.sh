#!/bin/bash
#SBATCH --job-name=nnp_nequip
#SBATCH --account=system
#SBATCH --partition=gpu4_medium
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/nequip_%j.out
#SBATCH --error=logs/nequip_%j.err

export PYTHONNOUSERSITE=1
module load anaconda3/gpu/new
source /gpfs/share/apps/anaconda3/gpu/new/etc/profile.d/conda.sh
export CONDA_ENVS_PATH=/gpfs/scratch/$USER/conda_envs
conda activate nnp_distill

cd /gpfs/scratch/$USER/projects/NNP
export PYTHONPATH=$(pwd):$PYTHONPATH
mkdir -p logs checkpoints/nequip_student checkpoints/nequip_student_repa

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID | Node: $SLURM_NODELIST"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Start: $(date)"
echo "=========================================="

# --- NequIP baseline ---
echo ""
echo ">>> NequIP baseline distillation..."
python -u training/train_nequip_student.py \
    --config configs/ethanol_nequip.yaml

if [ $? -ne 0 ]; then
    echo "NequIP baseline FAILED"
    exit 1
fi

# --- NequIP + REPA ---
echo ""
echo ">>> NequIP + REPA (lam=0.1, warmup=50)..."
python -u training/train_nequip_student.py \
    --config configs/ethanol_nequip.yaml \
    --repa --repa_weight 0.1 --repa_warmup 50

if [ $? -ne 0 ]; then
    echo "NequIP REPA FAILED"
    exit 1
fi

echo ""
echo ">>> All NequIP experiments done at $(date)"
echo "=========================================="