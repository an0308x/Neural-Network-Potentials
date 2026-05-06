#!/bin/bash
#SBATCH --job-name=nnp_repa
#SBATCH --account=system
#SBATCH --partition=gpu4_medium
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/repa_%j.out
#SBATCH --error=logs/repa_%j.err

export PYTHONNOUSERSITE=1
module load anaconda3/gpu/new
source /gpfs/share/apps/anaconda3/gpu/new/etc/profile.d/conda.sh
export CONDA_ENVS_PATH=/gpfs/scratch/$USER/conda_envs
conda activate nnp_distill

cd /gpfs/scratch/$USER/projects/NNP
export PYTHONPATH=$(pwd):$PYTHONPATH
mkdir -p logs checkpoints/student_repa

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID | Node: $SLURM_NODELIST"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Start: $(date)"
echo "=========================================="

echo ""
echo ">>> PaiNN + REPA (linear proj, lam=0.1, warmup=50)..."
python -u training/train_student_repa.py \
    --config configs/ethanol.yaml \
    --repa_weight 0.1 \
    --repa_warmup 50 \
    --repa_loss_type cosine

if [ $? -ne 0 ]; then
    echo "REPA training FAILED"
    exit 1
fi

echo ""
echo ">>> Done at $(date)"
echo "=========================================="