#!/bin/bash
#SBATCH --job-name=nnp_student
#SBATCH --account=system
#SBATCH --partition=gpu4_medium
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/student_%j.out
#SBATCH --error=logs/student_%j.err

export PYTHONNOUSERSITE=1
module load anaconda3/gpu/new
source /gpfs/share/apps/anaconda3/gpu/new/etc/profile.d/conda.sh
export CONDA_ENVS_PATH=/gpfs/scratch/$USER/conda_envs
conda activate nnp_distill

cd /gpfs/scratch/$USER/projects/NNP
export PYTHONPATH=$(pwd):$PYTHONPATH
mkdir -p logs

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID | Node: $SLURM_NODELIST"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Start: $(date)"
echo "=========================================="

echo ""
echo ">>> Phase 2: Generating distillation dataset..."
python -u data/distillation_dataset.py --config configs/ethanol.yaml

echo ""
echo ">>> Phase 3: Training PaiNN student (baseline)..."
python -u training/train_student.py --config configs/ethanol.yaml

echo ""
echo ">>> Done at $(date)"
EOF

sbatch submit_phase2.sh