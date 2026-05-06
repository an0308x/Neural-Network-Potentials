#!/bin/bash
#SBATCH --job-name=nnp_phase2_3
#SBATCH --account=system
#SBATCH --partition=gpu4_medium
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/phase2_3_%j.out
#SBATCH --error=logs/phase2_3_%j.err

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
if [ $? -ne 0 ]; then
    echo "Phase 2 FAILED"
    exit 1
fi

echo ""
echo ">>> Phase 3: Training PaiNN student (baseline)..."
python -u training/train_student.py --config configs/ethanol.yaml
if [ $? -ne 0 ]; then
    echo "Phase 3 FAILED"
    exit 1
fi

echo ""
echo ">>> All done at $(date)"
echo "=========================================="