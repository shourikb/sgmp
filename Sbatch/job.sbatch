#!/bin/bash
#SBATCH -J sgmp
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 03:00:00
#SBATCH -o logs/%x-%j.out

if [ -n "${CONDA_ENV:-}" ]; then
  command -v conda >/dev/null || module load anaconda3
  eval "$(conda shell.bash hook)"
  conda activate "$CONDA_ENV"
fi
: "${DATA:?set DATA to the ImageNet-100 root written by prepare_data.py}"
export DATA
export PYTHONUNBUFFERED=1
cd "${SLURM_SUBMIT_DIR:-.}"
nvidia-smi --query-gpu=name --format=csv
"$@"
