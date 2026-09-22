#!/bin/bash
#SBATCH --job-name=ctsnn-ext
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/home/zliu/ctsnn/logs/external_slurm_%j.out
#SBATCH --error=/home/zliu/ctsnn/logs/external_slurm_%j.err

set -euo pipefail
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate base
cd /home/zliu/ctsnn/project
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
python src/run_multicohort_external_validation.py

