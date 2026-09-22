#!/usr/bin/env bash
#SBATCH --job-name=gse48023-subj50
#SBATCH --partition=cpu
#SBATCH --array=1-50%5
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=/home/zliu/ctsnn/logs/gse48023_subject_50_%A_%a.log
set -euo pipefail
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate base
project_dir=/home/zliu/ctsnn/project
results_dir=/home/zliu/ctsnn/results/new_datasets/gse48023_subject_50
repeat_id="${SLURM_ARRAY_TASK_ID:?SLURM array task ID is required}"
mkdir -p "${results_dir}/repeat_${repeat_id}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export CTSNN_LONGFS_RSCRIPT=/home/zliu/miniconda3/envs/ctsnn-longfs-r/bin/Rscript
cd "${project_dir}"
python src/run_expanded_benchmark.py \
  --datasets GSE48023_H1N1 \
  --methods Lasso ElasticNet StabilitySelection Stabl-RP LongGroupLasso LLSS \
    RandomForest MutualInfo Boruta mRMR SNN-FS OutcomeSNN-FS geeVerse \
  --repeats 1 --repeat-start "${repeat_id}" --budgets 3 5 10 \
  --output-dir "${results_dir}/repeat_${repeat_id}"
