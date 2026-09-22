#!/bin/bash
#SBATCH --job-name=geo-smoke
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=72:00:00
#SBATCH --output=/home/zliu/ctsnn/logs/benchmark_geo_smoke_%j.out
#SBATCH --error=/home/zliu/ctsnn/logs/benchmark_geo_smoke_%j.err
set -euo pipefail
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate base
cd /home/zliu/ctsnn/project
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export CTSNN_LONGFS_RSCRIPT=/home/zliu/miniconda3/envs/ctsnn-longfs-r/bin/Rscript
RESULTS=/home/zliu/ctsnn/results/validation/new_geo_smoke
if [ -e "$RESULTS" ]; then
  echo "Refusing to overwrite existing results: $RESULTS" >&2
  exit 2
fi
mkdir -p "$RESULTS/figures"
python src/run_expanded_benchmark.py \
  --datasets GSE41848_MS GSE41849_MS GSE48023_H1N1 \
  --methods Lasso ElasticNet StabilitySelection Stabl-RP LongGroupLasso LLSS \
    RandomForest MutualInfo Boruta mRMR SNN-FS OutcomeSNN-FS geeVerse \
  --repeats 1 --budgets 3 5 10 --output-dir "$RESULTS"
