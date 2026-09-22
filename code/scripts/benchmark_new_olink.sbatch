#!/bin/bash
#SBATCH --job-name=ctsnn-olink
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=72:00:00
#SBATCH --output=/home/zliu/ctsnn/logs/benchmark_olink_%j.out
#SBATCH --error=/home/zliu/ctsnn/logs/benchmark_olink_%j.err
set -euo pipefail
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate base
cd /home/zliu/ctsnn/project
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
RESULTS=/home/zliu/ctsnn/results/new_datasets/olink_50
if [ -e "$RESULTS" ]; then
  echo "Refusing to overwrite existing results: $RESULTS" >&2
  exit 2
fi
mkdir -p "$RESULTS/figures"
python src/run_expanded_benchmark.py \
  --datasets OLINK_COVID \
  --methods Lasso ElasticNet StabilitySelection Stabl-RP LongGroupLasso LLSS \
    RandomForest MutualInfo Boruta mRMR SNN-FS OutcomeSNN-FS \
  --repeats 50 --budgets 3 5 10 --output-dir "$RESULTS"
python src/summarize_benchmark_results.py "$RESULTS"
python src/make_benchmark_figures.py "$RESULTS" --output-dir "$RESULTS/figures"
