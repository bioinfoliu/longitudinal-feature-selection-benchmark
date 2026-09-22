#!/bin/bash
#SBATCH --job-name=ctsnn-bench
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=72:00:00
#SBATCH --output=/home/zliu/ctsnn/logs/benchmark_full_%j.out
#SBATCH --error=/home/zliu/ctsnn/logs/benchmark_full_%j.err

set -euo pipefail
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate base
PROJECT=/home/zliu/ctsnn/project
cd "$PROJECT"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

RESULTS=/home/zliu/ctsnn/results/main_benchmark/benchmark_full50
rm -rf "$RESULTS"
mkdir -p "$RESULTS"

python src/run_expanded_benchmark.py \
  --datasets PE PE_LOPE_MATCHED FPG FPG_KARE KARE_STATUS IGG IGM TRA TRB \
  --methods Lasso ElasticNet StabilitySelection Stabl-RP LongGroupLasso LLSS \
    RandomForest MutualInfo Boruta mRMR SNN-FS OutcomeSNN-FS \
  --repeats 50 --budgets 3 5 10 --output-dir "$RESULTS"

python src/make_benchmark_figures.py "$RESULTS" --output-dir "$RESULTS/figures"
