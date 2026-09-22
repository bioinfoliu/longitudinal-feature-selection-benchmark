#!/usr/bin/env bash
# Submit with: sbatch scripts/run_geeverse_full50.sbatch
# Each array task executes one distinct repeat and writes no shared files.
# Aggregate after completion with src/aggregate_repeat_runs.py.
#SBATCH --job-name=geeVerse-50
#SBATCH --array=1-50%5
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=/home/zliu/ctsnn/logs/geeVerse_50_%A_%a.log

set -euo pipefail
project_dir="${CTSNN_PROJECT:-/home/zliu/ctsnn/project}"
results_dir="${CTSNN_RESULTS:-/home/zliu/ctsnn/results/main_benchmark/longitudinal_extensions/geeverse_full50}"
repeat_id="${SLURM_ARRAY_TASK_ID:?SLURM array task ID is required}"
mkdir -p "${results_dir}/repeat_${repeat_id}"

export CTSNN_LONGFS_RSCRIPT="${CTSNN_LONGFS_RSCRIPT:-/home/zliu/miniconda3/envs/ctsnn-longfs-r/bin/Rscript}"
cd "${project_dir}"
python src/run_expanded_benchmark.py \
  --datasets FPG FPG_KARE KARE_STATUS BRIST1D IGG IGM TRA TRB \
  --repeats 1 \
  --repeat-start "${repeat_id}" \
  --methods geeVerse \
  --budgets 3 5 10 \
  --output-dir "${results_dir}/repeat_${repeat_id}"
