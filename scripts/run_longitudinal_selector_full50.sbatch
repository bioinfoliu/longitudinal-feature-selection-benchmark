#!/usr/bin/env bash
# 14 datasets x 50 participant-level repeats. Submit with METHOD and SLUG.
#SBATCH --array=1-700%8
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --output=/home/zliu/ctsnn/logs/LONGFS_%A_%a.log

set -euo pipefail
project_dir="${CTSNN_PROJECT:-/home/zliu/ctsnn/project}"
results_root="${CTSNN_RESULTS:-/home/zliu/ctsnn/results/main_benchmark/longitudinal_extensions}"
method="${METHOD:?Submit with METHOD}"
slug="${SLUG:?Submit with SLUG}"
task_id="${SLURM_ARRAY_TASK_ID:?Missing Slurm array task ID}"
datasets=(PE PE_LOPE_MATCHED FPG FPG_KARE KARE_STATUS IGG IGM TRA TRB OLINK_COVID BRIST1D GSE41848_MS GSE41849_MS GSE48023_H1N1)
dataset_count="${#datasets[@]}"
repeat_id=$(( (task_id - 1) / dataset_count + 1 ))
dataset_index=$(( (task_id - 1) % dataset_count ))
dataset="${datasets[$dataset_index]}"
output_dir="${results_root}/${slug}/${dataset}/repeat_${repeat_id}"
mkdir -p "${output_dir}"
export CTSNN_LONGFS_RSCRIPT="${CTSNN_LONGFS_RSCRIPT:-/home/zliu/miniconda3/envs/ctsnn-longfs-r/bin/Rscript}"
cd "${project_dir}"
python src/run_expanded_benchmark.py \
  --datasets "${dataset}" \
  --repeats 1 \
  --repeat-start "${repeat_id}" \
  --methods "${method}" \
  --budgets 3 5 10 \
  --output-dir "${output_dir}"
