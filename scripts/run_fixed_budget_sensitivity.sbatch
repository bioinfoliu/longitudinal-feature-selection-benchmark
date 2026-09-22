#!/usr/bin/env bash
# Submit once per method, for example:
# METHOD=Lasso SLUG=lasso sbatch scripts/run_fixed_budget_sensitivity.sbatch
# 14 tasks x 50 repeats x 3 fixed budgets = 2,100 independent jobs.
#SBATCH --job-name=LONGFS-FIXED
#SBATCH --array=1-2100%8
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --output=/home/zliu/ctsnn/logs/FIXED_%A_%a.log

set -euo pipefail
project_dir="${LONGFS_PROJECT:-/home/zliu/ctsnn/project}"
results_root="${LONGFS_RESULTS:-/home/zliu/ctsnn/results/fixed_budget}"
method="${METHOD:?Submit with METHOD}"
slug="${SLUG:?Submit with SLUG}"
task_id="${SLURM_ARRAY_TASK_ID:?Missing Slurm array task ID}"
datasets=(PE PE_LOPE_MATCHED FPG FPG_KARE KARE_STATUS IGG IGM TRA TRB OLINK_COVID BRIST1D GSE41848_MS GSE41849_MS GSE48023_H1N1)
budgets=(3 5 10)
dataset_count="${#datasets[@]}"
repeat_count=50
zero_based=$((task_id - 1))
budget_index=$((zero_based / (dataset_count * repeat_count)))
within_budget=$((zero_based % (dataset_count * repeat_count)))
repeat_id=$((within_budget / dataset_count + 1))
dataset_index=$((within_budget % dataset_count))
budget="${budgets[$budget_index]}"
dataset="${datasets[$dataset_index]}"
output_dir="${results_root}/${slug}/${dataset}/budget_${budget}/repeat_${repeat_id}"
mkdir -p "${output_dir}"
export CTSNN_LONGFS_RSCRIPT="${CTSNN_LONGFS_RSCRIPT:-/home/zliu/miniconda3/envs/ctsnn-longfs-r/bin/Rscript}"
cd "${project_dir}"
python src/run_expanded_benchmark.py \
  --datasets "${dataset}" --repeats 1 --repeat-start "${repeat_id}" \
  --methods "${method}" --budgets "${budget}" --output-dir "${output_dir}"

