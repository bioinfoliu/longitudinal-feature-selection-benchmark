#!/usr/bin/env bash
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --output=/home/zliu/ctsnn/logs/LONGFS_AGG_%j.log

set -euo pipefail
project_dir="${CTSNN_PROJECT:-/home/zliu/ctsnn/project}"
method_root="${METHOD_ROOT:?Submit with METHOD_ROOT}"
cd "${project_dir}"
python src/aggregate_longitudinal_extension.py \
  --method-root "${method_root}" \
  --expected-repeats 50
