"""Aggregate and validate a 14-dataset longitudinal selector benchmark."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd


DATASETS = (
    "PE", "PE_LOPE_MATCHED", "FPG", "FPG_KARE", "KARE_STATUS",
    "IGG", "IGM", "TRA", "TRB", "OLINK_COVID", "BRIST1D",
    "GSE41848_MS", "GSE41849_MS", "GSE48023_H1N1",
)

COMBINED_TABLES = (
    "fold_metrics.csv", "repeat_metrics.csv", "performance_summary.csv",
    "selected_features.csv", "outer_predictions.csv", "inner_tuning.csv",
    "selection_stability.csv", "selected_budget_frequency.csv",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method-root", required=True)
    parser.add_argument("--expected-repeats", type=int, default=50)
    args = parser.parse_args()

    root = Path(args.method_root).resolve()
    aggregate_script = Path(__file__).with_name("aggregate_repeat_runs.py")
    output_root = root / "aggregated"
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, object]] = []

    for dataset in DATASETS:
        input_root = root / dataset
        repeat_dirs = sorted(path for path in input_root.glob("repeat_*") if path.is_dir())
        complete = sum((path / "fold_metrics.csv").exists() for path in repeat_dirs)
        manifest_rows.append({
            "Dataset": dataset,
            "ExpectedRepeats": args.expected_repeats,
            "RepeatDirectories": len(repeat_dirs),
            "CompleteRepeats": complete,
            "Complete": complete == args.expected_repeats,
        })
        if complete != args.expected_repeats:
            continue
        subprocess.run([
            sys.executable, str(aggregate_script),
            "--input-root", str(input_root),
            "--output-dir", str(output_root / dataset),
            "--expected-repeats", str(args.expected_repeats),
        ], check=True)

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(output_root / "completion_manifest.csv", index=False)
    if not bool(manifest["Complete"].all()):
        incomplete = manifest.loc[~manifest["Complete"], ["Dataset", "CompleteRepeats"]]
        raise RuntimeError("Incomplete benchmark:\n" + incomplete.to_string(index=False))

    for name in COMBINED_TABLES:
        frames = [pd.read_csv(output_root / dataset / name) for dataset in DATASETS]
        pd.concat(frames, ignore_index=True).to_csv(output_root / f"combined_{name}", index=False)

    print(manifest.to_string(index=False))


if __name__ == "__main__":
    main()
