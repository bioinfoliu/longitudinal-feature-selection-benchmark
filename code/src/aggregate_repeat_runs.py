"""Merge one-repeat Slurm-array outputs into a benchmark result directory."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from run_expanded_benchmark import load_dataset, stability_table


TABLES = (
    "fold_metrics.csv",
    "selected_features.csv",
    "outer_predictions.csv",
    "inner_tuning.csv",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", required=True, help="Directory containing repeat_1 ... repeat_50.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-repeats", type=int, default=50)
    args = parser.parse_args()

    input_root = Path(args.input_root)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    repeat_dirs = sorted(path for path in input_root.glob("repeat_*") if path.is_dir())
    if not repeat_dirs:
        raise FileNotFoundError(f"No repeat_* directories found in {input_root}")

    tables: dict[str, pd.DataFrame] = {}
    for name in TABLES:
        frames = [pd.read_csv(path / name) for path in repeat_dirs if (path / name).exists()]
        if not frames:
            raise FileNotFoundError(f"No {name} files found under {input_root}")
        tables[name] = pd.concat(frames, ignore_index=True)
        tables[name].to_csv(output / name, index=False)

    metrics = tables["fold_metrics.csv"]
    selected = tables["selected_features.csv"]
    tuning = tables["inner_tuning.csv"]
    metric_columns = [
        column for column in ("AUROC", "AUPRC", "RMSE", "MAE", "Accuracy", "BalancedAccuracy")
        if column in metrics.columns
    ]
    # Average the five outer folds within each repeat first. Thus SD and N
    # describe independent participant-level repeats, not 50*5 folds.
    repeat_metrics = metrics.groupby(["Dataset", "Method", "Repeat"], as_index=False)[metric_columns].mean()
    repeat_metrics.to_csv(output / "repeat_metrics.csv", index=False)
    summary_rows = []
    for (dataset, method), subset in repeat_metrics.groupby(["Dataset", "Method"]):
        for metric in metric_columns:
            values = subset[metric].dropna().to_numpy(float)
            if not len(values):
                continue
            mean = float(values.mean())
            sd = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            summary_rows.append({
                "Dataset": dataset, "Method": method, "Metric": metric,
                "Mean": mean, "SD": sd, "N": len(values),
                "CI95Lower": mean - 1.96 * sd / max(len(values), 1) ** 0.5,
                "CI95Upper": mean + 1.96 * sd / max(len(values), 1) ** 0.5,
            })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output / "performance_summary.csv", index=False)
    pd.DataFrame(columns=["Dataset", "Method", "Reference", "Metric", "MeanPairedDifference", "WilcoxonP", "BHAdjustedP"]).to_csv(
        output / "paired_comparisons.csv", index=False
    )

    total_features = {
        dataset: len(load_dataset(dataset)[3])
        for dataset in sorted(metrics.Dataset.unique())
    }
    stability_table(selected, total_features).to_csv(output / "selection_stability.csv", index=False)
    budget_frequency = (
        tuning.groupby(["Dataset", "Method", "SelectedBudget"], as_index=False)
        .size().rename(columns={"size": "Selections"})
    )
    budget_frequency["Frequency"] = budget_frequency.groupby(["Dataset", "Method"])["Selections"].transform(
        lambda values: values / values.sum()
    )
    budget_frequency.to_csv(output / "selected_budget_frequency.csv", index=False)

    found = sorted(metrics.Repeat.unique())
    pd.DataFrame({"Repeat": found, "Completed": True}).to_csv(output / "repeat_manifest.csv", index=False)
    if len(found) != args.expected_repeats:
        raise RuntimeError(f"Expected {args.expected_repeats} repeats but found {len(found)}: {found}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
