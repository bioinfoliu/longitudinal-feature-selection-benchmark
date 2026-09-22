"""Participant-cluster bootstrap summaries for secondary longitudinal outcomes."""

from __future__ import annotations

from pathlib import Path
import argparse

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "results" / "secondary_longitudinal_benchmark"


def rmse(frame):
    return float(np.sqrt(np.mean((frame.Outcome.to_numpy(float) - frame.Prediction.to_numpy(float)) ** 2)))


def sampled_rmse(frame, sampled_ids):
    squared_error = (frame.Outcome.to_numpy(float) - frame.Prediction.to_numpy(float)) ** 2
    identifiers = frame.ID.to_numpy()
    counts = pd.Series(sampled_ids).value_counts()
    weights = np.array([counts.get(identifier, 0) for identifier in identifiers], float)
    return float(np.sqrt(np.average(squared_error, weights=weights)))


def bh(values):
    values = np.asarray(values, float)
    order = np.argsort(values)
    ranked = values[order]
    adjusted = np.minimum.accumulate((ranked * len(values) / np.arange(1, len(values) + 1))[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.minimum(adjusted, 1.0)
    return result


def main(input_dir: Path = DEFAULT_OUT, iterations: int = 1000):
    input_dir = input_dir.resolve()
    data = pd.read_csv(input_dir / "outer_predictions.csv")
    tuning = pd.read_csv(input_dir / "inner_tuning.csv")
    summary_rows, comparison_rows, repeat_rows = [], [], []
    rng = np.random.default_rng(20260908)
    for dataset, dataset_data in data.groupby("Dataset"):
        methods = sorted(dataset_data.Method.unique())
        repeats = sorted(dataset_data.Repeat.unique())
        point = {
            method: float(np.mean([
                rmse(dataset_data[(dataset_data.Method == method) & (dataset_data.Repeat == repeat)])
                for repeat in repeats
            ])) for method in methods
        } 
        for method in methods:
            for repeat in repeats:
                repeat_data = dataset_data[
                    (dataset_data.Method == method) & (dataset_data.Repeat == repeat)
                ]
                repeat_rows.append({
                    "Dataset": dataset, "Method": method, "Repeat": repeat,
                    "RMSE": rmse(repeat_data),
                    "MAE": float(np.mean(np.abs(
                        repeat_data.Outcome.to_numpy(float)
                        - repeat_data.Prediction.to_numpy(float)
                    ))),
                })
        boot = {method: [] for method in methods}
        for _ in range(iterations):
            values = {method: [] for method in methods}
            for repeat in repeats:
                repeat_data = dataset_data[dataset_data.Repeat == repeat]
                ids = repeat_data.ID.drop_duplicates().to_numpy()
                sampled = rng.choice(ids, size=len(ids), replace=True)
                for method in methods:
                    values[method].append(sampled_rmse(repeat_data[repeat_data.Method == method], sampled))
            for method in methods:
                boot[method].append(np.mean(values[method]))
        reference = np.asarray(boot["Lasso"])
        for method in methods:
            values = np.asarray(boot[method])
            summary_rows.append({
                "Dataset": dataset, "Method": method, "MeanRepeatRMSE": point[method],
                "ParticipantBootstrapCI95Lower": np.quantile(values, .025),
                "ParticipantBootstrapCI95Upper": np.quantile(values, .975),
            })
            if method != "Lasso":
                difference = values - reference
                pvalue = min(1.0, 2 * min(np.mean(difference <= 0), np.mean(difference >= 0)))
                comparison_rows.append({
                    "Dataset": dataset, "Method": method, "Reference": "Lasso",
                    "RMSEDifference": point[method] - point["Lasso"],
                    "DifferenceCI95Lower": np.quantile(difference, .025),
                    "DifferenceCI95Upper": np.quantile(difference, .975),
                    "PairedBootstrapP": pvalue,
                })
    summary = pd.DataFrame(summary_rows)
    comparisons = pd.DataFrame(comparison_rows)
    repeat_metrics = pd.DataFrame(repeat_rows)
    repeat_summary = repeat_metrics.groupby(["Dataset", "Method"], as_index=False).agg(
        Mean=("RMSE", "mean"), SD=("RMSE", "std"), N=("RMSE", "count"),
        MeanMAE=("MAE", "mean"), SDMAE=("MAE", "std"),
    )
    repeat_summary.insert(2, "Metric", "RMSE")
    repeat_summary["CI95Lower"] = repeat_summary.Mean - 1.96 * repeat_summary.SD / np.sqrt(repeat_summary.N)
    repeat_summary["CI95Upper"] = repeat_summary.Mean + 1.96 * repeat_summary.SD / np.sqrt(repeat_summary.N)
    comparisons["BHAdjustedP"] = comparisons.groupby("Dataset")["PairedBootstrapP"].transform(lambda x: bh(x.to_numpy()))
    summary.to_csv(input_dir / "participant_bootstrap_summary.csv", index=False)
    comparisons.to_csv(input_dir / "participant_bootstrap_comparisons.csv", index=False)
    repeat_metrics.to_csv(input_dir / "repeat_metrics.csv", index=False)
    repeat_summary.to_csv(input_dir / "performance_summary.csv", index=False)
    budget_frequency = (
        tuning.groupby(["Dataset", "Method", "SelectedBudget"], as_index=False)
        .size().rename(columns={"size": "Selections"})
    )
    budget_frequency["Frequency"] = budget_frequency.groupby(["Dataset", "Method"])["Selections"].transform(
        lambda values: values / values.sum()
    )
    budget_frequency.to_csv(input_dir / "selected_budget_frequency.csv", index=False)
    print(summary.sort_values(["Dataset", "MeanRepeatRMSE"]).to_string(index=False))
    print(comparisons.sort_values(["Dataset", "BHAdjustedP"]).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--iterations", type=int, default=1000)
    args = parser.parse_args()
    main(args.input_dir, args.iterations)
