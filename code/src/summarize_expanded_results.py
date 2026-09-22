"""Participant-level summaries and paired bootstrap inference for expanded CV."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "main_benchmark" / "benchmark_with_new"


def bh(pvalues):
    values = np.asarray(pvalues, float)
    order = np.argsort(values)
    ranked = values[order]
    adjusted = np.minimum.accumulate((ranked * len(values) / np.arange(1, len(values) + 1))[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.minimum(adjusted, 1.0)
    return result


def fast_auc(y, scores):
    y = np.asarray(y, int)
    ranks = rankdata(np.asarray(scores, float), method="average")
    positives = y == 1
    n_positive = positives.sum()
    n_negative = len(y) - n_positive
    return (ranks[positives].sum() - n_positive * (n_positive + 1) / 2) / (n_positive * n_negative)


def classification_summary(predictions: pd.DataFrame, iterations: int = 2000):
    repeat_rows = []
    for (dataset, method, repeat), group in predictions.groupby(["Dataset", "Method", "Repeat"]):
        repeat_rows.append({
            "Dataset": dataset, "Method": method, "Repeat": repeat,
            "AUROC": roc_auc_score(group.Outcome, group.Prediction),
            "AUPRC": average_precision_score(group.Outcome, group.Prediction),
            "Brier": brier_score_loss(group.Outcome, group.Prediction),
        })
    repeats = pd.DataFrame(repeat_rows)
    aggregate = (
        predictions.groupby(["Dataset", "Method", "ID"], as_index=False)
        .agg(Outcome=("Outcome", "first"), Prediction=("Prediction", "mean"))
    )
    summary_rows, comparison_rows = [], []
    rng = np.random.default_rng(20260908)
    for dataset, data in aggregate.groupby("Dataset"):
        methods = sorted(data.Method.unique())
        wide = data.pivot(index="ID", columns="Method", values="Prediction")
        outcome = data.drop_duplicates("ID").set_index("ID").loc[wide.index, "Outcome"].to_numpy()
        bootstrap_auc = {method: [] for method in methods}
        difference = {method: [] for method in methods if method != "Lasso"}
        for _ in range(iterations):
            sample = rng.integers(0, len(wide), len(wide))
            if np.unique(outcome[sample]).size < 2:
                continue
            reference_auc = fast_auc(outcome[sample], wide["Lasso"].to_numpy()[sample])
            for method in methods:
                value = fast_auc(outcome[sample], wide[method].to_numpy()[sample])
                bootstrap_auc[method].append(value)
                if method != "Lasso":
                    difference[method].append(value - reference_auc)
        for method in methods:
            point = roc_auc_score(outcome, wide[method])
            values = np.asarray(bootstrap_auc[method])
            method_repeats = repeats[(repeats.Dataset == dataset) & (repeats.Method == method)]
            summary_rows.append({
                "Dataset": dataset, "Method": method,
                "OOF_AUROC": point,
                "ParticipantBootstrapCI95Lower": np.quantile(values, 0.025),
                "ParticipantBootstrapCI95Upper": np.quantile(values, 0.975),
                "MeanRepeatAUROC": method_repeats.AUROC.mean(),
                "SDRepeatAUROC": method_repeats.AUROC.std(),
                "MeanRepeatAUPRC": method_repeats.AUPRC.mean(),
                "MeanRepeatBrier": method_repeats.Brier.mean(),
            })
            if method != "Lasso":
                delta = np.asarray(difference[method])
                pvalue = min(1.0, 2 * min(np.mean(delta <= 0), np.mean(delta >= 0)))
                comparison_rows.append({
                    "Dataset": dataset, "Method": method, "Reference": "Lasso",
                    "OOF_AUROCDifference": point - roc_auc_score(outcome, wide["Lasso"]),
                    "DifferenceCI95Lower": np.quantile(delta, 0.025),
                    "DifferenceCI95Upper": np.quantile(delta, 0.975),
                    "ParticipantBootstrapP": pvalue,
                })
    summary = pd.DataFrame(summary_rows)
    comparisons = pd.DataFrame(comparison_rows)
    comparisons["BHAdjustedP"] = comparisons.groupby("Dataset")["ParticipantBootstrapP"].transform(
        lambda values: bh(values.to_numpy())
    )
    return repeats, summary, comparisons


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default=str(OUT))
    args = parser.parse_args()
    output = Path(args.input_dir)
    predictions = pd.read_csv(output / "outer_predictions.csv")
    tuning = pd.read_csv(output / "inner_tuning.csv")
    repeats, summary, comparisons = classification_summary(predictions)
    repeat_summary = repeats.groupby(["Dataset", "Method"], as_index=False).agg(
        Mean=("AUROC", "mean"), SD=("AUROC", "std"), N=("AUROC", "count"),
        MeanAUPRC=("AUPRC", "mean"), SDAUPRC=("AUPRC", "std"),
    )
    repeat_summary.insert(2, "Metric", "AUROC")
    repeat_summary["CI95Lower"] = repeat_summary.Mean - 1.96 * repeat_summary.SD / np.sqrt(repeat_summary.N)
    repeat_summary["CI95Upper"] = repeat_summary.Mean + 1.96 * repeat_summary.SD / np.sqrt(repeat_summary.N)
    repeats.to_csv(output / "repeat_oof_metrics.csv", index=False)
    summary.to_csv(output / "participant_bootstrap_summary.csv", index=False)
    comparisons.to_csv(output / "participant_bootstrap_comparisons.csv", index=False)
    repeat_summary.to_csv(output / "performance_summary.csv", index=False)
    budget_frequency = (
        tuning.groupby(["Dataset", "Method", "SelectedBudget"], as_index=False)
        .size().rename(columns={"size": "Selections"})
    )
    budget_frequency["Frequency"] = budget_frequency.groupby(["Dataset", "Method"])["Selections"].transform(
        lambda values: values / values.sum()
    )
    budget_frequency.to_csv(output / "selected_budget_frequency.csv", index=False)
    print(summary.sort_values("OOF_AUROC", ascending=False).to_string(index=False))
    print(comparisons.sort_values("BHAdjustedP").to_string(index=False))


if __name__ == "__main__":
    main()
