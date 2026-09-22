"""Summarize completed benchmark runs at the independent-repeat level."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)


METHOD_ORDER = [
    "Lasso", "ElasticNet", "StabilitySelection", "Stabl-RP",
    "LongGroupLasso", "LLSS", "RandomForest", "MutualInfo", "Boruta",
    "mRMR", "SNN-FS", "OutcomeSNN-FS",
]


def repeat_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, repeat, method), group in predictions.groupby(["Dataset", "Repeat", "Method"]):
        if dataset in {"PE", "PE_LOPE_MATCHED", "OLINK_COVID"}:
            subject = group.groupby("ID", as_index=False).agg(
                Outcome=("Outcome", "first"), Prediction=("Prediction", "mean"),
            )
            rows.append({
                "Dataset": dataset, "Repeat": int(repeat), "Method": method,
                "Metric": "AUROC",
                "AUROC": roc_auc_score(subject.Outcome, subject.Prediction),
                "AUPRC": average_precision_score(subject.Outcome, subject.Prediction),
            })
        else:
            rows.append({
                "Dataset": dataset, "Repeat": int(repeat), "Method": method,
                "Metric": "RMSE",
                "RMSE": float(np.sqrt(mean_squared_error(group.Outcome, group.Prediction))),
                "MAE": float(mean_absolute_error(group.Outcome, group.Prediction)),
            })
    return pd.DataFrame(rows)


def save_auc_figure(metrics: pd.DataFrame, output: Path):
    auc = metrics.loc[metrics["AUROC"].notna()].copy() if "AUROC" in metrics.columns else metrics.iloc[0:0].copy()
    if auc.empty:
        return
    tasks = list(auc.Dataset.drop_duplicates())
    colors = plt.get_cmap("tab20")(np.linspace(0, 1, len(METHOD_ORDER)))
    fig, axes = plt.subplots(1, len(tasks), figsize=(7.0 * len(tasks), 6.8), squeeze=False)
    for ax, task in zip(axes[0], tasks):
        summary = auc[auc.Dataset == task].groupby("Method").AUROC.agg(["mean", "std"])
        summary = summary.reindex(METHOD_ORDER).dropna().sort_values("mean")
        y = np.arange(len(summary))
        ax.barh(y, summary["mean"], xerr=summary["std"], color=colors[:len(summary)], capsize=3)
        ax.set_yticks(y, summary.index)
        ax.set_xlabel("AUROC (repeat-level mean ± SD; axis shown from 0.5)")
        ax.set_title(task)
        ax.set_xlim(0.5, 1)
        ax.grid(axis="x", alpha=0.25)
    fig.suptitle("Main classification benchmark", y=1.02, fontsize=15)
    fig.tight_layout()
    fig.savefig(output / "figure_main_auc_bar_repeat_level.png", dpi=300, bbox_inches="tight")
    fig.savefig(output / "figure_main_auc_bar_repeat_level.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    predictions = pd.read_csv(args.results / "outer_predictions.csv")
    metrics = repeat_metrics(predictions)
    metrics.to_csv(args.results / "repeat_level_metrics.csv", index=False)

    summary_rows = []
    for (dataset, method), group in metrics.groupby(["Dataset", "Method"]):
        metric = "AUROC" if "AUROC" in group.columns and group["AUROC"].notna().any() else "RMSE"
        row = {
            "Dataset": dataset, "Method": method, "Metric": metric,
            "Mean": group[metric].mean(), "SD": group[metric].std(ddof=1), "N": len(group),
        }
        if metric == "AUROC":
            row.update(MeanAUPRC=group.AUPRC.mean(), SDAUPRC=group.AUPRC.std(ddof=1))
        else:
            row.update(MeanMAE=group.MAE.mean(), SDMAE=group.MAE.std(ddof=1))
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(args.results / "repeat_level_summary.csv", index=False)

    rank_rows = []
    for dataset, group in summary.groupby("Dataset"):
        ascending = group.Metric.iloc[0] == "RMSE"
        ranked = group.assign(Rank=group.Mean.rank(method="min", ascending=ascending))
        rank_rows.append(ranked)
    ranked = pd.concat(rank_rows, ignore_index=True)
    ranked.to_csv(args.results / "repeat_level_ranks.csv", index=False)
    ranked.groupby("Method", as_index=False).agg(
        MeanRank=("Rank", "mean"), SD=("Rank", "std"), Datasets=("Rank", "count"),
    ).sort_values("MeanRank").to_csv(
        args.results / "average_method_rank.csv", index=False,
    )
    save_auc_figure(metrics, args.results / "figures")
    print(summary.sort_values(["Dataset", "Mean"]).to_string(index=False))


if __name__ == "__main__":
    main()
