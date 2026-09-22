"""Regenerate publication-facing tables and figures from compact completed results."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHOD_COLORS = {
    "Lasso": "#7F7F7F", "ElasticNet": "#A6A6A6",
    "StabilitySelection": "#56B4E9", "Stabl-RP": "#E69F00",
    "LongGroupLasso": "#CC79A7", "LLSS": "#D55E00",
    "RandomForest": "#009E73", "MutualInfo": "#0072B2",
    "Boruta": "#F0E442", "mRMR": "#8C6BB1",
    "SNN-FS": "#1F78B4", "OutcomeSNN-FS": "#E31A1C",
    "geeVerse": "#6A3D9A", "glmmLasso": "#1B9E77", "PGEE": "#7570B3",
}


def task_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["Task", "Dataset", "Cohort", "TaskType", "OutcomeDescription", "Method"]
    for key, group in metrics.groupby(keys, dropna=False):
        task, dataset, cohort, task_type, outcome, method = key
        primary, secondary = ("AUROC", "AUPRC") if task_type == "classification" else ("RMSE", "MAE")
        rows.append({
            "Task": task, "Dataset": dataset, "Cohort": cohort, "TaskType": task_type,
            "Outcome": outcome, "Method": method, "Metric": primary,
            "Mean": group[primary].mean(), "SD": group[primary].std(ddof=1),
            "Repeats": group.Repeat.nunique(), f"Mean{secondary}": group[secondary].mean(),
            f"SD{secondary}": group[secondary].std(ddof=1),
        })
    return pd.DataFrame(rows)


def rank_tables(metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for task, group in metrics.groupby("Task"):
        task_type = group.TaskType.iloc[0]
        metric = "AUROC" if task_type == "classification" else "RMSE"
        wide = group.pivot(index="Repeat", columns="Method", values=metric).dropna(axis=1)
        ranks = wide.rank(axis=1, ascending=(metric == "RMSE"), method="average")
        for method in ranks:
            rows.append({
                "Task": task, "Cohort": group.Cohort.iloc[0], "TaskType": task_type,
                "Method": method, "MeanRank": ranks[method].mean(),
                "SDRank": ranks[method].std(ddof=1), "WinRate": (ranks[method] == 1).mean(),
                "Repeats": len(ranks),
            })
    task_ranks = pd.DataFrame(rows)
    overall = task_ranks.groupby("Method").agg(
        MeanRank=("MeanRank", "mean"), SDRankAcrossTasks=("MeanRank", "std"),
        MeanWinRate=("WinRate", "mean"), Tasks=("Task", "nunique"),
    ).reset_index().sort_values("MeanRank")
    return task_ranks, overall


def task_panels(summary: pd.DataFrame, task_type: str, output: Path) -> None:
    tasks = summary.loc[summary.TaskType == task_type, "Task"].drop_duplicates().tolist()
    ncols = 3
    nrows = math.ceil(len(tasks) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4.2 * nrows), squeeze=False)
    metric = "AUROC" if task_type == "classification" else "RMSE"
    for ax, task in zip(axes.flat, tasks):
        item = summary[summary.Task == task].set_index("Method")
        item = item.sort_values("Mean", ascending=(metric == "AUROC"))
        y = np.arange(len(item))
        ax.barh(y, item.Mean, xerr=item.SD, capsize=2.5,
                color=[METHOD_COLORS.get(method, "#808080") for method in item.index])
        ax.set_yticks(y, item.index, fontsize=8)
        ax.set_xlabel(f"{metric} (50-repeat mean ± SD)")
        ax.set_title(f"{task}: {item.Dataset.iloc[0]}", weight="bold")
        ax.grid(axis="x", alpha=.25)
        if metric == "AUROC":
            ax.set_xlim(.5, 1.0)
        else:
            values, errors = item.Mean.to_numpy(float), item.SD.fillna(0).to_numpy(float)
            spread = max(float(np.nanmax(values) - np.nanmin(values)), 1e-8)
            pad = max(.10 * spread, 1.5 * float(np.nanmax(errors)))
            ax.set_xlim(float(np.nanmin(values)) - pad, float(np.nanmax(values)) + pad)
        ax.invert_yaxis()
    for ax in axes.flat[len(tasks):]:
        ax.axis("off")
    fig.tight_layout()
    name = "figure_main_classification_auroc.png" if task_type == "classification" else "figure_main_regression_rmse.png"
    fig.savefig(output / name, dpi=300, bbox_inches="tight")
    plt.close(fig)


def rank_heatmap(task_ranks: pd.DataFrame, output: Path) -> None:
    matrix = task_ranks.pivot(index="Method", columns="Task", values="MeanRank")
    matrix = matrix.loc[matrix.mean(axis=1).sort_values().index]
    fig, ax = plt.subplots(figsize=(13, 7))
    image = ax.imshow(matrix.to_numpy(), aspect="auto", cmap="viridis_r")
    ax.set_xticks(range(len(matrix.columns)), matrix.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(matrix.index)), matrix.index)
    for i in range(len(matrix.index)):
        for j in range(len(matrix.columns)):
            value = matrix.iloc[i, j]
            if np.isfinite(value):
                ax.text(j, i, f"{value:.1f}", ha="center", va="center", fontsize=7,
                        color="white" if value > np.nanmedian(matrix.to_numpy()) else "black")
    fig.colorbar(image, ax=ax, label="Mean rank (lower is better)")
    fig.tight_layout()
    fig.savefig(output / "figure_task_method_rank_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("results/summary"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/reproduced"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics = pd.read_csv(args.input_dir / "repeat_level_metrics_all_tasks.csv")
    summary = task_summary(metrics)
    task_ranks, overall = rank_tables(metrics)
    summary.to_csv(args.output_dir / "final_task_performance_summary.csv", index=False)
    task_ranks.to_csv(args.output_dir / "task_level_rank_summary.csv", index=False)
    overall.to_csv(args.output_dir / "overall_method_rank_summary.csv", index=False)
    task_panels(summary, "classification", args.output_dir)
    task_panels(summary, "regression", args.output_dir)
    rank_heatmap(task_ranks, args.output_dir)
    print(f"Regenerated publication outputs in {args.output_dir}")


if __name__ == "__main__":
    main()

