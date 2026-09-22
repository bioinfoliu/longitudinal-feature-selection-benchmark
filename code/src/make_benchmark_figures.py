"""Create the main benchmark figures from a completed run.

The primary panel is deliberately restricted to classification tasks, where
the x-axis is AUROC as requested. Regression tasks are reported separately
with RMSE so incompatible metrics are never combined in one bar plot.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHOD_ORDER = [
    "Lasso", "ElasticNet", "StabilitySelection", "Stabl-RP",
    "LongGroupLasso", "LLSS", "RandomForest", "MutualInfo", "Boruta",
    "mRMR", "SNN-FS", "OutcomeSNN-FS",
]

COLORS = {
    "Lasso": "#777777", "ElasticNet": "#999999",
    "StabilitySelection": "#56B4E9", "Stabl-RP": "#E69F00",
    "LongGroupLasso": "#CC79A7", "LLSS": "#D55E00",
    "RandomForest": "#009E73", "MutualInfo": "#0072B2",
    "Boruta": "#F0E442", "mRMR": "#8C6BB1",
    "SNN-FS": "#1F78B4", "OutcomeSNN-FS": "#E31A1C",
}


def _save(fig, path: Path):
    fig.savefig(path, dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics = pd.read_csv(args.results / "fold_metrics.csv")
    if "AUROC" not in metrics.columns: metrics["AUROC"] = np.nan
    if "RMSE" not in metrics.columns: metrics["RMSE"] = np.nan

    auc = metrics.loc[metrics["AUROC"].notna()].copy()
    if not auc.empty:
        tasks = list(auc["Dataset"].drop_duplicates())
        fig, axes = plt.subplots(1, len(tasks), figsize=(7.0 * len(tasks), 6.8), squeeze=False)
        for ax, task in zip(axes[0], tasks):
            subset = auc[auc.Dataset == task]
            summary = subset.groupby("Method").AUROC.agg(["mean", "std"]).reindex(METHOD_ORDER).dropna()
            summary = summary.sort_values("mean")
            y = range(len(summary))
            ax.barh(list(y), summary["mean"], xerr=summary["std"],
                    color=[COLORS.get(m, "#999999") for m in summary.index],
                    alpha=0.9, capsize=3)
            ax.set_yticks(list(y), summary.index)
            ax.set_xlabel("AUROC (mean ± SD; axis shown from 0.5)")
            ax.set_title(task)
            ax.set_xlim(0.5, 1.0)
            ax.grid(axis="x", alpha=0.25)
        fig.suptitle("Classification benchmark: longitudinal feature selection", y=1.02, fontsize=15)
        fig.tight_layout()
        _save(fig, args.output_dir / "figure_main_auc_bar")
        auc.groupby("Method").AUROC.agg(Mean="mean", SD="std", N="count").reset_index().to_csv(
            args.output_dir / "main_auc_summary.csv", index=False,
        )

    reg = metrics.loc[metrics["RMSE"].notna()].copy()
    if not reg.empty:
        tasks = list(reg["Dataset"].drop_duplicates())
        fig, axes = plt.subplots(1, len(tasks), figsize=(5.8 * len(tasks), 6.8), squeeze=False)
        for ax, task in zip(axes[0], tasks):
            subset = reg[reg.Dataset == task]
            summary = subset.groupby("Method").RMSE.agg(["mean", "std"]).reindex(METHOD_ORDER).dropna()
            summary = summary.sort_values("mean", ascending=False)
            y = range(len(summary))
            ax.barh(list(y), summary["mean"], xerr=summary["std"],
                    color=[COLORS.get(m, "#999999") for m in summary.index],
                    alpha=0.9, capsize=3)
            ax.set_yticks(list(y), summary.index)
            ax.set_xlabel("RMSE (mean ± SD; lower is better)")
            ax.set_title(task)
            ax.grid(axis="x", alpha=0.25)
        fig.suptitle("Regression benchmark: longitudinal feature selection", y=1.02, fontsize=15)
        fig.tight_layout()
        _save(fig, args.output_dir / "figure_regression_rmse_bar")


if __name__ == "__main__":
    main()

