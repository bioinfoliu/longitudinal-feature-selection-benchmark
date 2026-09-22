"""Analyze the 3/5/10 panel-budget evidence saved by completed runs.

The saved benchmark contains inner-validation scores for every candidate budget,
but outer-test metrics only for the budget selected inside each outer fold.  This
script therefore labels these two evidence levels explicitly and never presents
the conditional outer-test analysis as a fixed-budget experiment.
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TASK_TYPES = {
    "PE": "classification", "PE_LOPE_MATCHED": "classification",
    "OLINK_COVID": "classification", "GSE41848_MS": "classification",
    "GSE41849_MS": "classification", "FPG": "regression",
    "FPG_KARE": "regression", "KARE_STATUS": "regression",
    "IGG": "regression", "IGM": "regression", "TRA": "regression",
    "TRB": "regression", "BRIST1D": "regression",
    "GSE48023_H1N1": "regression",
}
BUDGETS = (3, 5, 10)


def source_directories(results_root: Path) -> list[tuple[str, Path, bool]]:
    return [
        ("benchmark_full50", results_root / "main_benchmark/benchmark_full50", False),
        ("olink_50", results_root / "new_datasets/olink_50", False),
        ("brist1d_50", results_root / "new_datasets/brist1d_50", False),
        ("geo_expansion_50", results_root / "new_datasets/geo_expansion_50", False),
        ("gse48023_subject_50", results_root / "new_datasets/gse48023_subject_50", False),
        ("geeverse_full50", results_root / "main_benchmark/longitudinal_extensions/geeverse_full50", False),
        ("glmmlasso_full50", results_root / "main_benchmark/longitudinal_extensions/glmmlasso_full50/aggregated", True),
        ("pgee_full50", results_root / "main_benchmark/longitudinal_extensions/pgee_full50/aggregated", True),
    ]


def _read(run_dir: Path, name: str, combined: bool) -> pd.DataFrame:
    path = run_dir / (("combined_" if combined else "") + name)
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def load_saved_evidence(results_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    folds, tuning, selected = [], [], []
    for source, run_dir, combined in source_directories(results_root):
        for collection, filename in [
            (folds, "fold_metrics.csv"),
            (tuning, "inner_tuning.csv"),
            (selected, "selected_features.csv"),
        ]:
            frame = _read(run_dir, filename, combined)
            if source == "geo_expansion_50":
                frame = frame[frame["Dataset"] != "GSE48023_H1N1"].copy()
            frame["SourceRun"] = source
            collection.append(frame)
    return tuple(pd.concat(frames, ignore_index=True) for frames in (folds, tuning, selected))


def budget_frequency(tuning: pd.DataFrame) -> pd.DataFrame:
    out = (
        tuning.groupby(["Dataset", "Method", "SelectedBudget"], dropna=False)
        .size().rename("Selections").reset_index()
    )
    totals = out.groupby(["Dataset", "Method"])["Selections"].transform("sum")
    out["Frequency"] = out["Selections"] / totals
    return out.rename(columns={"Dataset": "Task"}).sort_values(["Task", "Method", "SelectedBudget"])


def inner_budget_scores(tuning: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for row in tuning.itertuples(index=False):
        scores = json.loads(row.InnerScores)
        by_budget: dict[int, list[float]] = {k: [] for k in BUDGETS}
        for key, value in scores.items():
            budget = int(key.split(";")[0].split("=")[1])
            if budget in by_budget and np.isfinite(value):
                by_budget[budget].append(float(value))
        task_type = TASK_TYPES[row.Dataset]
        for budget, values in by_budget.items():
            if not values:
                continue
            best_utility = max(values)
            rows.append({
                "Task": row.Dataset, "TaskType": task_type, "Repeat": row.Repeat,
                "Fold": row.Fold, "Method": row.Method, "PanelSize": budget,
                "Metric": "AUROC" if task_type == "classification" else "RMSE",
                "InnerValidationPerformance": best_utility if task_type == "classification" else -best_utility,
                "EvidenceLevel": "inner validation; regularization optimized within panel size",
            })
    detail = pd.DataFrame(rows)
    summary = (
        detail.groupby(["Task", "TaskType", "Method", "PanelSize", "Metric", "EvidenceLevel"])
        ["InnerValidationPerformance"].agg(Mean="mean", SD="std", Evaluations="size").reset_index()
    )
    return detail, summary


def conditional_outer_performance(folds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for task, group in folds.groupby("Dataset"):
        task_type = TASK_TYPES[task]
        metric = "AUROC" if task_type == "classification" else "RMSE"
        for (method, budget), item in group.groupby(["Method", "n_features"]):
            values = item[metric].dropna().to_numpy(float)
            if len(values):
                rows.append({
                    "Task": task, "TaskType": task_type, "Method": method,
                    "SelectedPanelSize": int(budget), "Metric": metric,
                    "Mean": values.mean(), "SD": values.std(ddof=1), "OuterFolds": len(values),
                    "EvidenceLevel": "outer test conditional on the inner-selected panel size; not a fixed-budget experiment",
                })
    return pd.DataFrame(rows)


def conditional_stability(selected: pd.DataFrame) -> pd.DataFrame:
    feature_sets = []
    keys = ["Dataset", "Method", "Repeat", "Fold"]
    for key, group in selected.groupby(keys):
        features = frozenset(group.Feature.astype(str))
        feature_sets.append((*key, len(features), features))
    sets = pd.DataFrame(feature_sets, columns=keys + ["SelectedPanelSize", "Features"])
    rows = []
    for (task, method, budget), group in sets.groupby(["Dataset", "Method", "SelectedPanelSize"]):
        values = []
        items = group.Features.tolist()
        for left, right in combinations(items, 2):
            union = left | right
            values.append(len(left & right) / len(union) if union else 1.0)
        rows.append({
            "Task": task, "Method": method, "SelectedPanelSize": int(budget),
            "MeanJaccard": np.mean(values) if values else np.nan,
            "SDJaccard": np.std(values, ddof=1) if len(values) > 1 else np.nan,
            "Pairs": len(values),
            "EvidenceLevel": "selection stability conditional on the inner-selected panel size",
        })
    return pd.DataFrame(rows)


def make_figures(frequency: pd.DataFrame, inner: pd.DataFrame, stability: pd.DataFrame, out: Path) -> None:
    overall = frequency.groupby(["Method", "SelectedBudget"])["Selections"].sum().unstack(fill_value=0)
    overall = overall.reindex(columns=BUDGETS, fill_value=0)
    overall = overall.div(overall.sum(axis=1), axis=0).sort_values(10)
    fig, ax = plt.subplots(figsize=(10, 7))
    left = np.zeros(len(overall))
    colors = {3: "#56B4E9", 5: "#E69F00", 10: "#009E73"}
    for budget in BUDGETS:
        ax.barh(overall.index, overall[budget], left=left, label=f"{budget} features", color=colors[budget])
        left += overall[budget].to_numpy()
    ax.set(xlabel="Selection frequency across outer folds", xlim=(0, 1))
    ax.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.01))
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out / "figure_budget_selection_frequency.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    plot = inner.copy()
    plot["RelativeLoss"] = np.nan
    for _, group in plot.groupby(["Task", "Method", "Repeat", "Fold"]):
        values = group.InnerValidationPerformance.to_numpy(float)
        if group.TaskType.iloc[0] == "classification":
            loss = np.nanmax(values) - values
        else:
            denominator = max(np.nanmin(values), 1e-12)
            loss = values / denominator - 1.0
        plot.loc[group.index, "RelativeLoss"] = loss
    curve = plot.groupby(["TaskType", "PanelSize"])["RelativeLoss"].agg(["mean", "sem"]).reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True)
    for ax, task_type in zip(axes, ["classification", "regression"]):
        item = curve[curve.TaskType == task_type]
        ax.errorbar(item.PanelSize, item["mean"], yerr=item["sem"], marker="o", capsize=4)
        ax.set_title(task_type.capitalize())
        ax.set_xlabel("Panel size")
        ax.set_ylabel("Mean relative inner-validation loss")
        ax.set_xticks(BUDGETS)
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "figure_inner_panel_size_sensitivity.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    groups = [stability.loc[stability.SelectedPanelSize == k, "MeanJaccard"].dropna() for k in BUDGETS]
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.boxplot(groups, tick_labels=[str(k) for k in BUDGETS], showmeans=True)
    ax.set(xlabel="Inner-selected panel size", ylabel="Mean pairwise Jaccard similarity")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "figure_conditional_stability_by_panel_size.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    folds, tuning, selected = load_saved_evidence(args.results_root)
    frequency = budget_frequency(tuning)
    inner_detail, inner_summary = inner_budget_scores(tuning)
    outer = conditional_outer_performance(folds)
    stability = conditional_stability(selected)
    frequency.to_csv(args.output_dir / "panel_budget_selection_frequency.csv", index=False)
    inner_detail.to_csv(args.output_dir / "inner_validation_panel_size_detail.csv", index=False)
    inner_summary.to_csv(args.output_dir / "inner_validation_panel_size_summary.csv", index=False)
    outer.to_csv(args.output_dir / "outer_performance_by_inner_selected_panel_size.csv", index=False)
    stability.to_csv(args.output_dir / "conditional_selection_stability_by_panel_size.csv", index=False)
    make_figures(frequency, inner_detail, stability, args.output_dir)
    (args.output_dir / "README.md").write_text(
        "# Panel-size sensitivity\n\n"
        "The inner-validation table compares all candidate budgets (3, 5, 10). "
        "Outer-test and Jaccard tables are conditional on the budget selected inside "
        "each outer fold; they are not fixed-budget experiments. Strict fixed-budget "
        "outer-test curves require rerunning `scripts/run_fixed_budget_sensitivity.sbatch`.\n"
    )
    print(f"Wrote panel-size sensitivity outputs to {args.output_dir}")


if __name__ == "__main__":
    main()

