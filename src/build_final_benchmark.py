"""Build a reproducible, publication-ready longitudinal FS benchmark package.

This script combines completed 50-repeat runs without re-fitting models.  It
keeps cohort/outcome tasks distinct from source cohorts, produces rank and
paired-comparison statistics, audits longitudinal structure, and writes two
main figures: classification (AUROC) and regression (RMSE).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon


PROJECT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = PROJECT.parent / "results" if (PROJECT.parent / "results").exists() else PROJECT / "results"

METHOD_ORDER = [
    "Lasso", "ElasticNet", "StabilitySelection", "Stabl-RP",
    "LongGroupLasso", "LLSS", "RandomForest", "MutualInfo", "Boruta",
    "mRMR", "SNN-FS", "OutcomeSNN-FS", "geeVerse", "glmmLasso", "PGEE",
]

METHOD_COLORS = {
    "Lasso": "#7F7F7F", "ElasticNet": "#A6A6A6",
    "StabilitySelection": "#56B4E9", "Stabl-RP": "#E69F00",
    "LongGroupLasso": "#CC79A7", "LLSS": "#D55E00",
    "RandomForest": "#009E73", "MutualInfo": "#0072B2",
    "Boruta": "#F0E442", "mRMR": "#8C6BB1",
    "SNN-FS": "#1F78B4", "OutcomeSNN-FS": "#E31A1C",
    "geeVerse": "#6A3D9A", "glmmLasso": "#1B9E77", "PGEE": "#7570B3",
}

RUNS = {
    "benchmark_full50": RESULTS_ROOT / "main_benchmark" / "benchmark_full50",
    "olink_50": RESULTS_ROOT / "new_datasets" / "olink_50",
    "brist1d_50": RESULTS_ROOT / "new_datasets" / "brist1d_50",
    "geeverse_full50": RESULTS_ROOT / "main_benchmark" / "longitudinal_extensions" / "geeverse_full50",
    "geo_expansion_50": RESULTS_ROOT / "new_datasets" / "geo_expansion_50",
    "gse48023_subject_50": RESULTS_ROOT / "new_datasets" / "gse48023_subject_50",
}

# These two completed official-package extensions write their 50-repeat output
# as combined CSV files rather than the generic benchmark-run layout.
EXTENSION_RUNS = {
    "glmmlasso_full50": RESULTS_ROOT / "main_benchmark" / "longitudinal_extensions" / "glmmlasso_full50" / "aggregated",
    "pgee_full50": RESULTS_ROOT / "main_benchmark" / "longitudinal_extensions" / "pgee_full50" / "aggregated",
}

# A task is a cohort--outcome evaluation. Multiple tasks can originate from one
# source cohort; the latter is what should be counted as a dataset in prose.
TASK_METADATA = {
    "PE": ("Stanford PE", "Stanford PE", "classification", "PE versus term control"),
    "PE_LOPE_MATCHED": ("Stanford PE", "Stanford PE", "classification", "Late-onset PE versus term control"),
    "FPG": ("Japan T2D", "Japan T2D", "regression", "Fasting plasma glucose"),
    "FPG_KARE": ("KARE", "Korea KARE", "regression", "Fasting plasma glucose"),
    "KARE_STATUS": ("KARE", "Korea KARE", "regression", "Longitudinal status score"),
    "IGG": ("Korea COVID-19 (CODA)", "Korea COVID-19", "regression", "BCR-IGG clonal diversity"),
    "IGM": ("Korea COVID-19 (CODA)", "Korea COVID-19", "regression", "BCR-IGM clonal diversity"),
    "TRA": ("Korea COVID-19 (CODA)", "Korea COVID-19", "regression", "TCR-alpha clonal diversity"),
    "TRB": ("Korea COVID-19 (CODA)", "Korea COVID-19", "regression", "TCR-beta clonal diversity"),
    "OLINK_COVID": ("Olink COVID", "Olink COVID", "classification", "Fatal disease"),
    "BRIST1D": ("BrisT1D", "BrisT1D", "regression", "Daily mean blood glucose"),
    "GSE41848_MS": ("GSE41848 MS discovery", "GSE41848", "classification", "MS versus healthy control"),
    "GSE41849_MS": ("GSE41849 MS replication", "GSE41849", "classification", "MS versus healthy control"),
    "GSE48023_H1N1": ("GSE48023 influenza vaccination", "GSE48023", "regression", "H1N1 HAI response, Day 14 minus Day 0"),
}

AUDIT_SOURCES = {
    "PE": (PROJECT / "data/processed/PE_final.csv", "ID", "time", "label", []),
    "PE_LOPE_MATCHED": (
        PROJECT / "data/external/processed/Stanford_SomaLogic_longitudinal_PE.csv",
        "ID", "GestationalAge", "Label", ["Cohort", "Phenotype", "LatePE", "ClinicalGroup", "MaternalAge", "BMI", "Race", "Gravidity", "Parity"],
    ),
    "FPG": (PROJECT / "data/processed/FPG_final.csv", "ID", "time", "Fasting_plasma_glucose", []),
    "FPG_KARE": (PROJECT / "data/processed/FPG_KARE_final.csv", "ID", "time", "FPG", []),
    "KARE_STATUS": (PROJECT / "data/processed/status_KARE_final.csv", "ID", "time", "status", []),
    "IGG": (PROJECT / "data/processed/IGG_final.csv", "ID", "time", "shannon_diversity", []),
    "IGM": (PROJECT / "data/processed/IGM_final.csv", "ID", "time", "shannon_diversity", []),
    "TRA": (PROJECT / "data/processed/TRA_final.csv", "ID", "time", "shannon_diversity", []),
    "TRB": (PROJECT / "data/processed/TRB_final.csv", "ID", "time", "shannon_diversity", []),
    "OLINK_COVID": (PROJECT / "data/processed/OLINK_COVID_LONGITUDINAL.csv", "ID", "time", "label", ["SampleID"]),
    "BRIST1D": (PROJECT / "data/processed/BRIST1D_DAILY.csv", "ID", "time", "label", []),
    "GSE41848_MS": (PROJECT / "data/processed/GSE41848_MS_LONGITUDINAL.csv", "ID", "time", "label", []),
    "GSE41849_MS": (PROJECT / "data/processed/GSE41849_MS_LONGITUDINAL.csv", "ID", "time", "label", []),
    "GSE48023_H1N1": (PROJECT / "data/processed/GSE48023_H1N1_RESPONSE_LONGITUDINAL.csv", "ID", "time", "label", []),
}


def _holm(p_values: Iterable[float]) -> np.ndarray:
    """Holm-adjust p values while preserving input order."""
    p = np.asarray(list(p_values), dtype=float)
    out = np.full(len(p), np.nan)
    valid = np.flatnonzero(np.isfinite(p))
    if not len(valid):
        return out
    order = valid[np.argsort(p[valid])]
    m = len(order)
    adjusted_sorted = np.maximum.accumulate([(m - i) * p[idx] for i, idx in enumerate(order)])
    out[order] = np.minimum(adjusted_sorted, 1.0)
    return out


def load_completed_runs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metrics, selections, manifest = [], [], []
    for run_name, run_dir in RUNS.items():
        metric_file = run_dir / "repeat_level_metrics.csv"
        compact_file = run_dir / "repeat_metrics.csv"
        selection_file = run_dir / "selected_features.csv"
        if metric_file.exists():
            frame = pd.read_csv(metric_file)
        elif compact_file.exists():
            frame = pd.read_csv(compact_file)
            if "AUROC" not in frame:
                frame["AUROC"] = np.nan
            if "AUPRC" not in frame:
                frame["AUPRC"] = np.nan
            if "RMSE" not in frame:
                frame["RMSE"] = np.nan
            if "MAE" not in frame:
                frame["MAE"] = np.nan
            frame["Metric"] = np.where(frame["AUROC"].notna(), "AUROC", "RMSE")
            frame["Fold"] = 0
        else:
            raise FileNotFoundError(f"Missing completed metrics: {metric_file} or {compact_file}")
        # GSE48023 was initially scored at the visit level even though its
        # vaccine-response outcome is participant-level. Keep the two MS
        # tasks from that original GEO run, but replace GSE48023 with the
        # corrected participant-level rerun below.
        if run_name == "geo_expansion_50":
            dataset_col = "Dataset" if "Dataset" in frame.columns else "Task"
            frame = frame[frame[dataset_col] != "GSE48023_H1N1"].copy()
        frame["SourceRun"] = run_name
        metrics.append(frame)
        manifest.append({
            "SourceRun": run_name,
            "Path": str(run_dir),
            "RepeatMetricRows": len(frame),
            "SelectionFilePresent": selection_file.exists(),
        })
        if selection_file.exists():
            selected = pd.read_csv(selection_file)
            if run_name == "geo_expansion_50":
                dataset_col = "Dataset" if "Dataset" in selected.columns else "Task"
                selected = selected[selected[dataset_col] != "GSE48023_H1N1"].copy()
            selected["SourceRun"] = run_name
            selections.append(selected)

    # Integrate completed PGEE and glmmLasso runs.  Their scripts aggregate
    # repeat-level metrics and selected features into combined CSV files.
    for run_name, run_dir in EXTENSION_RUNS.items():
        metric_file = run_dir / "combined_repeat_metrics.csv"
        selection_file = run_dir / "combined_selected_features.csv"
        if not metric_file.exists():
            raise FileNotFoundError(f"Missing completed extension metrics: {metric_file}")
        frame = pd.read_csv(metric_file)
        frame["SourceRun"] = run_name
        metrics.append(frame)
        manifest.append({
            "SourceRun": run_name,
            "Path": str(run_dir),
            "RepeatMetricRows": len(frame),
            "SelectionFilePresent": selection_file.exists(),
        })
        if selection_file.exists():
            selected = pd.read_csv(selection_file)
            selected["SourceRun"] = run_name
            selections.append(selected)
    return pd.concat(metrics, ignore_index=True), pd.concat(selections, ignore_index=True), pd.DataFrame(manifest)


def add_metadata(frame: pd.DataFrame) -> pd.DataFrame:
    metadata = pd.DataFrame.from_dict(
        TASK_METADATA, orient="index", columns=["Dataset", "Cohort", "TaskType", "OutcomeDescription"],
    ).rename_axis("Task").reset_index()
    return frame.rename(columns={"Dataset": "Task"}).merge(metadata, on="Task", how="left", validate="many_to_one")


def metric_column(task_type: str) -> str:
    return "AUROC" if task_type == "classification" else "RMSE"


def task_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    records = []
    for (task, dataset, cohort, task_type, outcome, method), group in metrics.groupby(
        ["Task", "Dataset", "Cohort", "TaskType", "OutcomeDescription", "Method"], dropna=False,
    ):
        primary = metric_column(task_type)
        row = {
            "Task": task, "Dataset": dataset, "Cohort": cohort, "TaskType": task_type,
            "Outcome": outcome, "Method": method, "Metric": primary,
            "Mean": group[primary].mean(), "SD": group[primary].std(ddof=1), "Repeats": group["Repeat"].nunique(),
        }
        secondary = "AUPRC" if primary == "AUROC" else "MAE"
        if secondary in group:
            row[f"Mean{secondary}"] = group[secondary].mean()
            row[f"SD{secondary}"] = group[secondary].std(ddof=1)
        records.append(row)
    return pd.DataFrame(records)


def rank_and_statistics(metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rank_records, pairwise_records, friedman_records, global_rows = [], [], [], []
    for task, group in metrics.groupby("Task"):
        task_type = group["TaskType"].iloc[0]
        primary = metric_column(task_type)
        wide = group.pivot(index="Repeat", columns="Method", values=primary)
        available = [m for m in METHOD_ORDER if m in wide.columns and wide[m].notna().all()]
        wide = wide[available].dropna()
        if wide.empty or len(available) < 2:
            continue
        ranks = wide.rank(axis=1, ascending=(primary == "RMSE"), method="average")
        for method in wide.columns:
            rank_records.append({
                "Task": task, "Dataset": group["Dataset"].iloc[0], "Cohort": group["Cohort"].iloc[0],
                "TaskType": task_type, "Method": method, "Metric": primary,
                "MeanRank": ranks[method].mean(), "SDRank": ranks[method].std(ddof=1),
                "WinRate": (ranks[method] == 1).mean(), "Repeats": len(ranks),
            })
        try:
            stat, p_value = friedmanchisquare(*[wide[m].to_numpy() for m in wide.columns])
        except ValueError:
            stat, p_value = np.nan, np.nan
        friedman_records.append({"Scope": task, "Level": "repeat", "Methods": len(wide.columns), "Blocks": len(wide), "ChiSquare": stat, "PValue": p_value})
        for i, method_a in enumerate(wide.columns):
            for method_b in wide.columns[i + 1:]:
                oriented_difference = (wide[method_a] - wide[method_b]) if primary == "AUROC" else (wide[method_b] - wide[method_a])
                try:
                    _, p_value = wilcoxon(oriented_difference, zero_method="wilcox", alternative="two-sided")
                except ValueError:
                    p_value = 1.0
                pairwise_records.append({
                    "Scope": task, "TaskType": task_type, "Metric": primary,
                    "MethodA": method_a, "MethodB": method_b,
                    "MeanDifference_AminusB_oriented": oriented_difference.mean(),
                    "MedianDifference_AminusB_oriented": oriented_difference.median(),
                    "NPairedRepeats": len(oriented_difference), "PValue": p_value,
                })
        means = wide.mean()
        global_rows.append({"Task": task, "TaskType": task_type, "Metric": primary, **means.to_dict()})

    pairwise = pd.DataFrame(pairwise_records)
    if not pairwise.empty:
        pairwise["PValueHolmWithinTask"] = pairwise.groupby("Scope")["PValue"].transform(_holm)

    # The primary rank estimand is the mean paired-repeat rank for each
    # task-method combination, then its unweighted average across tasks.
    # This preserves the repeated-split pairing and matches the reported
    # rank SD and Friedman analysis; it does not re-rank only task means.
    task_ranks = pd.DataFrame(rank_records)
    if not task_ranks.empty:
        overall = task_ranks.groupby("Method", as_index=False).agg(
            MeanRank=("MeanRank", "mean"), SDRank=("MeanRank", "std"),
            WinRate=("WinRate", "mean"), Tasks=("Task", "count"),
        ).sort_values("MeanRank")
        mean_scopes = [("All tasks", task_ranks)] + [
            (f"{kind.capitalize()} tasks", frame)
            for kind, frame in task_ranks.groupby("TaskType")
        ]
        for scope, subset in mean_scopes:
            rank_matrix = subset.pivot(index="Task", columns="Method", values="MeanRank")
            available = [method for method in METHOD_ORDER if method in rank_matrix and rank_matrix[method].notna().all()]
            rank_matrix = rank_matrix[available].dropna()
            if len(rank_matrix) >= 2 and len(available) >= 2:
                stat, p_value = friedmanchisquare(*[rank_matrix[method].to_numpy() for method in available])
                friedman_records.append({"Scope": scope, "Level": "task", "Methods": len(available), "Blocks": len(rank_matrix), "ChiSquare": stat, "PValue": p_value})
    else:
        overall = pd.DataFrame()
    return task_ranks, pairwise, pd.DataFrame(friedman_records), overall


def audit_datasets() -> pd.DataFrame:
    records = []
    for task, (path, id_col, time_col, outcome_col, extras) in AUDIT_SOURCES.items():
        if not path.exists():
            raise FileNotFoundError(f"Data file for audit is missing: {path}")
        frame = pd.read_csv(path)
        if task == "PE_LOPE_MATCHED":
            frame = frame[(frame["Phenotype"] != "PE-early") & (frame["GestationalAge"] <= 22)].copy()
        non_features = {id_col, time_col, outcome_col, *extras}
        feature_count = len([c for c in frame.columns if c not in non_features])
        # The matched-assay PE task deliberately restricts Stanford to the
        # 1,116 SomaId intersection available in Detroit; its raw Stanford
        # table has additional assays that are not candidate features here.
        if task == "PE_LOPE_MATCHED":
            feature_count = 1116
        visits_per_subject = frame.groupby(id_col).size()
        times = pd.to_numeric(frame[time_col], errors="coerce")
        dataset, cohort, task_type, outcome = TASK_METADATA[task]
        records.append({
            "Task": task, "Dataset": dataset, "Cohort": cohort, "TaskType": task_type, "Outcome": outcome,
            "DataFile": str(path), "Observations": len(frame), "Participants": frame[id_col].nunique(),
            "UniqueTimeValues": frame[time_col].nunique(), "MedianVisitsPerParticipant": visits_per_subject.median(),
            "MinVisitsPerParticipant": visits_per_subject.min(), "MaxVisitsPerParticipant": visits_per_subject.max(),
            "FeatureCandidates": feature_count, "TimeMinimum": times.min(), "TimeMaximum": times.max(),
        })
    return pd.DataFrame(records)


def task_winners(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for task, group in summary.groupby("Task"):
        winner = group.loc[group["Mean"].idxmax()] if group["Metric"].iloc[0] == "AUROC" else group.loc[group["Mean"].idxmin()]
        rows.append(winner)
    return pd.DataFrame(rows).sort_values(["TaskType", "Task"])


def structural_summary(task_ranks: pd.DataFrame, audit: pd.DataFrame) -> pd.DataFrame:
    """Descriptive method ranks stratified by observed dataset structure.

    This is deliberately exploratory: eleven tasks are insufficient for causal
    claims about why a method wins.  It creates transparent strata for the
    manuscript discussion without over-interpreting them.
    """
    table = task_ranks.merge(audit[["Task", "Participants", "FeatureCandidates", "MedianVisitsPerParticipant"]], on="Task", how="left")
    table["ParticipantStratum"] = pd.cut(table.Participants, [0, 100, 500, np.inf], labels=["<100", "100–500", ">500"])
    table["FeatureStratum"] = pd.cut(table.FeatureCandidates, [0, 100, 300, np.inf], labels=["<100", "100–300", ">300"])
    table["FollowUpStratum"] = pd.cut(table.MedianVisitsPerParticipant, [0, 2, 4, np.inf], labels=["≤2", "2–4", ">4"])
    records = []
    for variable in ["TaskType", "ParticipantStratum", "FeatureStratum", "FollowUpStratum"]:
        grouped = table.groupby([variable, "Method"], observed=True)
        for (stratum, method), frame in grouped:
            records.append({
                "Stratifier": variable, "Stratum": str(stratum), "Method": method,
                "MeanRank": frame.MeanRank.mean(), "SDRank": frame.MeanRank.std(ddof=1), "Tasks": frame.Task.nunique(),
            })
    return pd.DataFrame(records).sort_values(["Stratifier", "Stratum", "MeanRank"])


def write_interpretation(
    out: Path, overall: pd.DataFrame, winners: pd.DataFrame,
    friedman: pd.DataFrame, metrics: pd.DataFrame,
) -> None:
    global_test = friedman[(friedman.Scope == "All tasks") & (friedman.Level == "task")]
    global_p = float(global_test.PValue.iloc[0]) if len(global_test) else np.nan
    global_methods = int(global_test.Methods.iloc[0]) if len(global_test) else 0
    global_blocks = int(global_test.Blocks.iloc[0]) if len(global_test) else 0
    if np.isfinite(global_p) and global_p < 0.05:
        global_statement = (
            f"The task-level Friedman test for the {global_methods} methods available across all "
            f"{global_blocks} tasks gave p = {global_p:.3g}, indicating an overall difference in average ranks. "
            "However, the winner changed across tasks and source cohorts, so this does not establish a universally dominant method."
        )
    else:
        global_statement = (
            f"The task-level Friedman test for the {global_methods} methods available across all "
            f"{global_blocks} tasks gave p = {global_p:.3g}; it did not detect an overall rank difference. "
            "The winner nevertheless varied across tasks and source cohorts."
        )
    lines = [
        "# Final benchmark interpretation",
        "",
        "## Scope",
        "",
        f"The primary evidence comprises {metrics.Task.nunique()} cohort–outcome tasks from {metrics.Cohort.nunique()} source cohorts, evaluated with {metrics.Method.nunique()} feature-selection methods and 50 repeated subject-level outer splits per task.",
        "The locked Stanford-to-Detroit transfer analysis is retained separately because it does not share the repeated internal-split design.",
        "",
        "## Main finding",
        global_statement,
        "The supported conclusion is task-dependent performance rather than universal superiority; a non-winning method on average may still be best for a particular data structure.",
        "",
        "## Best method by task",
        "",
        "| Task | Outcome type | Best method | Mean | SD |",
        "|---|---|---:|---:|---:|",
    ]
    for _, row in winners.iterrows():
        lines.append(f"| {row.Task} | {row.TaskType} ({row.Metric}) | {row.Method} | {row.Mean:.4f} | {row.SD:.4f} |")
    lines += [
        "",
        "## Interpretation guardrails",
        "",
        "- Repeat-level Friedman and Wilcoxon tests characterize paired split-to-split differences; repeated splits are not fully independent biological replications.",
        f"- The structural-stratum table is descriptive only. With {metrics.Task.nunique()} tasks, it supports hypothesis generation rather than causal statements about why a method performs best.",
        "- BrisT1D has only 20 participants despite many daily observations; it should be interpreted as a dense-within-subject stress test, not as broad population validation.",
        "- SNN-FS and OutcomeSNN-FS are retained as proposed comparators. The results do not support claiming that either is uniformly superior.",
    ]
    (out / "benchmark_interpretation.md").write_text("\n".join(lines) + "\n")


def feature_frequency(selections: pd.DataFrame) -> pd.DataFrame:
    if selections.empty:
        return selections
    total = selections.groupby(["Task", "Method"])["Repeat"].nunique().rename("Repeats").reset_index()
    frequency = selections.groupby(["Task", "Method", "Feature"], as_index=False).agg(
        SelectionEvents=("Feature", "size"), RepeatsSelected=("Repeat", "nunique"), MeanRank=("Rank", "mean"),
    ).merge(total, on=["Task", "Method"], how="left")
    frequency["RepeatSelectionFrequency"] = frequency["RepeatsSelected"] / frequency["Repeats"]
    return frequency.sort_values(["Task", "Method", "RepeatSelectionFrequency", "MeanRank"], ascending=[True, True, False, True])


def _plot_task_panels(summary: pd.DataFrame, task_type: str, output: Path) -> None:
    table = summary[summary.TaskType == task_type].copy()
    tasks = list(table.Task.drop_duplicates())
    if not tasks:
        return
    ncols = min(3, len(tasks))
    nrows = int(np.ceil(len(tasks) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.7 * ncols, 6.0 * nrows), squeeze=False)
    metric = "AUROC" if task_type == "classification" else "RMSE"
    for ax, task in zip(axes.flat, tasks):
        item = table[table.Task == task].set_index("Method").reindex(METHOD_ORDER).dropna(subset=["Mean"])
        item = item.sort_values("Mean", ascending=(metric == "AUROC"))
        y = np.arange(len(item))
        ax.barh(y, item.Mean, xerr=item.SD, capsize=2.5, alpha=0.92,
                color=[METHOD_COLORS.get(method, "#808080") for method in item.index])
        ax.set_yticks(y, item.index, fontsize=9)
        ax.set_xlabel(f"{metric} (50-repeat mean ± SD)")
        ax.set_title(f"{task}: {item.Dataset.iloc[0]}", fontsize=12, weight="bold")
        ax.grid(axis="x", alpha=0.25)
        if metric == "AUROC":
            ax.set_xlim(0.5, 1.0)
        else:
            values = item["Mean"].to_numpy(float)
            errors = item["SD"].fillna(0).to_numpy(float)
            vmin, vmax = float(np.nanmin(values)), float(np.nanmax(values))
            spread = max(vmax - vmin, 1e-8)
            pad = max(0.10 * spread, 1.5 * float(np.nanmax(errors)))
            # Use a task-specific zoomed range; do not force RMSE axes to zero.
            ax.set_xlim(vmin - pad, vmax + pad)
        ax.invert_yaxis()
    for ax in axes.flat[len(tasks):]:
        ax.axis("off")
    title = "Classification benchmark across longitudinal tasks" if task_type == "classification" else "Regression benchmark across longitudinal tasks"
    fig.suptitle(title, fontsize=17, weight="bold", y=1.01)
    fig.tight_layout()
    stem = output / ("figure_main_classification_auroc" if task_type == "classification" else "figure_main_regression_rmse")
    fig.savefig(stem.with_suffix(".png"), dpi=350, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=RESULTS_ROOT / "main_benchmark" / "final_benchmark")
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    raw_metrics, selections, manifest = load_completed_runs()
    metrics = add_metadata(raw_metrics)
    if metrics[["Dataset", "Cohort", "TaskType"]].isna().any().any():
        missing = metrics.loc[metrics.Dataset.isna(), "Task"].unique().tolist()
        raise ValueError(f"Missing task metadata: {missing}")
    selections = add_metadata(selections) if not selections.empty else selections
    summary = task_summary(metrics)
    task_ranks, pairwise, friedman, overall = rank_and_statistics(metrics)
    audit = audit_datasets()
    frequencies = feature_frequency(selections)
    winners = task_winners(summary)
    structural = structural_summary(task_ranks, audit)

    metrics.to_csv(out / "repeat_level_metrics_all_tasks.csv", index=False)
    summary.to_csv(out / "final_task_performance_summary.csv", index=False)
    task_ranks.to_csv(out / "task_level_rank_summary.csv", index=False)
    overall.to_csv(out / "overall_method_rank_summary.csv", index=False)
    pairwise.to_csv(out / "paired_wilcoxon_holm.csv", index=False)
    friedman.to_csv(out / "friedman_tests.csv", index=False)
    audit.to_csv(out / "dataset_longitudinal_audit.csv", index=False)
    winners.to_csv(out / "task_winner_summary.csv", index=False)
    structural.to_csv(out / "method_rank_by_data_structure.csv", index=False)
    frequencies.to_csv(out / "selected_feature_frequency_all_tasks.csv", index=False)
    selections.to_csv(out / "selected_features_all_tasks.csv", index=False)
    manifest.to_csv(out / "reproducibility_manifest.csv", index=False)
    with (out / "benchmark_scope.json").open("w") as handle:
        json.dump({
            "unit_of_analysis": "cohort-outcome task",
            "completed_runs": {name: str(path) for name, path in RUNS.items()},
            "methods": METHOD_ORDER,
            "main_figures": ["classification AUROC", "regression RMSE"],
            "external_validation": str(RESULTS_ROOT / "validation" / "external_validation"),
            "external_validation_note": "Locked Detroit transfer is indexed separately and not pooled with repeated internal benchmark estimates.",
        }, handle, indent=2)
    _plot_task_panels(summary, "classification", out)
    _plot_task_panels(summary, "regression", out)
    write_interpretation(out, overall, winners, friedman, metrics)
    print(f"Wrote final benchmark package to {out}")
    print(f"Tasks: {metrics.Task.nunique()}; source cohorts: {metrics.Cohort.nunique()}; methods: {metrics.Method.nunique()}")


if __name__ == "__main__":
    main()
