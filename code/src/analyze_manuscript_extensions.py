from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr


ROOT = Path("/home/zliu/ctsnn/results/main_benchmark/final_benchmark")
SRC = ROOT
OUT = ROOT / "manuscript_extensions"
OUT.mkdir(parents=True, exist_ok=True)

METHOD_ORDER = [
    "Lasso", "ElasticNet", "StabilitySelection", "Stabl-RP",
    "LongGroupLasso", "LLSS", "RandomForest", "MutualInfo", "Boruta",
    "mRMR", "SNN-FS", "OutcomeSNN-FS", "geeVerse",
]

METHOD_CLASS = {
    "Lasso": "General-purpose",
    "ElasticNet": "General-purpose",
    "StabilitySelection": "Resampling",
    "Stabl-RP": "Resampling",
    "LongGroupLasso": "Longitudinal-aware",
    "LLSS": "Longitudinal-aware",
    "RandomForest": "Tree-based",
    "MutualInfo": "Filter",
    "Boruta": "Tree-based",
    "mRMR": "Filter",
    "SNN-FS": "Proposed",
    "OutcomeSNN-FS": "Proposed",
    "geeVerse": "Longitudinal-aware",
}


def bh(values):
    p = np.asarray(values, float)
    order = np.argsort(p)
    ranked = p[order]
    adjusted = np.minimum.accumulate(
        (ranked * len(ranked) / np.arange(1, len(ranked) + 1))[::-1]
    )[::-1]
    out = np.empty_like(adjusted)
    out[order] = np.minimum(adjusted, 1.0)
    return out


audit = pd.read_csv(SRC / "dataset_longitudinal_audit.csv")
ranks = pd.read_csv(SRC / "task_level_rank_summary.csv")
stability = pd.read_csv(SRC / "selection_stability.csv")
performance = pd.read_csv(SRC / "final_task_performance_summary.csv")

# One task-method row. MeanRank is the average paired-repeat rank and is the
# scale used for the exploratory structure analysis.
task_rank = ranks[["Task", "Cohort", "TaskType", "Method", "MeanRank", "SDRank", "WinRate"]].copy()
task_rank = task_rank.merge(
    audit[[
        "Task", "Participants", "Observations", "FeatureCandidates",
        "MedianVisitsPerParticipant", "MaxVisitsPerParticipant",
    ]], on="Task", how="left", validate="many_to_one",
)
task_rank["LogParticipants"] = np.log10(task_rank["Participants"])
task_rank["LogFeatureParticipantRatio"] = np.log10(
    task_rank["FeatureCandidates"] / task_rank["Participants"]
)
task_rank["LogMedianVisits"] = np.log10(task_rank["MedianVisitsPerParticipant"])

characteristics = {
    "Participants": "LogParticipants",
    "Feature to participant ratio": "LogFeatureParticipantRatio",
    "Visits per participant": "LogMedianVisits",
}
assoc_rows = []
for method, group in task_rank.groupby("Method"):
    for label, column in characteristics.items():
        use = group[["MeanRank", column]].dropna()
        rho, p = spearmanr(use[column], use["MeanRank"])
        assoc_rows.append({
            "Method": method, "DatasetCharacteristic": label,
            "SpearmanRhoWithRank": rho, "PValue": p, "Tasks": len(use),
        })
assoc = pd.DataFrame(assoc_rows)
assoc["BHAdjustedP"] = bh(assoc["PValue"].fillna(1.0))
assoc.to_csv(OUT / "exploratory_data_structure_associations.csv", index=False)

# A rank heat map is more comparable across outcomes than raw AUROC/RMSE.
mean_perf = performance.pivot(index="Method", columns="Task", values="Mean")
task_type = performance.drop_duplicates("Task").set_index("Task")["TaskType"]
rank_matrix = mean_perf.copy()
for task in rank_matrix.columns:
    rank_matrix[task] = rank_matrix[task].rank(
        ascending=(task_type[task] == "regression"), method="average"
    )
rank_matrix = rank_matrix.reindex(METHOD_ORDER)
task_order = [
    *audit.loc[audit.TaskType.eq("classification"), "Task"],
    *audit.loc[audit.TaskType.eq("regression"), "Task"],
]
rank_matrix = rank_matrix.reindex(columns=task_order)
rank_matrix.to_csv(OUT / "task_method_rank_matrix.csv")

sns.set_theme(style="whitegrid", font_scale=1.0)
fig, ax = plt.subplots(figsize=(14.5, 7.3))
sns.heatmap(
    rank_matrix, cmap="viridis_r", vmin=1, vmax=13, linewidths=0.5,
    linecolor="white", annot=True, fmt=".1f", cbar_kws={"label": "Rank (lower is better)"}, ax=ax,
)
ax.set_xlabel("Cohort-outcome task")
ax.set_ylabel("")
ax.set_title("Method ranks vary across longitudinal biomedical tasks", weight="bold", pad=12)
ax.tick_params(axis="x", rotation=45, labelsize=9)
fig.tight_layout()
fig.savefig(OUT / "figure_task_method_rank_heatmap.png", dpi=350, bbox_inches="tight")
fig.savefig(OUT / "figure_task_method_rank_heatmap.pdf", bbox_inches="tight")
plt.close(fig)

# Performance-stability trade-off. Stability is averaged over tasks and the
# prespecified panel-size views (3, 5, 10); both axes are descriptive.
stab_summary = stability.groupby("Method", as_index=False).agg(
    MeanJaccard=("MeanJaccard", "mean"), SDJaccard=("MeanJaccard", "std"),
    StabilityCells=("MeanJaccard", "size"),
)
rank_summary = task_rank.groupby("Method", as_index=False).agg(
    MeanRank=("MeanRank", "mean"), SDRankAcrossTasks=("MeanRank", "std"),
    Tasks=("Task", "nunique"),
)
trade = rank_summary.merge(stab_summary, on="Method", how="left")
trade["MethodClass"] = trade["Method"].map(METHOD_CLASS)
trade.to_csv(OUT / "performance_stability_tradeoff.csv", index=False)

palette = {
    "General-purpose": "#7F7F7F", "Resampling": "#E69F00",
    "Longitudinal-aware": "#009E73", "Tree-based": "#CC79A7",
    "Filter": "#8C6BB1", "Proposed": "#0072B2",
}
fig, ax = plt.subplots(figsize=(9.5, 7.0))
for family, group in trade.groupby("MethodClass"):
    ax.scatter(
        group["MeanJaccard"], group["MeanRank"], s=90,
        color=palette[family], label=family, alpha=0.9, edgecolor="white", linewidth=0.7,
    )
for _, row in trade.iterrows():
    ax.annotate(
        row.Method, (row.MeanJaccard, row.MeanRank), xytext=(5, 4),
        textcoords="offset points", fontsize=8.5,
    )
ax.invert_yaxis()
ax.set_xlabel("Mean pairwise Jaccard stability (higher is better)")
ax.set_ylabel("Mean task rank (lower is better)")
ax.set_title("Predictive rank and selection stability are distinct objectives", weight="bold", pad=12)
ax.grid(alpha=0.25)
ax.legend(frameon=False, ncol=2, fontsize=9, loc="lower right")
fig.tight_layout()
fig.savefig(OUT / "figure_performance_stability_tradeoff.png", dpi=350, bbox_inches="tight")
fig.savefig(OUT / "figure_performance_stability_tradeoff.pdf", bbox_inches="tight")
plt.close(fig)

# Heat map of descriptive Spearman associations. Positive rho means worse rank
# as the characteristic increases; no causal interpretation is intended.
assoc_matrix = assoc.pivot(
    index="Method", columns="DatasetCharacteristic", values="SpearmanRhoWithRank"
).reindex(METHOD_ORDER)
fig, ax = plt.subplots(figsize=(8.4, 7.0))
sns.heatmap(
    assoc_matrix, cmap="vlag", center=0, vmin=-1, vmax=1, annot=True,
    fmt=".2f", linewidths=0.5, linecolor="white",
    cbar_kws={"label": "Spearman rho with rank"}, ax=ax,
)
ax.set_xlabel("")
ax.set_ylabel("")
ax.set_title("Exploratory associations between data structure and method rank", weight="bold", pad=12)
fig.tight_layout()
fig.savefig(OUT / "figure_data_structure_associations.png", dpi=350, bbox_inches="tight")
fig.savefig(OUT / "figure_data_structure_associations.pdf", bbox_inches="tight")
plt.close(fig)

# Empirical decision guide. Only strata containing at least two tasks are
# retained, and the three best mean ranks are reported without causal claims.
guide_rows = []
strata = {
    "Classification outcome": task_rank.TaskType.eq("classification"),
    "Continuous or ordinal outcome": task_rank.TaskType.eq("regression"),
    "At most two median visits": task_rank.MedianVisitsPerParticipant.le(2),
    "Three to four median visits": task_rank.MedianVisitsPerParticipant.between(3, 4),
    "More than four median visits": task_rank.MedianVisitsPerParticipant.gt(4),
    "Feature to participant ratio below one": (
        task_rank.FeatureCandidates / task_rank.Participants
    ).lt(1),
    "Feature to participant ratio at least one": (
        task_rank.FeatureCandidates / task_rank.Participants
    ).ge(1),
}
for stratum, mask in strata.items():
    subset = task_rank.loc[mask].copy()
    tasks = subset.Task.nunique()
    if tasks < 2:
        continue
    ordered = subset.groupby("Method", as_index=False).agg(
        MeanRank=("MeanRank", "mean"), Tasks=("Task", "nunique"),
    )
    ordered = ordered[ordered.Tasks >= 2].sort_values("MeanRank").head(3)
    guide_rows.append({
        "DataSetting": stratum, "Tasks": tasks,
        "BestObserved": "; ".join(
            f"{row.Method} ({row.MeanRank:.2f})" for row in ordered.itertuples()
        ),
        "Interpretation": "Exploratory benchmark guidance; validate within the target cohort.",
    })
pd.DataFrame(guide_rows).to_csv(OUT / "empirical_decision_guide.csv", index=False)

# Transparent implementation audit used directly in the manuscript.
audit_rows = [
    ("Lasso", "L1-penalized model", "scikit-learn estimator", "Library implementation", "No", "Equal visit weights; participant-level splitting"),
    ("ElasticNet", "Elastic net", "scikit-learn estimator", "Library implementation", "No", "Equal visit weights; participant-level splitting"),
    ("StabilitySelection", "Stability selection", "Project wrapper using scikit-learn sparse models", "Compatible reimplementation", "Partial", "Participants, not visits, are subsampled"),
    ("Stabl-RP", "Stabl random-permutation logic", "Project lightweight RP implementation", "Compatible reimplementation", "Partial", "Participant subsampling plus permuted artificial features"),
    ("LongGroupLasso", "Grouped between/within effects", "Project proximal-gradient implementation", "Project adaptation", "Yes", "Pairs between-participant means with within-participant deviations"),
    ("LLSS", "Longitudinal-LASSO-inspired stability selector", "Project implementation", "Project adaptation", "Yes", "Participant-level resampling; not the official Longitudinal LASSO implementation"),
    ("RandomForest", "Tree ensemble importance", "scikit-learn ExtraTrees", "Library implementation under benchmark label", "No", "Benchmark label retained for continuity; implementation is ExtraTrees"),
    ("MutualInfo", "Mutual-information filter", "scikit-learn estimators", "Library implementation", "No", "Outcome association only"),
    ("Boruta", "Boruta shadow-feature principle", "Project ExtraTrees shadow-feature ranking", "Compatible reimplementation", "No", "Fixed-panel ranking rather than the official Boruta accept/reject procedure"),
    ("mRMR", "Minimum-redundancy maximum-relevance", "Project greedy relevance-minus-correlation ranking", "Compatible reimplementation", "No", "Uses univariate relevance and mean absolute-correlation redundancy"),
    ("SNN-FS", "Shared-nearest-neighbor observation weighting", "Project implementation", "Proposed method", "No", "Weighted elastic-net feature ranking"),
    ("OutcomeSNN-FS", "Outcome-coherent SNN weighting", "Project implementation", "Proposed method", "No", "SNN support modulated by local outcome coherence"),
    ("geeVerse", "Penalized quantile GEE", "Official CRAN geeVerse package", "Official package adapter", "Yes", "Regression tasks only; HBIC-tuned qpgee ranking"),
]
columns = ["BenchmarkLabel", "MethodFamily", "SoftwareUsed", "ImplementationStatus", "LongitudinalAware", "BenchmarkAdaptation"]
pd.DataFrame(audit_rows, columns=columns).to_csv(OUT / "method_implementation_audit.csv", index=False)

print("Wrote manuscript analyses to", OUT)
print(trade.sort_values("MeanRank").to_string(index=False))
print("\nExploratory decision guide\n", pd.DataFrame(guide_rows).to_string(index=False))
