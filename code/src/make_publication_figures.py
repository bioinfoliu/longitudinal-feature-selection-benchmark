"""Generate the manuscript figures from machine-readable result tables."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "figures_v2"
OUT.mkdir(parents=True, exist_ok=True)
COLORS = {
    "SNN": "#0072B2", "SNN-FS": "#0072B2",
    "TrajSNN": "#009E73", "TrajSNN-FS": "#009E73",
    "STAR": "#56B4E9", "w/o Traj": "#56B4E9",
    "Hybrid": "#009E73", "w/o Robust": "#009E73",
    "CTSNN-R": "#009E73", "CTSNN": "#009E73", "LLSS": "#D55E00", "Lasso": "#777777",
    "LongGroupLasso": "#CC79A7", "Stabl-RP": "#E69F00",
    "SNN-FS": "#0072B2", "CTSNN": "#009E73",
}

# Older result files retain historical keys for reproducibility. The
# publication-facing figures use the manuscript's canonical display names.
DISPLAY_METHOD = {
    "SNN": "SNN-FS", "SNN-FS": "SNN-FS",
    "TrajSNN": "TrajSNN", "TrajSNN-FS": "TrajSNN",
    "STAR": "w/o Traj", "w/o Traj": "w/o Traj",
    "Hybrid": "TrajSNN", "w/o Robust": "TrajSNN",
    "CTSNN-R": "CTSNN", "CTSNN": "CTSNN",
    "LongGroupLasso": "LongGroupLasso",
}


def display_methods(values):
    return [DISPLAY_METHOD.get(value, value) for value in values]


def canonicalize_method_column(frame):
    frame = frame.copy()
    frame["Method"] = frame["Method"].map(DISPLAY_METHOD).fillna(frame["Method"])
    return frame


def save(fig, stem):
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def method_schematic():
    fig, ax = plt.subplots(figsize=(16.4, 7.1))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    blue = "#156082"
    navy = "#17324D"
    light = "#F7F9FA"
    edge = "#2F4655"
    orange = "#D95F02"
    green = "#4DAF4A"

    def rounded_box(x, y, w, h, radius=0.04, lw=1.6):
        patch = FancyBboxPatch(
            (x, y), w, h,
            boxstyle=f"round,pad=0.012,rounding_size={radius}",
            facecolor=light,
            edgecolor=edge,
            linewidth=lw,
            linestyle="--",
        )
        ax.add_patch(patch)
        return patch

    def arrow(x0, y0, x1, y1, label=None, label_pos=None):
        ax.annotate(
            "",
            xy=(x1, y1),
            xytext=(x0, y0),
            arrowprops=dict(arrowstyle="-|>", color=blue, lw=2.4, mutation_scale=18),
        )
        if label:
            lx, ly = label_pos if label_pos is not None else ((x0 + x1) / 2, (y0 + y1) / 2 + 0.035)
            ax.text(lx, ly, label,
                    ha="center", va="bottom", fontsize=14, color=navy, weight="bold")

    ax.text(0.5, 0.955, "TrajSNN workflow",
            ha="center", va="top", fontsize=20, weight="bold", color=navy)

    # Longitudinal input panel.
    x0, y0, w0, h0 = 0.030, 0.36, 0.175, 0.33
    t = np.array([0.0, 0.34, 0.68, 1.0])
    colors = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#999999"]
    for idx, c in enumerate(colors):
        values = np.array([
            0.20 + 0.08 * idx,
            0.55 - 0.06 * idx + 0.12 * (idx % 2),
            0.40 + 0.07 * idx,
            0.28 + 0.05 * ((idx + 2) % 4),
        ])
        xs = x0 + 0.025 + t * (w0 - 0.05)
        ys = y0 + 0.05 + np.clip(values, 0.05, 0.90) * (h0 - 0.10)
        ax.plot(xs, ys, color=c, lw=1.9, marker="o", ms=5, alpha=0.95)
    ax.plot([x0 + 0.02, x0 + w0 - 0.02], [y0 + 0.05, y0 + 0.05], color="#666666", lw=0.9)
    ax.plot([x0 + 0.02, x0 + 0.02], [y0 + 0.05, y0 + h0 - 0.02], color="#666666", lw=0.9)
    for k in range(4):
        ax.text(x0 + 0.025 + t[k] * (w0 - 0.05), y0 + 0.005, str(k + 1),
                ha="center", va="top", fontsize=10, color="#333333")
    ax.text(x0 + w0 / 2, y0 - 0.060, "Longitudinal data", ha="center", fontsize=14, weight="bold")
    ax.text(x0 + w0 / 2, y0 - 0.092, "visits from the same participant", ha="center", fontsize=11)
    ax.text(x0 + w0 / 2, y0 + h0 + 0.035, "Measurement over time", ha="center", fontsize=12, color=navy)

    # SNN neighborhood panel.
    rounded_box(0.285, 0.23, 0.215, 0.53)
    center = np.array([0.392, 0.535])
    pts = np.array([
        [0.350, 0.46], [0.372, 0.66], [0.425, 0.62], [0.448, 0.47],
        [0.363, 0.35], [0.437, 0.34], [0.405, 0.76], [0.467, 0.60],
    ])
    point_colors = ["#9EC3E6", "#D62728", "#D62728", "#D62728", "#9EC3E6", "#D62728", "#8BD17C", "#8BD17C"]
    for p, c in zip(pts, point_colors):
        ax.plot([center[0], p[0]], [center[1], p[1]], color=c, ls="--", lw=1.0, alpha=0.75)
        ax.add_patch(Circle(p, 0.008, facecolor=c, edgecolor="white", lw=0.8, alpha=0.95))
    ax.add_patch(Circle(center, 0.010, facecolor="#1F77B4", edgecolor="black", lw=1.0))
    ax.text(0.392, 0.295, r"$r_i=\frac{1}{k}\sum_{j\in N_k(i)}|N_k(i)\cap N_k(j)|$" + "\n" + r"$w_i^{\mathrm{SNN}}=\operatorname{scale}_{\epsilon}(r_i)$",
            ha="center", va="bottom", fontsize=11.0,
            bbox=dict(facecolor=light, edgecolor="none", pad=1.5))
    ax.text(0.392, 0.200, "Each point: observation\nRed links: shared neighbors",
            ha="center", va="top", fontsize=11)
    arrow(0.207, 0.535, 0.282, 0.535, "SNN-FS\nsimilarity")

    # Trajectory score panel.
    rounded_box(0.590, 0.590, 0.280, 0.285)
    xt = 0.640 + np.array([0.0, 0.05, 0.10, 0.15])
    trajectory_a = 0.67 + np.array([0.00, 0.05, 0.08, 0.13])
    trajectory_b = 0.77 + np.array([0.00, -0.02, -0.04, -0.06])
    ax.plot(xt, trajectory_a, color="#4DAF4A", lw=2.0, marker="o", ms=5)
    ax.plot(xt, trajectory_b, color="#22B8C7", lw=2.0, marker="o", ms=5)
    ax.text(0.730, 0.842, "Trajectory-aware feature score", ha="center", fontsize=12, weight="bold")
    ax.text(0.730, 0.660, r"$m_f=\mathrm{score}(\bar{x}_{gf},y_g)$",
            ha="center", va="top", fontsize=11.8,
            bbox=dict(facecolor=light, edgecolor="none", pad=1.0))
    ax.text(0.730, 0.615, r"$b_f=\mathrm{score}(\mathrm{slope}_{gf},y_g)$",
            ha="center", va="top", fontsize=10.5,
            bbox=dict(facecolor=light, edgecolor="none", pad=1.0))
    arrow(0.502, 0.610, 0.588, 0.715, "Trajectory\nscore", label_pos=(0.535, 0.790))

    # SNN and trajectory fusion panel.
    rounded_box(0.590, 0.135, 0.280, 0.300)
    ax.text(0.730, 0.405, "SNN + trajectory ranking", ha="center", fontsize=12, weight="bold")
    ax.text(0.730, 0.350, r"$a_f=|\hat{\beta}_f(w_i^{\mathrm{SNN}})|$",
            ha="center", va="top", fontsize=11.2,
            bbox=dict(facecolor=light, edgecolor="none", pad=1.0))
    ax.text(0.730, 0.285, r"$t_f=\max(m_f,b_f)$",
            ha="center", va="top", fontsize=10.8,
            bbox=dict(facecolor=light, edgecolor="none", pad=1.0))
    ax.text(0.730, 0.205, r"$R_f=\mathrm{rank\ fusion}(a_f,t_f)$",
            ha="center", va="top", fontsize=10.8,
            bbox=dict(facecolor=light, edgecolor="none", pad=1.0))
    arrow(0.502, 0.460, 0.588, 0.300, "Feature\nranking", label_pos=(0.535, 0.475))

    # Final aggregation panel.
    ax.plot([0.872, 0.902, 0.902, 0.872], [0.730, 0.730, 0.295, 0.295], color=blue, lw=2.6)
    arrow(0.902, 0.512, 0.930, 0.512)
    rounded_box(0.930, 0.385, 0.060, 0.245, radius=0.018)
    final_box = FancyBboxPatch(
        (0.940, 0.495), 0.040, 0.060,
        boxstyle="round,pad=0.006,rounding_size=0.008",
        facecolor="white", edgecolor=edge, linewidth=1.8,
    )
    ax.add_patch(final_box)
    ax.text(0.960, 0.525, "TrajSNN", ha="center", va="center", fontsize=11, weight="bold")
    ax.text(0.960, 0.455, r"$\mathcal{F}_{K}$", ha="center", va="center", fontsize=12, weight="bold")
    ax.text(0.960, 0.352, "selected feature\npanel", ha="center", va="top", fontsize=10)

    ax.text(0.50, 0.055, "Graph construction, reweighting, and feature ranking are fitted inside participant-disjoint training folds.",
            ha="center", fontsize=13, color=navy, weight="bold")
    save(fig, "figure1_method_schematic")


def internal_performance():
    source = ROOT / "results/main_benchmark/benchmark_full50"
    perf = pd.read_csv(source / "participant_bootstrap_summary.csv")
    stab = pd.read_csv(source / "selection_stability.csv")
    perf = canonicalize_method_column(perf)
    stab = canonicalize_method_column(stab)
    methods = ["Lasso", "Stabl-RP", "LongGroupLasso", "LLSS", "SNN-FS", "TrajSNN"]
    perf = perf[perf.Method.isin(methods)]
    stab = stab[stab.Method.isin(methods)]
    perf = perf.sort_values("OOF_AUROC")
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.4))
    y = np.arange(len(perf))
    xerr = np.vstack([perf.OOF_AUROC - perf.ParticipantBootstrapCI95Lower,
                      perf.ParticipantBootstrapCI95Upper - perf.OOF_AUROC])
    axes[0].errorbar(perf.OOF_AUROC, y, xerr=xerr, fmt="none", ecolor="#888888", capsize=2)
    axes[0].scatter(perf.OOF_AUROC, y, c=[COLORS.get(m, "#56B4E9") for m in perf.Method], s=45)
    axes[0].set_yticks(y, display_methods(perf.Method))
    axes[0].axvline(.5, ls="--", color="black", lw=1)
    axes[0].set(xlabel="Participant-level OOF AUROC", title="A  Matched Stanford internal discrimination")
    stability = stab.sort_values("MeanJaccard")
    axes[1].barh(display_methods(stability.Method), stability.MeanJaccard,
                 color=[COLORS.get(m, "#56B4E9") for m in stability.Method])
    axes[1].set(xlabel="Mean pairwise Jaccard", title="B  Panel stability")
    axes[1].set_xlim(0, max(.34, stability.MeanJaccard.max() * 1.1))
    fig.tight_layout()
    save(fig, "figure2_internal_performance_stability")


def simulation():
    table = pd.read_csv(ROOT / "results/archive/simulation/simulation_summary.csv")
    conditions = ["clean", "heterogeneous", "irregular_missing", "outliers"]
    methods = ["Lasso", "Stabl-RP", "LongGroupLasso", "LLSS", "SNN-FS", "TrajSNN"]
    fig, axes = plt.subplots(1, 3, figsize=(17.2, 6.8), constrained_layout=True)
    for ax, metric, title in zip(axes, ["Precision", "FDR", "TestAUROC"], ["A  Precision", "B  False discovery rate", "C  Test AUROC"]):
        matrix = table.pivot(index="Method", columns="Condition", values=metric).reindex(index=methods, columns=conditions)
        image = ax.imshow(matrix, cmap="viridis" if metric != "FDR" else "viridis_r", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(conditions)), ["Clean", "Hetero.", "Missing", "Outliers"], rotation=35, ha="right", fontsize=15)
        ax.set_yticks(range(len(methods)), display_methods(methods) if ax is axes[0] else [], fontsize=15)
        ax.set_title(title, fontsize=18, pad=12)
        ax.tick_params(axis="both", length=0, pad=5)
        for i in range(len(methods)):
            for j in range(len(conditions)):
                value = matrix.iloc[i, j]
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=17, fontweight="bold",
                        color="white" if value < .35 or value > .75 else "black")
    cbar = fig.colorbar(image, ax=axes, shrink=.76, label="Metric value")
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label("Metric value", fontsize=15)
    save(fig, "figure3_simulation")


def external_validation():
    table = pd.read_csv(ROOT / "results/validation/external_validation/transfer_performance.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.4), sharex=True)
    for ax, (train, test), title in zip(
        axes,
        [("Stanford", "Detroit"), ("Detroit", "Stanford")],
        ["A  Stanford → Detroit (primary)", "B  Detroit → Stanford (reverse)"],
    ):
        subset = table[(table.TrainCohort == train) & (table.TestCohort == test)].copy()
        subset = subset[
            (subset.ModelVariant == "ProteinOnly") &
            (subset.Method != "ClinicalBaseline")
        ]
        subset["Label"] = subset.Method.map(lambda value: DISPLAY_METHOD.get(value, value))
        subset = subset[subset.Label.isin(["Lasso", "Stabl-RP", "LongGroupLasso", "LLSS", "SNN-FS", "TrajSNN"])]
        subset = subset.sort_values("AUROC")
        y = np.arange(len(subset))
        xerr = np.vstack([
            subset.AUROC - subset.AUROC_CI95Lower,
            subset.AUROC_CI95Upper - subset.AUROC,
        ])
        ax.errorbar(subset.AUROC, y, xerr=xerr, fmt="none", ecolor="#777777", capsize=3)
        ax.scatter(subset.AUROC, y, c=[COLORS.get(m, "#56B4E9") for m in subset.Label], s=55)
        ax.set_yticks(y, subset.Label)
        ax.axvline(.5, ls="--", color="black", lw=1)
        ax.set(xlim=(.25, .92), xlabel="Locked participant-level AUROC (95% CI)", title=title)
    fig.tight_layout()
    save(fig, "figure4_external_validation")


def secondary_longitudinal():
    table = pd.read_csv(ROOT / "results/main_benchmark/benchmark_with_new/participant_bootstrap_summary.csv")
    table = canonicalize_method_column(table)
    pivot = table.pivot(index="Method", columns="Dataset", values="MeanRepeatRMSE")
    datasets = ["FPG", "IGG", "IGM", "TRA", "TRB"]
    methods = ["Stabl-RP", "LongGroupLasso", "LLSS", "SNN-FS", "TrajSNN"]
    relative = 100 * (pivot.loc[methods, datasets] / pivot.loc["Lasso", datasets] - 1)
    limit = max(5, float(np.nanmax(np.abs(relative.to_numpy()))))
    fig, ax = plt.subplots(figsize=(8.7, 4.6))
    image = ax.imshow(relative, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    ax.set_xticks(range(len(datasets)), datasets, fontsize=13)
    ax.set_yticks(range(len(methods)), display_methods(methods), fontsize=12.5)
    ax.set_title("Five additional real longitudinal outcomes", fontsize=14, pad=10)
    for i in range(len(methods)):
        for j in range(len(datasets)):
            value = relative.iloc[i, j]
            ax.text(j, i, f"{value:+.1f}%", ha="center", va="center",
                    fontsize=16, fontweight="bold",
                    color="white" if abs(value) > .55 * limit else "black")
    colorbar = fig.colorbar(image, ax=ax, shrink=.8)
    colorbar.set_label("RMSE change versus Lasso (%)\nnegative is better", fontsize=12)
    colorbar.ax.tick_params(labelsize=11)
    fig.tight_layout()
    save(fig, "figure5_secondary_longitudinal")


def biological_context():
    table = pd.read_csv(ROOT / "results/biological_validation/time_adjusted_feature_associations.csv").head(15)
    table = table.sort_values("TimeAdjustedMedianDifference")
    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    color = np.where(table.BHAdjustedP < .05, "#0072B2", "#BBBBBB")
    ax.barh(table.Feature, table.TimeAdjustedMedianDifference, color=color)
    ax.axvline(0, color="black", lw=.8)
    ax.set(xlabel="Time-adjusted PE minus control median difference",
           title="Exploratory associations among repeatedly selected proteins")
    fig.tight_layout()
    save(fig, "figure6_biological_context")


def main():
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    method_schematic()
    internal_performance()
    simulation()
    external_validation()
    secondary_longitudinal()
    biological_context()
    print(f"Wrote publication figures to {OUT}")


if __name__ == "__main__":
    main()
