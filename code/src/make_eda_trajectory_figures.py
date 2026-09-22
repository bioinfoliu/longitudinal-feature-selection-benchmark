"""Create EDA trajectory figures for the CTSNN progress presentation."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "results" / "figures_v2"
FIG_DIR.mkdir(parents=True, exist_ok=True)

NAVY = "#17324D"
BLUE = "#0072B2"
LIGHT_BLUE = "#7EA6FF"
GRAY = "#B8C0C8"
ORANGE = "#D97904"
RED = "#E64B35"
LIGHT_GRAY = "#E8EDF2"
DARK_TEAL = "#0E6A86"


def mean_ci(values: pd.Series) -> tuple[float, float, float, int]:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    n = int(clean.shape[0])
    mean = float(clean.mean()) if n else np.nan
    if n <= 1:
        return mean, mean, mean, n
    se = float(clean.std(ddof=1) / np.sqrt(n))
    return mean, mean - 1.96 * se, mean + 1.96 * se, n


def style_axis(ax, xlabel: str, ylabel: str, ylim: tuple[float, float] | None = None) -> None:
    ax.set_xlabel(xlabel, fontsize=10, color=NAVY)
    ax.set_ylabel(ylabel, fontsize=10, color=NAVY)
    ax.tick_params(axis="both", labelsize=9, colors=NAVY)
    ax.grid(True, color="#D6DCE2", linewidth=0.8)
    for spine in ax.spines.values():
        spine.set_color(NAVY)
        spine.set_linewidth(0.9)
    if ylim is not None:
        ax.set_ylim(*ylim)


def measurement_frequency(frame: pd.DataFrame) -> pd.DataFrame:
    counts = frame.groupby("ID").size().value_counts().sort_index()
    return pd.DataFrame({
        "# measures per\nsample": counts.index.astype(int),
        "# sample": counts.values.astype(int),
    })


def add_frequency_table(ax, frame: pd.DataFrame) -> None:
    table_data = measurement_frequency(frame)
    ax.axis("off")
    ax.set_title(
        "Distribution of Measurement\nFrequency Across Samples",
        fontsize=12,
        fontweight="bold",
        color=NAVY,
        pad=10,
    )
    table = ax.table(
        cellText=table_data.values,
        colLabels=table_data.columns,
        loc="center",
        cellLoc="center",
        colLoc="center",
        colWidths=[0.68, 0.50],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10.5)
    table.scale(1.28, 1.72)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("white")
        cell.set_linewidth(0.7)
        if row == 0:
            cell.set_facecolor(DARK_TEAL)
            cell.get_text().set_color("white")
            cell.get_text().set_fontweight("bold")
        else:
            cell.set_facecolor("#CED6DE" if row % 2 else LIGHT_GRAY)
            cell.get_text().set_color("black")


def plot_subject_lines(ax, frame: pd.DataFrame, x_col: str, y_col: str, y_jitter: float = 0.0) -> None:
    visit_counts = frame.groupby("ID").size()
    single_ids = set(visit_counts.loc[visit_counts == 1].index)
    multi_frame = frame.loc[~frame["ID"].isin(single_ids)].copy()
    single_frame = frame.loc[frame["ID"].isin(single_ids)].copy()
    colors = plt.get_cmap("tab20").colors
    rng = np.random.default_rng(20260909)
    for idx, (_, sub) in enumerate(multi_frame.sort_values(x_col).groupby("ID")):
        if sub.shape[0] < 2:
            continue
        y_values = sub[y_col].to_numpy(float)
        if y_jitter > 0:
            y_values = y_values + rng.normal(0, y_jitter, size=y_values.shape)
        ax.plot(sub[x_col], y_values, color=colors[idx % len(colors)], linewidth=1.0, alpha=0.34, zorder=1)
    multi_y = multi_frame[y_col].to_numpy(float)
    single_y = single_frame[y_col].to_numpy(float)
    if y_jitter > 0:
        multi_y = multi_y + rng.normal(0, y_jitter, size=multi_y.shape)
        single_y = single_y + rng.normal(0, y_jitter, size=single_y.shape)
    ax.scatter(multi_frame[x_col], multi_y, s=11, color=ORANGE, alpha=0.35, edgecolor="none", zorder=2)
    ax.scatter(single_frame[x_col], single_y, s=18, color=RED, alpha=0.78, edgecolor="white", linewidth=0.25, zorder=5)


def add_mean_ci(ax, summary: pd.DataFrame) -> None:
    ax.fill_between(
        summary["x"].to_numpy(float),
        summary["low"].to_numpy(float),
        summary["high"].to_numpy(float),
        color=LIGHT_BLUE,
        alpha=0.28,
        label="95% CI",
        zorder=3,
    )
    ax.plot(
        summary["x"],
        summary["mean"],
        color=BLUE,
        marker="o",
        markersize=5,
        linewidth=2.1,
        label="Mean",
        zorder=4,
    )


def save_figure(fig: plt.Figure, name: str) -> None:
    for suffix in (".png", ".pdf"):
        fig.savefig(FIG_DIR / f"{name}{suffix}", dpi=320, bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{name}.jpg", dpi=260, bbox_inches="tight")
    plt.close(fig)


def make_pe_plot(cohort: str) -> None:
    frame = pd.read_csv(ROOT / "data" / "external" / "processed" / f"{cohort}_SomaLogic_longitudinal_PE.csv")
    frame = frame.loc[frame["GestationalAge"] <= 22].copy()
    if cohort == "Stanford":
        frame = frame.loc[frame["Phenotype"] != "PE-early"].copy()
    frame["Outcome"] = frame["LatePE"].astype(float)
    bins = np.array([0, 10, 14, 18, 22.01])
    labels = ["<=10", "10-14", "14-18", "18-22"]
    centers = np.array([9, 12, 16, 20], dtype=float)
    frame["TimeBin"] = pd.cut(frame["GestationalAge"], bins=bins, labels=labels, include_lowest=True, right=True)
    center_map = dict(zip(labels, centers))
    frame["TimeCenter"] = frame["TimeBin"].astype(str).map(center_map).astype(float)

    rows = []
    for label, x in center_map.items():
        vals = frame.loc[frame["TimeBin"].astype(str) == label, "Outcome"]
        mean, low, high, n = mean_ci(vals)
        rows.append({"x": x, "mean": mean, "low": max(0.0, low), "high": min(1.0, high), "n": n})
    summary = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(8.4, 4.9))
    plot_subject_lines(ax, frame.dropna(subset=["TimeCenter"]), "TimeCenter", "Outcome", y_jitter=0.018)
    add_mean_ci(ax, summary.dropna())
    style_axis(ax, "Gestational age bin (weeks)", "Mean PE label", ylim=(-0.08, 1.08))
    ax.set_xticks(centers, labels)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["0", "1"])
    ax.legend(loc="upper right", frameon=True, fontsize=9)
    ax.text(
        0.01,
        -0.18,
        f"Participants: {frame['ID'].nunique()}  Visits: {frame.shape[0]}  Outcome: late-onset PE label",
        transform=ax.transAxes,
        fontsize=9,
        color=NAVY,
    )
    save_figure(fig, f"eda_{cohort.lower()}_pe_trajectory")


def make_fpg_plot() -> None:
    frame = pd.read_csv(ROOT / "data" / "processed" / "FPG_final.csv")
    order = ["0M", "1M", "3M", "12M"]
    x_map = {time: idx + 1 for idx, time in enumerate(order)}
    frame = frame.loc[frame["time"].isin(order)].copy()
    frame["TimeIndex"] = frame["time"].map(x_map).astype(float)
    frame["Outcome"] = pd.to_numeric(frame["Fasting_plasma_glucose"], errors="coerce")

    rows = []
    for time, x in x_map.items():
        mean, low, high, n = mean_ci(frame.loc[frame["time"] == time, "Outcome"])
        rows.append({"x": x, "mean": mean, "low": low, "high": high, "n": n})
    summary = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(8.4, 4.9))
    plot_subject_lines(ax, frame, "TimeIndex", "Outcome")
    add_mean_ci(ax, summary)
    style_axis(ax, "Visit time", "Fasting plasma glucose")
    ax.set_xticks(list(x_map.values()), order)
    ax.legend(loc="upper right", frameon=True, fontsize=9)
    ax.text(
        0.01,
        -0.18,
        f"Participants: {frame['ID'].nunique()}  Visits: {frame.shape[0]}",
        transform=ax.transAxes,
        fontsize=9,
        color=NAVY,
    )
    save_figure(fig, "eda_japan_t2d_fpg_trajectory")


def make_covid_plot() -> None:
    tasks = [("IGG", "BCR-IGG"), ("IGM", "BCR-IGM"), ("TRA", "TCR-alpha"), ("TRB", "TCR-beta")]
    fig, axes = plt.subplots(2, 2, figsize=(9.4, 6.1), sharex=True)
    panel_labels = ["(A)", "(B)", "(C)", "(D)"]
    for ax, panel_label, (task, label) in zip(axes.ravel(), panel_labels, tasks):
        frame = pd.read_csv(ROOT / "data" / "processed" / f"{task}_final.csv")
        frame["Outcome"] = pd.to_numeric(frame["shannon_diversity"], errors="coerce")
        frame["TimeIndex"] = pd.to_numeric(frame["time"], errors="coerce")
        rows = []
        for time in sorted(frame["TimeIndex"].dropna().unique()):
            mean, low, high, n = mean_ci(frame.loc[frame["TimeIndex"] == time, "Outcome"])
            rows.append({"x": time, "mean": mean, "low": low, "high": high, "n": n})
        summary = pd.DataFrame(rows)
        plot_subject_lines(ax, frame, "TimeIndex", "Outcome")
        add_mean_ci(ax, summary)
        style_axis(ax, "Time", "Shannon diversity")
        ax.set_xticks([1, 2, 3, 4])
        ax.set_title(label, fontsize=12, fontweight="bold", color=NAVY, pad=8)
        ax.text(
            -0.18,
            1.08,
            panel_label,
            transform=ax.transAxes,
            fontsize=14,
            fontweight="bold",
            color="black",
            ha="left",
            va="bottom",
        )
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=True, fontsize=9)
    fig.subplots_adjust(bottom=0.12, top=0.96, wspace=0.22, hspace=0.32)
    save_figure(fig, "eda_korea_covid_trajectory")


def main() -> None:
    make_pe_plot("Stanford")
    make_pe_plot("Detroit")
    make_fpg_plot()
    make_covid_plot()


if __name__ == "__main__":
    main()
