"""Aggregate completed longitudinal feature-selection benchmark runs."""

from pathlib import Path

import pandas as pd
from scipy.stats import ttest_rel


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "longitudinal_lasso"


def main():
    files = sorted(RESULTS.glob("fold_metrics_*_k3.csv"))
    if not files:
        raise FileNotFoundError("No k=3 fold-level result files found.")
    folds = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    summary_rows = []
    paired_rows = []
    for (dataset, task), block in folds.groupby(["Dataset", "Task"]):
        metric = "AUROC" if task == "classification" else "RMSE"
        wide = block.pivot(index="Fold", columns="Method", values=metric)
        base = [c for c in wide if c.startswith("Univariate")][0]
        proposed = [c for c in wide if c.startswith("Longitudinal")][0]
        difference = wide[proposed] - wide[base]
        # Positive is better for AUROC; lower is better for RMSE.
        improvement = difference if metric == "AUROC" else -difference
        ttest = ttest_rel(wide[proposed], wide[base])
        summary_rows.append({
            "Dataset": dataset,
            "Task": task,
            "Metric": metric,
            "BaselineMean": wide[base].mean(),
            "ProposedMean": wide[proposed].mean(),
            "AbsoluteImprovement": improvement.mean(),
            "RelativeImprovementPct": 100 * improvement.mean() / wide[base].mean(),
            "PairedTTestPValue": ttest.pvalue,
        })
        paired_rows.extend({"Dataset": dataset, "Metric": metric, "Fold": fold, "Improvement": value}
                           for fold, value in improvement.items())
    summary = pd.DataFrame(summary_rows).sort_values("Dataset")
    summary.to_csv(RESULTS / "performance_summary_k3.csv", index=False)
    pd.DataFrame(paired_rows).to_csv(RESULTS / "paired_fold_improvements_k3.csv", index=False)
    print(summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
