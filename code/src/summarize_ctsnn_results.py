"""Create manuscript-ready summaries for the canonical CTSNN benchmark."""

from pathlib import Path

import pandas as pd
from scipy.stats import ttest_rel


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "ctsnn_benchmark"


def summarize(files, output):
    folds = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
    rows = []
    for (dataset, task), block in folds.groupby(["Dataset", "Task"]):
        metric = "AUROC" if task == "classification" else "RMSE"
        wide = block.pivot(index="Fold", columns="Method", values=metric)
        baseline = wide["Equal"]
        for method in wide:
            gain = wide[method] - baseline if metric == "AUROC" else baseline - wide[method]
            test = ttest_rel(wide[method], baseline) if method != "Equal" else None
            rows.append({
                "Dataset": dataset,
                "Task": task,
                "Metric": metric,
                "Method": method,
                "Mean": wide[method].mean(),
                "SD": wide[method].std(ddof=1),
                "AbsoluteImprovementVsEqual": gain.mean(),
                "RelativeImprovementPct": 100 * gain.mean() / baseline.mean(),
                "PairedTTestPValue": test.pvalue if test else float("nan"),
            })
    summary = pd.DataFrame(rows).sort_values(["Dataset", "Mean"])
    summary.to_csv(output, index=False)
    return summary


def main():
    primary_files = sorted(RESULTS.glob("fold_metrics_*_k3.csv"))
    primary = summarize(primary_files, RESULTS / "primary_summary_k3.csv")
    sensitivity_files = [RESULTS / f"fold_metrics_PE_k{k}.csv" for k in (3, 5, 10)]
    sensitivity_parts = []
    for k, path in zip((3, 5, 10), sensitivity_files):
        part = pd.read_csv(path).groupby("Method", as_index=False)[["AUROC", "Accuracy"]].mean()
        part.insert(0, "n_features", k)
        sensitivity_parts.append(part)
    sensitivity = pd.concat(sensitivity_parts, ignore_index=True)
    sensitivity.to_csv(RESULTS / "pe_feature_budget_sensitivity.csv", index=False)
    print(primary.round(4).to_string(index=False))
    print("\nPE feature-budget sensitivity\n")
    print(sensitivity.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
