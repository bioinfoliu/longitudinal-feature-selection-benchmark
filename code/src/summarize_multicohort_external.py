"""Paired participant-bootstrap comparisons for multicohort transfer."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from summarize_expanded_results import fast_auc


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "validation" / "external_validation"

COMPARISONS = (
    ("Stanford", "Detroit", "SNN-FS", "ProteinOnly", "Lasso", "ProteinOnly"),
    ("Stanford", "Detroit", "SNN-FS", "ProteinPlusClinical", "ClinicalBaseline", "ClinicalOnly"),
    ("Detroit", "Stanford", "w/o Robust", "ProteinOnly", "Lasso", "ProteinOnly"),
    ("Detroit", "Stanford", "w/o Robust", "ProteinPlusClinical", "ClinicalBaseline", "ClinicalOnly"),
)


def main(iterations: int = 5000):
    predictions = pd.read_csv(OUT / "subject_predictions.csv")
    rows = []
    rng = np.random.default_rng(20260908)
    for train, test, method, variant, reference, reference_variant in COMPARISONS:
        subset = predictions[(predictions.TrainCohort == train) & (predictions.TestCohort == test)]
        left = subset[(subset.Method == method) & (subset.ModelVariant == variant)].set_index("ID")
        right = subset[(subset.Method == reference) & (subset.ModelVariant == reference_variant)].set_index("ID")
        common = left.index.intersection(right.index)
        y = left.loc[common, "Outcome"].to_numpy(int)
        a = left.loc[common, "Prediction"].to_numpy(float)
        b = right.loc[common, "Prediction"].to_numpy(float)
        point = roc_auc_score(y, a) - roc_auc_score(y, b)
        differences = []
        for _ in range(iterations):
            sample = rng.integers(0, len(common), len(common))
            if np.unique(y[sample]).size < 2:
                continue
            differences.append(fast_auc(y[sample], a[sample]) - fast_auc(y[sample], b[sample]))
        differences = np.asarray(differences)
        pvalue = min(1.0, 2 * min(np.mean(differences <= 0), np.mean(differences >= 0)))
        rows.append({
            "TrainCohort": train, "TestCohort": test, "Method": method,
            "ModelVariant": variant, "Reference": reference,
            "ReferenceVariant": reference_variant, "AUROCDifference": point,
            "DifferenceCI95Lower": np.quantile(differences, .025),
            "DifferenceCI95Upper": np.quantile(differences, .975),
            "PairedBootstrapP": pvalue,
        })
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "paired_transfer_comparisons.csv", index=False)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
