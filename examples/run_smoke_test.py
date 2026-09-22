"""Run four representative selectors on the bundled example data."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from longfsbench import rank_features


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    frame = pd.read_csv(root / "data/example/minimal_longitudinal.csv")
    feature_names = [column for column in frame if column.startswith("feature_")]
    X = frame[feature_names].to_numpy(float)
    y = frame.label.to_numpy(int)
    groups = frame.ID.to_numpy(str)
    times = frame.time.to_numpy(float)
    outputs = {}
    for method in ("Lasso", "LongGroupLasso", "SNN-FS", "OutcomeSNN-FS"):
        result = rank_features(method, X, y, groups, times, "classification", seed=7)
        top = result.ranking[:5]
        assert len(np.unique(top)) == 5
        assert np.isfinite(result.scores[top]).all()
        outputs[method] = [feature_names[index] for index in top]
    print(json.dumps(outputs, indent=2))
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()

