import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from longfsbench import rank_features


class SmokeTest(unittest.TestCase):
    def test_representative_rankers_return_finite_unique_features(self):
        root = Path(__file__).resolve().parents[1]
        frame = pd.read_csv(root / "data/example/minimal_longitudinal.csv")
        feature_names = [column for column in frame if column.startswith("feature_")]
        X = frame[feature_names].to_numpy(float)
        y = frame.label.to_numpy(int)
        groups = frame.ID.to_numpy(str)
        times = frame.time.to_numpy(float)
        for method in ("Lasso", "LongGroupLasso", "SNN-FS", "OutcomeSNN-FS"):
            with self.subTest(method=method):
                result = rank_features(method, X, y, groups, times, "classification", seed=7)
                top = result.ranking[:5]
                self.assertEqual(len(set(top)), 5)
                self.assertTrue(np.isfinite(result.scores[top]).all())


if __name__ == "__main__":
    unittest.main()

