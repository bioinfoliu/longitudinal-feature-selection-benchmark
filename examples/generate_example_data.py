"""Generate the small, synthetic, non-identifiable example dataset."""

from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    rng = np.random.default_rng(20260922)
    rows = []
    for subject in range(1, 25):
        label = subject % 2
        subject_shift = rng.normal(0, 0.35)
        for time in (0.0, 1.0, 2.0):
            noise = rng.normal(0, 1, 8)
            features = noise
            features[0] += 1.2 * label + 0.30 * time + subject_shift
            features[1] += 0.8 * label - 0.25 * time
            features[2] += 0.5 * label * time
            rows.append({
                "ID": f"S{subject:02d}", "time": time, "label": label,
                **{f"feature_{index + 1}": value for index, value in enumerate(features)},
            })
    output = Path(__file__).resolve().parents[1] / "data/example/minimal_longitudinal.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()

