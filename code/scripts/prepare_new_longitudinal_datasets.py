"""Convert newly downloaded public longitudinal datasets to benchmark tables.

The benchmark consumes one row per visit/sample with columns ``ID``, ``time``,
one outcome column, and numeric feature columns.  This script deliberately
keeps provenance columns in a separate audit file and excludes clinical
variables that would leak the outcome into the feature matrix.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import glob


def prepare_olink(root: Path, out: Path) -> None:
    data = root / "data_sources" / "olink" / "data"
    expression = pd.read_csv(data / "plasma_npx_level.csv")
    sample = pd.read_csv(data / "plasma_sample_level.csv")
    # Fatal disease is a subject-level endpoint repeated over the available
    # visits; the first sample remains a valid longitudinal observation.
    keep = sample["Fatal_Disease"].notna() & sample["Time_From_First_Symptoms"].notna()
    sample = sample.loc[keep].copy()
    sample["label"] = sample["Fatal_Disease"].astype(bool).astype(int)
    sample["time"] = sample["Time_From_First_Symptoms"].astype(float)
    wide = expression.pivot_table(index="SampleID", columns="Assay", values="NPX", aggfunc="mean")
    wide = wide.reset_index()
    frame = sample[["SampleID", "Individual_ID", "time", "label"]].merge(wide, on="SampleID", how="inner")
    frame = frame.rename(columns={"Individual_ID": "ID"})
    frame = frame.dropna(axis=1, how="all")
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / "OLINK_COVID_LONGITUDINAL.csv", index=False)
    sample.to_csv(out / "OLINK_COVID_sample_audit.csv", index=False)
    print("OLINK_COVID_LONGITUDINAL", frame.shape, "subjects", frame.ID.nunique(), "positive", frame.label.sum())


def prepare_brist1d(root: Path, out: Path) -> None:
    """Aggregate the open smartwatch/diabetes streams to one row per person-day.

    Blood glucose is the continuous outcome. It is deliberately excluded from
    the predictors; the remaining device channels are summarized within day.
    """
    base = root / "data_sources" / "brist1d" / "33z5jc8fa6tob21ptrugzqog08" / "device_data" / "processed_state"
    rows = []
    for path in sorted(glob.glob(str(base / "P*.csv"))):
        participant = Path(path).stem
        d = pd.read_csv(path, low_memory=False)
        d["timestamp"] = pd.to_datetime(d["timestamp"], errors="coerce", utc=True)
        d = d.dropna(subset=["timestamp"])
        d["day"] = d["timestamp"].dt.floor("D")
        numeric = [c for c in ["insulin", "carbs", "hr", "dist", "steps", "cals"] if c in d]
        if not numeric:
            continue
        target = d.groupby("day")["bg"].agg(label="mean", n_bg="count").reset_index()
        features = d.groupby("day").agg(
            **{f"{c}_mean": (c, "mean") for c in numeric},
            **{f"{c}_sd": (c, "std") for c in numeric},
        ).reset_index()
        agg = target.merge(features, on="day", how="left")
        agg.insert(0, "ID", participant)
        agg["time"] = (agg["day"] - agg["day"].min()).dt.total_seconds() / 86400.0
        rows.append(agg.drop(columns="day"))
    frame = pd.concat(rows, ignore_index=True)
    frame = frame.loc[frame["n_bg"] >= 3].drop(columns="n_bg")
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / "BRIST1D_DAILY.csv", index=False)
    print("BRIST1D_DAILY", frame.shape, "subjects", frame.ID.nunique())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare_olink(args.server_root, args.output)
    prepare_brist1d(args.server_root, args.output)


if __name__ == "__main__":
    main()
