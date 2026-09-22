"""Export a retrospective Slurm resource audit for completed official adapters."""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd


JOBS = {
    "409863": ("geeVerse", 8),
    "410117": ("glmmLasso", 1),
    "410132": ("PGEE", 1),
}


def memory_mb(value: str) -> float:
    if not value:
        return np.nan
    match = re.fullmatch(r"([0-9.]+)([KMGTP]?)", value.strip())
    if not match:
        return np.nan
    number, unit = float(match.group(1)), match.group(2)
    factors = {"": 1.0, "K": 1 / 1024, "M": 1.0, "G": 1024.0, "T": 1024**2, "P": 1024**3}
    return number * factors[unit]


def duration_seconds(value: str) -> float:
    if not value:
        return np.nan
    days = 0
    if "-" in value:
        day, value = value.split("-", 1)
        days = int(day)
    parts = [float(part) for part in value.split(":")]
    if len(parts) == 3:
        hours, minutes, seconds = parts
    elif len(parts) == 2:
        hours, minutes, seconds = 0, *parts
    else:
        return parts[0]
    return days * 86400 + hours * 3600 + minutes * 60 + seconds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        "sacct", "-j", ",".join(JOBS),
        "--format=JobID,JobName,State,ElapsedRaw,TotalCPU,MaxRSS,AllocCPUS,ReqMem",
        "-P", "--units=M",
    ]
    lines = subprocess.run(command, check=True, capture_output=True, text=True).stdout.splitlines()
    raw = pd.read_csv(pd.io.common.StringIO("\n".join(lines)), sep="|")
    raw = raw.loc[:, ~raw.columns.str.match("^Unnamed")]
    task_rows = raw[~raw.JobID.str.contains("\\.", regex=True)].copy()
    batch = raw[raw.JobID.str.endswith(".batch")].copy()
    batch["TaskJobID"] = batch.JobID.str.removesuffix(".batch")
    batch = batch.set_index("TaskJobID")
    records = []
    for row in task_rows.itertuples(index=False):
        array_id = str(row.JobID).split("_")[0]
        if array_id not in JOBS or not str(row.State).startswith("COMPLETED"):
            continue
        method, work_units = JOBS[array_id]
        child = batch.loc[row.JobID] if row.JobID in batch.index else None
        records.append({
            "ArrayJobID": array_id, "ArrayTask": row.JobID, "Method": method,
            "CompletedDatasetRepeats": work_units,
            "WallSeconds": float(row.ElapsedRaw),
            "CPUSeconds": duration_seconds(child.TotalCPU) if child is not None else np.nan,
            "PeakRSS_MB": memory_mb(child.MaxRSS) if child is not None else np.nan,
            "AllocatedCPUs": int(row.AllocCPUS), "RequestedMemory": row.ReqMem,
        })
    detail = pd.DataFrame(records)
    summary = detail.groupby("Method").agg(
        CompletedSlurmTasks=("ArrayTask", "size"),
        CompletedDatasetRepeats=("CompletedDatasetRepeats", "sum"),
        TotalWallHours=("WallSeconds", lambda x: x.sum() / 3600),
        MedianTaskWallSeconds=("WallSeconds", "median"),
        IQRTaskWallSeconds=("WallSeconds", lambda x: x.quantile(.75) - x.quantile(.25)),
        TotalCPUHours=("CPUSeconds", lambda x: x.sum() / 3600),
        MedianPeakRSS_MB=("PeakRSS_MB", "median"),
        MaximumPeakRSS_MB=("PeakRSS_MB", "max"),
        AllocatedCPUs=("AllocatedCPUs", "median"),
    ).reset_index()
    summary["Interpretation"] = "Retrospective Slurm audit; job granularity differs by adapter, so not a controlled speed comparison"
    detail.to_csv(args.output_dir / "slurm_resource_audit_detail.csv", index=False)
    summary.to_csv(args.output_dir / "slurm_resource_audit_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

