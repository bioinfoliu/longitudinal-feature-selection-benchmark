"""Run leakage-safe, subject-level benchmark experiments for CTSNN.

Usage:
  /Library/Frameworks/Python.framework/Versions/3.11/bin/python3 \
      src/run_longitudinal_benchmark.py
"""

from __future__ import annotations

import json
import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from longitudinal_lasso import LongitudinalLassoSelector


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
OUT = ROOT / "results" / "longitudinal_lasso"
OUT.mkdir(parents=True, exist_ok=True)

DATASETS = {
    "IGG": ("IGG_final.csv", "shannon_diversity", "regression"),
    "IGM": ("IGM_final.csv", "shannon_diversity", "regression"),
    "TRA": ("TRA_final.csv", "shannon_diversity", "regression"),
    "TRB": ("TRB_final.csv", "shannon_diversity", "regression"),
    "FPG": ("FPG_final.csv", "Fasting_plasma_glucose", "regression"),
    "PE": ("PE_final.csv", "label", "classification"),
}


def read_csv(path: Path) -> pd.DataFrame:
    for encoding in ("utf-8", "cp949", "latin1"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"Unable to read {path}")


def log_transform(X: np.ndarray) -> np.ndarray:
    # Biomarker matrices are non-negative.  Log1p reduces the large dynamic
    # range while preserving zero values; signed values are kept meaningful.
    return np.sign(X) * np.log1p(np.abs(X))


def prepare_features(df: pd.DataFrame, target: str):
    feature_cols = [c for c in df.columns if c not in {"ID", "time", target}]
    X = df[feature_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    X = SimpleImputer(strategy="median").fit_transform(X)
    return X, feature_cols


def add_time(X: np.ndarray, time_train: np.ndarray, time_test: np.ndarray, X_test: np.ndarray):
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    t_train = encoder.fit_transform(time_train.reshape(-1, 1))
    t_test = encoder.transform(time_test.reshape(-1, 1))
    return np.hstack((X, t_train)), np.hstack((X_test, t_test))


def fit_and_predict(task, selector_name, X_train, y_train, groups_train, time_train, X_test, time_test, n_features, seed):
    if selector_name == "Longitudinal-LASSO":
        selector = LongitudinalLassoSelector(
            task=task, n_features=n_features, n_resamples=5 if task == "classification" else 10,
            prefilter_features=300, random_state=seed,
        ).fit(X_train, y_train, groups_train)
        selected = selector.selected_indices_
        frequencies = selector.selection_frequency_
        prefilter = selector.prefilter_idx_
    else:
        score_fn = f_regression if task == "regression" else f_classif
        selector = SelectKBest(score_func=score_fn, k=min(n_features, X_train.shape[1])).fit(X_train, y_train)
        selected = selector.get_support(indices=True)
        frequencies = np.full(len(selected), np.nan)
        prefilter = selected

    scale = Pipeline([
        ("log", FunctionTransformer(log_transform)),
        ("scale", StandardScaler()),
    ])
    train_selected = scale.fit_transform(X_train[:, selected])
    test_selected = scale.transform(X_test[:, selected])
    train_model, test_model = add_time(train_selected, np.asarray(time_train), np.asarray(time_test), test_selected)

    if task == "regression":
        # Selection is the object of comparison, so both selectors feed the
        # same modestly regularised, interpretable prediction model.
        estimator = Ridge(alpha=10.0)
        estimator.fit(train_model, y_train)
        pred = estimator.predict(test_model)
        return pred, selected, frequencies, prefilter
    else:
        estimator = LogisticRegression(
            C=0.3, penalty="l1", solver="liblinear", class_weight="balanced",
            max_iter=2_000, random_state=seed,
        )
        estimator.fit(train_model, y_train)
        return estimator.predict_proba(test_model)[:, 1], selected, frequencies, prefilter


def evaluate_dataset(name: str, filename: str, target: str, task: str, n_features: int) -> tuple[list[dict], list[dict]]:
    df = read_csv(DATA / filename)
    X, feature_names = prepare_features(df, target)
    y = df[target].to_numpy()
    if task == "classification":
        y = (y == "PE").astype(int) if y.dtype.kind in {"O", "U", "S"} else y.astype(int)
    else:
        y = y.astype(float)
    groups = df["ID"].to_numpy()
    times = df["time"].astype(str).to_numpy()
    n_splits = min(5, len(np.unique(groups)))
    outer = GroupKFold(n_splits=n_splits)
    rows, features = [], []
    methods = ["Univariate", "Longitudinal-LASSO"]
    for fold, (train, test) in enumerate(outer.split(X, y, groups), 1):
        for selector in methods:
            label = selector + (" + Ridge" if task == "regression" else " + Logistic")
            pred, selected, freq, prefilter = fit_and_predict(
                task, selector, X[train], y[train], groups[train], times[train], X[test], times[test], n_features, 2000 + fold
            )
            row = {"Dataset": name, "Task": task, "Fold": fold, "Method": label, "n_features": len(selected)}
            if task == "regression":
                row.update({"RMSE": mean_squared_error(y[test], pred, squared=False), "MAE": mean_absolute_error(y[test], pred)})
            else:
                row.update({"AUROC": roc_auc_score(y[test], pred), "Accuracy": accuracy_score(y[test], pred >= 0.5)})
            rows.append(row)
            for idx in selected:
                frequency = float(freq[np.where(prefilter == idx)[0][0]]) if selector == "Longitudinal-LASSO" else np.nan
                features.append({"Dataset": name, "Fold": fold, "Method": label, "Feature": feature_names[idx], "StabilityFrequency": frequency})
    return rows, features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", choices=sorted(DATASETS), default=sorted(DATASETS))
    parser.add_argument("--n-features", type=int, default=20)
    args = parser.parse_args()
    all_rows, all_features = [], []
    for name in args.datasets:
        filename, target, task = DATASETS[name]
        print(f"Running {name} ({task})", flush=True)
        rows, features = evaluate_dataset(name, filename, target, task, args.n_features)
        all_rows.extend(rows)
        all_features.extend(features)
        pd.DataFrame(rows).to_csv(OUT / f"fold_metrics_{name}_k{args.n_features}.csv", index=False)
        pd.DataFrame(features).to_csv(OUT / f"selected_features_{name}_k{args.n_features}.csv", index=False)
    results = pd.DataFrame(all_rows)
    selected = pd.DataFrame(all_features)
    run_label = "_".join(args.datasets)
    results.to_csv(OUT / f"fold_metrics_{run_label}_k{args.n_features}.csv", index=False)
    selected.to_csv(OUT / f"selected_features_{run_label}_k{args.n_features}.csv", index=False)
    numeric = ["RMSE", "MAE"] if "RMSE" in results else []
    summary = results.groupby(["Dataset", "Task", "Method"], as_index=False).mean(numeric_only=True)
    summary.to_csv(OUT / f"summary_metrics_{run_label}_k{args.n_features}.csv", index=False)
    with open(OUT / f"run_metadata_{run_label}_k{args.n_features}.json", "w", encoding="utf-8") as f:
        json.dump({"outer_cv": "GroupKFold by participant ID", "n_folds": 5, "n_resamples": 10, "n_features": args.n_features}, f, indent=2)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    sys.exit(main())
