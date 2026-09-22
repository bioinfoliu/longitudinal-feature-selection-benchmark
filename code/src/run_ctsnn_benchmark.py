"""Benchmark the user's SNN/STAR/Hybrid methods and SOTA-enhanced CTSNN."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import f_classif, f_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ctsnn import ctsnn_weights
from longitudinal_lasso import LongitudinalLassoSelector


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
OUT = ROOT / "results" / "ctsnn_benchmark"
OUT.mkdir(parents=True, exist_ok=True)

DATASETS = {
    "IGG": ("IGG_final.csv", "shannon_diversity", "regression"),
    "IGM": ("IGM_final.csv", "shannon_diversity", "regression"),
    "TRA": ("TRA_final.csv", "shannon_diversity", "regression"),
    "TRB": ("TRB_final.csv", "shannon_diversity", "regression"),
    "FPG": ("FPG_final.csv", "Fasting_plasma_glucose", "regression"),
    "PE": ("PE_final.csv", "label", "classification"),
}
WEIGHT_METHODS = ("Equal", "SNN-FS", "w/o Traj", "w/o Robust", "CTSNN")


def load_dataset(name: str):
    filename, target, task = DATASETS[name]
    df = pd.read_csv(DATA / filename)
    columns = [c for c in df if c not in {"ID", "time", target}]
    X = df[columns].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    X = SimpleImputer(strategy="median").fit_transform(X)
    y = df[target].to_numpy()
    if task == "classification":
        y = (y == "PE").astype(int)
    else:
        y = y.astype(float)
    return df, X, y, columns, task


def supervised_prefilter(X, y, task, limit=300):
    if X.shape[1] <= limit:
        return np.arange(X.shape[1])
    scores, _ = (f_regression(X, y) if task == "regression" else f_classif(X, y))
    scores = np.nan_to_num(scores, nan=-np.inf)
    return np.argsort(scores)[-limit:]


def weighted_sparse_selector(X, y, task, weights, n_features):
    prefilter = supervised_prefilter(X, y, task)
    Xt = np.sign(X[:, prefilter]) * np.log1p(np.abs(X[:, prefilter]))
    Xt = StandardScaler().fit_transform(Xt)
    if task == "regression":
        model = ElasticNet(alpha=0.03, l1_ratio=0.8, max_iter=20_000, random_state=2026)
        model.fit(Xt, y, sample_weight=weights)
        strength = np.abs(model.coef_)
    else:
        model = LogisticRegression(
            C=0.3, penalty="l1", solver="liblinear", class_weight="balanced",
            max_iter=2_000, random_state=2026,
        )
        model.fit(Xt, y, sample_weight=weights)
        strength = np.abs(model.coef_).ravel()
    local = np.argsort(strength)[-min(n_features, len(strength)):]
    return np.sort(prefilter[local])


def predict_from_selected(X_train, X_test, y_train, task, train_time, test_time, selected):
    scaler = StandardScaler()
    A = np.sign(X_train[:, selected]) * np.log1p(np.abs(X_train[:, selected]))
    B = np.sign(X_test[:, selected]) * np.log1p(np.abs(X_test[:, selected]))
    A = scaler.fit_transform(A)
    B = scaler.transform(B)
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    A = np.column_stack((A, encoder.fit_transform(train_time.reshape(-1, 1))))
    B = np.column_stack((B, encoder.transform(test_time.reshape(-1, 1))))
    if task == "regression":
        model = Ridge(alpha=10.0).fit(A, y_train)
        return model.predict(B)
    model = LogisticRegression(
        C=0.3, penalty="l1", solver="liblinear", class_weight="balanced",
        max_iter=2_000, random_state=2026,
    ).fit(A, y_train)
    return model.predict_proba(B)[:, 1]


def evaluate(name: str, n_features: int):
    df, X, y, feature_names, task = load_dataset(name)
    groups = df.ID.to_numpy()
    times = df.time.astype(str).to_numpy()
    if task == "classification":
        outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=2026)
    else:
        outer = GroupKFold(n_splits=5)
    rows, selected_rows, weight_rows = [], [], []
    for fold, (train, test) in enumerate(outer.split(X, y, groups), 1):
        weights_by_method = {
            method: ctsnn_weights(X[train], groups[train], method)
            for method in WEIGHT_METHODS
        }
        method_selections = {
            method: weighted_sparse_selector(X[train], y[train], task, weights, n_features)
            for method, weights in weights_by_method.items()
        }
        # Published Longitudinal-LASSO-inspired comparator.
        llss = LongitudinalLassoSelector(
            task=task, n_features=n_features,
            n_resamples=5 if task == "classification" else 10,
            random_state=2026 + fold,
        ).fit(X[train], y[train], groups[train])
        method_selections["LLSS"] = llss.selected_indices_
        # SOTA component now augments the user's robust SNN/STAR fusion.
        ctsnn_llss = LongitudinalLassoSelector(
            task=task, n_features=n_features,
            n_resamples=5 if task == "classification" else 10,
            random_state=3026 + fold,
        ).fit(X[train], y[train], groups[train], sample_weight=weights_by_method["CTSNN"])
        method_selections["CTSNN+LLSS"] = ctsnn_llss.selected_indices_

        for method, selected in method_selections.items():
            prediction = predict_from_selected(
                X[train], X[test], y[train], task, times[train], times[test], selected
            )
            row = {"Dataset": name, "Task": task, "Fold": fold, "Method": method,
                   "n_features": len(selected)}
            if task == "regression":
                row.update(RMSE=mean_squared_error(y[test], prediction, squared=False),
                           MAE=mean_absolute_error(y[test], prediction))
            else:
                row.update(AUROC=roc_auc_score(y[test], prediction),
                           Accuracy=accuracy_score(y[test], prediction >= 0.5))
            rows.append(row)
            selected_rows.extend(
                {"Dataset": name, "Fold": fold, "Method": method, "Feature": feature_names[index]}
                for index in selected
            )
        for method, weights in weights_by_method.items():
            weight_rows.extend(
                {"Dataset": name, "Fold": fold, "Method": method,
                 "ID": groups[train][i], "time": times[train][i], "weight": weights[i]}
                for i in range(len(train))
            )
    return pd.DataFrame(rows), pd.DataFrame(selected_rows), pd.DataFrame(weight_rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", choices=sorted(DATASETS), default=sorted(DATASETS))
    parser.add_argument("--n-features", type=int, default=3)
    args = parser.parse_args()
    metrics = []
    for name in args.datasets:
        print(f"Running CTSNN benchmark: {name}", flush=True)
        fold_metrics, selected, weights = evaluate(name, args.n_features)
        fold_metrics.to_csv(OUT / f"fold_metrics_{name}_k{args.n_features}.csv", index=False)
        selected.to_csv(OUT / f"selected_features_{name}_k{args.n_features}.csv", index=False)
        weights.to_csv(OUT / f"training_weights_{name}_k{args.n_features}.csv", index=False)
        metrics.append(fold_metrics)
    combined = pd.concat(metrics, ignore_index=True)
    summary = combined.groupby(["Dataset", "Task", "Method"], as_index=False).mean(numeric_only=True)
    label = "_".join(args.datasets)
    summary.to_csv(OUT / f"summary_{label}_k{args.n_features}.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
