"""Ground-truth longitudinal simulations for feature-selection reliability."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from feature_selectors import rank_features


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "simulation"
OUT.mkdir(parents=True, exist_ok=True)

DEFAULT_METHODS = (
    "Lasso", "StabilitySelection", "Stabl-RP", "LongGroupLasso",
    "SNN-FS", "w/o Traj", "w/o Robust", "CTSNN",
)
CONDITIONS = ("clean", "heterogeneous", "irregular_missing", "outliers")


def simulate(condition: str, seed: int, n_subjects: int = 120, p: int = 200, informative: int = 10):
    rng = np.random.default_rng(seed)
    subject_y = rng.binomial(1, 0.5, n_subjects)
    subject_latent = rng.normal(size=(n_subjects, 4))
    loadings = rng.normal(scale=0.35, size=(4, p))
    rows, groups, times, labels = [], [], [], []
    for subject in range(n_subjects):
        if condition == "irregular_missing":
            n_visits = int(rng.integers(2, 6))
            subject_times = np.sort(rng.uniform(0, 1, n_visits))
        else:
            n_visits = 4
            subject_times = np.linspace(0, 1, n_visits)
        intercept_scale = 1.4 if condition == "heterogeneous" else 0.7
        random_intercept = rng.normal(scale=intercept_scale, size=p)
        for time in subject_times:
            value = subject_latent[subject] @ loadings + random_intercept + rng.normal(scale=1.0, size=p)
            # Five persistent and five trajectory-dependent causal features.
            value[: informative // 2] += 1.0 * subject_y[subject]
            value[informative // 2 : informative] += 1.7 * subject_y[subject] * time
            value[:informative] += rng.normal(scale=0.25, size=informative) * time
            if condition == "outliers" and rng.random() < 0.10:
                value += rng.normal(scale=6.0, size=p)
            if condition == "irregular_missing":
                value[rng.random(p) < 0.20] = np.nan
            rows.append(value)
            groups.append(subject)
            times.append(time)
            labels.append(subject_y[subject])
    return np.asarray(rows), np.asarray(labels), np.asarray(groups), np.asarray(times), np.arange(informative)


def participant_split(y, groups, seed):
    unique, first = np.unique(groups, return_index=True)
    train_groups, test_groups = train_test_split(
        unique, test_size=0.30, random_state=seed, stratify=y[first],
    )
    return np.flatnonzero(np.isin(groups, train_groups)), np.flatnonzero(np.isin(groups, test_groups))


def subject_auc(y, prediction, groups):
    table = pd.DataFrame({"y": y, "prediction": prediction, "group": groups})
    table = table.groupby("group", as_index=False).agg(y=("y", "first"), prediction=("prediction", "mean"))
    return roc_auc_score(table.y, table.prediction)


def run(repeats: int, methods, p: int, n_subjects: int):
    records, selections = [], []
    for condition_index, condition in enumerate(CONDITIONS):
        for repeat in range(1, repeats + 1):
            seed = 9000 + condition_index * 1000 + repeat
            X, y, groups, times, truth = simulate(condition, seed, n_subjects=n_subjects, p=p)
            train, test = participant_split(y, groups, seed)
            imputer = SimpleImputer(strategy="median", keep_empty_features=True)
            train_X = imputer.fit_transform(X[train])
            test_X = imputer.transform(X[test])
            for method_index, method in enumerate(methods):
                ranking = rank_features(
                    method, train_X, y[train], groups[train], times[train], "classification",
                    seed + method_index * 100,
                    prefilter_limit=p,
                ).ranking
                selected = ranking[: len(truth)]
                true_positive = len(set(selected).intersection(truth))
                false_positive = len(selected) - true_positive
                scaler = StandardScaler()
                A = scaler.fit_transform(train_X[:, selected])
                B = scaler.transform(test_X[:, selected])
                model = LogisticRegression(
                    C=0.3, class_weight="balanced", max_iter=3_000, random_state=seed,
                ).fit(A, y[train])
                prediction = model.predict_proba(B)[:, 1]
                records.append({
                    "Condition": condition, "Repeat": repeat, "Method": method,
                    "Precision": true_positive / len(selected),
                    "Recall": true_positive / len(truth),
                    "FDR": false_positive / len(selected),
                    "TruthJaccard": true_positive / len(set(selected).union(truth)),
                    "TestAUROC": subject_auc(y[test], prediction, groups[test]),
                })
                selections.extend({
                    "Condition": condition, "Repeat": repeat, "Method": method,
                    "FeatureIndex": int(feature), "IsCausal": bool(feature in truth),
                } for feature in selected)
                print(f"{condition} repeat={repeat} {method}", flush=True)
    return pd.DataFrame(records), pd.DataFrame(selections)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS))
    parser.add_argument("--features", type=int, default=200)
    parser.add_argument("--subjects", type=int, default=120)
    args = parser.parse_args()
    metrics, selections = run(args.repeats, args.methods, args.features, args.subjects)
    summary = metrics.groupby(["Condition", "Method"], as_index=False).agg(
        Precision=("Precision", "mean"), Recall=("Recall", "mean"), FDR=("FDR", "mean"),
        TruthJaccard=("TruthJaccard", "mean"), TestAUROC=("TestAUROC", "mean"),
        TestAUROC_SD=("TestAUROC", "std"),
    )
    metrics.to_csv(OUT / "simulation_replicates.csv", index=False)
    selections.to_csv(OUT / "simulation_selected_features.csv", index=False)
    summary.to_csv(OUT / "simulation_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
