"""Repeated nested participant-level benchmark for SNN-based feature selection."""

from __future__ import annotations

import argparse
import json
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

from feature_selectors import rank_features

warnings.filterwarnings("ignore", message=".*penalty.*deprecated.*", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*Inconsistent values: penalty.*", category=UserWarning)


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
OUT = ROOT / "results" / "main_benchmark" / "benchmark_with_new"
OUT.mkdir(parents=True, exist_ok=True)

DATASETS = {
    "IGG": ("IGG_final.csv", "shannon_diversity", "regression"),
    "IGM": ("IGM_final.csv", "shannon_diversity", "regression"),
    "TRA": ("TRA_final.csv", "shannon_diversity", "regression"),
    "TRB": ("TRB_final.csv", "shannon_diversity", "regression"),
    "OLINK_COVID": ("OLINK_COVID_LONGITUDINAL.csv", "label", "classification"),
    "BRIST1D": ("BRIST1D_DAILY.csv", "label", "regression"),
    "FPG": ("FPG_final.csv", "Fasting_plasma_glucose", "regression"),
    "PE": ("PE_final.csv", "label", "classification"),
    "FPG_KARE": ("FPG_KARE_final.csv", "FPG", "regression"),
    "KARE_STATUS": ("status_KARE_final.csv", "status", "regression"),
    "PE_LOPE_MATCHED": (None, "Label", "classification"),
    "GSE41848_MS": ("GSE41848_MS_LONGITUDINAL.csv", "label", "classification"),
    "GSE41849_MS": ("GSE41849_MS_LONGITUDINAL.csv", "label", "classification"),
    "GSE48023_H1N1": ("GSE48023_H1N1_RESPONSE_LONGITUDINAL.csv", "label", "regression"),
}

DEFAULT_METHODS = (
    "Lasso", "ElasticNet", "StabilitySelection", "Stabl-RP", "LongGroupLasso", "LLSS", "PGEE", "geeVerse", "glmmLasso",
    "RandomForest", "MutualInfo", "Boruta", "mRMR", "SNN-FS", "TrajSNN",
    "TSNN-Stability", "GraphSNN-FS", "SubjectSummarySNN", "OutcomeSNN-FS",
    "LongSNN-FS", "OutcomeTrajSNN-FS", "MultiScaleSNN-FS", "CrossTimeSNN-FS", "DeltaSNN-FS",
    "PrototypeSNN-FS", "ConsensusSNN-FS",
)

# Some official longitudinal selectors are outcome-type specific. They remain
# in the shared registry, but are not forced onto scientifically incompatible
# tasks and therefore do not create artificial missing/failed scores.
METHOD_TASKS = {
    "geeVerse": {"regression"},
}


def _numeric_time(series: pd.Series) -> np.ndarray:
    mapping = {
        "0M": 0.0, "1M": 1.0, "3M": 3.0, "12M": 12.0,
        "1st Tri": 1.0, "2nd Tri": 2.0, "3rd Tri": 3.0,
    }
    converted = series.astype(str).map(mapping)
    fallback = pd.to_numeric(series, errors="coerce")
    result = converted.fillna(fallback)
    if result.isna().any():
        codes, _ = pd.factorize(series.astype(str), sort=True)
        result = result.fillna(pd.Series(codes, index=series.index))
    return result.to_numpy(float)


def load_dataset(name: str):
    filename, target, task = DATASETS[name]
    include_time = True
    if name == "PE_LOPE_MATCHED":
        external = ROOT / "data" / "external" / "processed"
        frame = pd.read_csv(external / "Stanford_SomaLogic_longitudinal_PE.csv")
        detroit_columns = set(pd.read_csv(
            external / "Detroit_SomaLogic_longitudinal_PE.csv", nrows=0,
        ).columns)
        frame = frame[
            (frame["Phenotype"] != "PE-early") & (frame["GestationalAge"] <= 22)
        ].copy()
        metadata = {
            "Cohort", "ID", "GestationalAge", "Phenotype", "Label", "LatePE",
            "ClinicalGroup", "MaternalAge", "BMI", "Race", "Gravidity", "Parity",
        }
        candidates = [column for column in frame if column in detroit_columns and column not in metadata]
        frame = frame[["ID", "GestationalAge", target, *candidates]].rename(
            columns={"GestationalAge": "time"}
        )
        include_time = False
    else:
        frame = pd.read_csv(DATA / filename)
    candidates = [column for column in frame if column not in {"ID", "time", target}]
    numeric = frame[candidates].apply(pd.to_numeric, errors="coerce")
    keep = numeric.notna().mean() >= 0.8
    feature_names = list(numeric.columns[keep])
    X = numeric.loc[:, keep].to_numpy(float)
    if task == "classification":
        if pd.api.types.is_numeric_dtype(frame[target]):
            y = pd.to_numeric(frame[target], errors="raise").astype(int).to_numpy()
        else:
            y = (frame[target].astype(str) == "PE").astype(int).to_numpy()
    else:
        y = pd.to_numeric(frame[target], errors="raise").to_numpy(float)
    subject_level_regression = bool(
        task == "regression"
        and frame.groupby("ID")[target].nunique(dropna=False).max() == 1
    )
    return (
        frame, X, y, feature_names, task, _numeric_time(frame["time"]),
        include_time, subject_level_regression,
    )


def group_folds(y: np.ndarray, groups: np.ndarray, task: str, n_splits: int, seed: int):
    unique_groups, first = np.unique(groups, return_index=True)
    group_y = y[first]
    if task == "classification":
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        iterator = splitter.split(unique_groups, group_y)
    else:
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        iterator = splitter.split(unique_groups)
    for group_train, group_test in iterator:
        train_groups = unique_groups[group_train]
        test_groups = unique_groups[group_test]
        yield np.flatnonzero(np.isin(groups, train_groups)), np.flatnonzero(np.isin(groups, test_groups))


def preprocess(train_X: np.ndarray, other_X: np.ndarray):
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    return imputer.fit_transform(train_X), imputer.transform(other_X)


def fit_predict(
    train_X: np.ndarray,
    test_X: np.ndarray,
    train_y: np.ndarray,
    train_time: np.ndarray,
    test_time: np.ndarray,
    selected: np.ndarray,
    task: str,
    regularization: float,
    include_time: bool = True,
) -> np.ndarray:
    scaler = StandardScaler()
    A = np.sign(train_X[:, selected]) * np.log1p(np.abs(train_X[:, selected]))
    B = np.sign(test_X[:, selected]) * np.log1p(np.abs(test_X[:, selected]))
    A = scaler.fit_transform(A)
    B = scaler.transform(B)
    if include_time:
        time_scaler = StandardScaler()
        t_train = time_scaler.fit_transform(train_time.reshape(-1, 1))
        t_test = time_scaler.transform(test_time.reshape(-1, 1))
        A = np.column_stack([A, t_train])
        B = np.column_stack([B, t_test])
    if task == "regression":
        return Ridge(alpha=regularization).fit(A, train_y).predict(B)
    model = LogisticRegression(
        C=regularization, penalty="l2", class_weight="balanced", max_iter=4_000,
        random_state=2026,
    ).fit(A, train_y)
    return model.predict_proba(B)[:, 1]


def _subject_classification(y, prediction, groups):
    table = pd.DataFrame({"y": y, "prediction": prediction, "group": groups})
    aggregate = table.groupby("group", as_index=False).agg(y=("y", "first"), prediction=("prediction", "mean"))
    return aggregate.y.to_numpy(), aggregate.prediction.to_numpy(), aggregate.group.to_numpy()


def _subject_regression(y, prediction, groups):
    table = pd.DataFrame({"y": y, "prediction": prediction, "group": groups})
    aggregate = table.groupby("group", as_index=False).agg(
        y=("y", "first"), prediction=("prediction", "mean"),
    )
    return aggregate.y.to_numpy(), aggregate.prediction.to_numpy(), aggregate.group.to_numpy()


def score_predictions(y, prediction, groups, task, subject_level_regression=False):
    if task == "regression":
        if subject_level_regression:
            y, prediction, _ = _subject_regression(y, prediction, groups)
        return -float(np.sqrt(mean_squared_error(y, prediction)))
    y_subject, p_subject, _ = _subject_classification(y, prediction, groups)
    return roc_auc_score(y_subject, p_subject)


def metric_rows(
    dataset, repeat, fold, method, y, prediction, groups, task, budget,
    regularization, subject_level_regression=False,
):
    base = {
        "Dataset": dataset, "Repeat": repeat, "Fold": fold, "Method": method,
        "n_features": int(budget), "PredictorRegularization": float(regularization),
    }
    if task == "classification":
        y, prediction, _ = _subject_classification(y, prediction, groups)
        labels = prediction >= 0.5
        base.update(
            AUROC=roc_auc_score(y, prediction),
            AUPRC=average_precision_score(y, prediction),
            Accuracy=accuracy_score(y, labels),
            BalancedAccuracy=balanced_accuracy_score(y, labels),
        )
    else:
        if subject_level_regression:
            y, prediction, _ = _subject_regression(y, prediction, groups)
        base.update(
            RMSE=float(np.sqrt(mean_squared_error(y, prediction))),
            MAE=mean_absolute_error(y, prediction),
        )
    return base


def tune_method(
    method, X, y, groups, times, task, budgets, seed, include_time=True,
    subject_level_regression=False,
):
    regularizations = (0.1, 1.0, 10.0) if task == "classification" else (1.0, 10.0, 100.0)
    results = {(budget, reg): [] for budget in budgets for reg in regularizations}
    for inner_fold, (train, valid) in enumerate(group_folds(y, groups, task, 3, seed), 1):
        A, B = preprocess(X[train], X[valid])
        ranking = rank_features(
            method, A, y[train], groups[train], times[train], task,
            seed + 100 * inner_fold,
        ).ranking
        for budget, regularization in results:
            selected = ranking[: min(budget, len(ranking))]
            prediction = fit_predict(
                A, B, y[train], times[train], times[valid], selected, task, regularization,
                include_time,
            )
            results[(budget, regularization)].append(
                score_predictions(
                    y[valid], prediction, groups[valid], task,
                    subject_level_regression,
                )
            )
    mean_scores = {key: float(np.mean(values)) for key, values in results.items()}
    # Both tasks are maximized because regression score is negative RMSE.
    best = max(mean_scores, key=lambda key: (mean_scores[key], -key[0], -key[1]))
    return best, mean_scores


def benjamini_hochberg(pvalues: np.ndarray) -> np.ndarray:
    pvalues = np.asarray(pvalues, float)
    order = np.argsort(pvalues)
    ranked = pvalues[order]
    adjusted = np.minimum.accumulate((ranked * len(ranked) / np.arange(1, len(ranked) + 1))[::-1])[::-1]
    output = np.empty_like(adjusted)
    output[order] = np.minimum(adjusted, 1.0)
    return output


def stability_table(selected: pd.DataFrame, total_features: dict[str, int]) -> pd.DataFrame:
    rows = []
    for (dataset, method), subset in selected.groupby(["Dataset", "Method"]):
        sets = [set(group.Feature) for _, group in subset.groupby(["Repeat", "Fold"])]
        jaccard, kuncheva = [], []
        p = total_features[dataset]
        for left, right in combinations(sets, 2):
            union = left | right
            jaccard.append(len(left & right) / max(1, len(union)))
            k = min(len(left), len(right))
            if k and p > k:
                kuncheva.append((len(left & right) - k * k / p) / (k - k * k / p))
        rows.append({
            "Dataset": dataset, "Method": method,
            "MeanJaccard": float(np.mean(jaccard)) if jaccard else np.nan,
            "MeanKuncheva": float(np.mean(kuncheva)) if kuncheva else np.nan,
            "SelectionRuns": len(sets),
        })
    return pd.DataFrame(rows)


def summarize(metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_parts = []
    comparisons = []
    for dataset, subset in metrics.groupby("Dataset"):
        metric_name = "AUROC" if "AUROC" in subset and subset.AUROC.notna().any() else "RMSE"
        part = subset.groupby("Method", as_index=False).agg(
            Mean=(metric_name, "mean"), SD=(metric_name, "std"), N=(metric_name, "count")
        )
        part.insert(0, "Dataset", dataset)
        part.insert(2, "Metric", metric_name)
        part["CI95Lower"] = part.Mean - 1.96 * part.SD / np.sqrt(part.N)
        part["CI95Upper"] = part.Mean + 1.96 * part.SD / np.sqrt(part.N)
        summary_parts.append(part)
        pivot = subset.pivot(index=["Repeat", "Fold"], columns="Method", values=metric_name)
        if "Lasso" not in pivot:
            continue
        for method in pivot.columns:
            if method == "Lasso":
                continue
            pair = pivot[[method, "Lasso"]].dropna()
            difference = pair[method] - pair["Lasso"]
            if len(pair) >= 3 and np.any(np.abs(difference) > 1e-12):
                pvalue = wilcoxon(difference, alternative="two-sided").pvalue
            else:
                pvalue = 1.0
            comparisons.append({
                "Dataset": dataset, "Method": method, "Reference": "Lasso",
                "Metric": metric_name, "MeanPairedDifference": float(difference.mean()),
                "WilcoxonP": float(pvalue),
            })
    comparisons = pd.DataFrame(comparisons)
    if not comparisons.empty:
        comparisons["BHAdjustedP"] = comparisons.groupby("Dataset")["WilcoxonP"].transform(
            lambda values: benjamini_hochberg(values.to_numpy())
        )
    return pd.concat(summary_parts, ignore_index=True), comparisons


def evaluate_dataset(
    name: str,
    methods,
    repeats: int,
    budgets,
    checkpoint_dir: Path | None = None,
    repeat_start: int = 1,
):
    (
        frame, X, y, feature_names, task, times, include_time,
        subject_level_regression,
    ) = load_dataset(name)
    groups = frame.ID.to_numpy()
    methods = [method for method in methods if task in METHOD_TASKS.get(method, {"classification", "regression"})]
    if not methods:
        raise ValueError(f"No eligible methods for {name} ({task}).")
    metric_records, selection_records, prediction_records, tuning_records = [], [], [], []
    for repeat in range(repeat_start, repeat_start + repeats):
        seed = 2026 + repeat * 1000
        for fold, (train, test) in enumerate(group_folds(y, groups, task, 5, seed), 1):
            train_X, test_X = preprocess(X[train], X[test])
            for method_index, method in enumerate(methods):
                method_seed = seed + fold * 100 + method_index
                (budget, regularization), tuning = tune_method(
                    method, train_X, y[train], groups[train], times[train], task, budgets, method_seed,
                    include_time, subject_level_regression,
                )
                ranking_result = rank_features(
                    method, train_X, y[train], groups[train], times[train], task, method_seed + 700,
                )
                selected = ranking_result.ranking[: min(budget, len(feature_names))]
                prediction = fit_predict(
                    train_X, test_X, y[train], times[train], times[test], selected, task, regularization,
                    include_time,
                )
                metric_records.append(metric_rows(
                    name, repeat, fold, method, y[test], prediction, groups[test], task,
                    budget, regularization, subject_level_regression,
                ))
                for rank, feature_index in enumerate(selected, 1):
                    selection_records.append({
                        "Dataset": name, "Repeat": repeat, "Fold": fold, "Method": method,
                        "Rank": rank, "Feature": feature_names[feature_index],
                        "Score": ranking_result.scores[feature_index],
                    })
                if task == "classification":
                    yy, pp, gg = _subject_classification(y[test], prediction, groups[test])
                    for subject, outcome, value in zip(gg, yy, pp):
                        prediction_records.append({
                            "Dataset": name, "Repeat": repeat, "Fold": fold, "Method": method,
                            "ID": subject, "Outcome": outcome, "Prediction": value,
                        })
                else:
                    if subject_level_regression:
                        yy, pp, gg = _subject_regression(y[test], prediction, groups[test])
                        for subject, outcome, value in zip(gg, yy, pp):
                            prediction_records.append({
                                "Dataset": name, "Repeat": repeat, "Fold": fold, "Method": method,
                                "ID": subject, "Outcome": outcome, "Prediction": value,
                            })
                    else:
                        for row, value in zip(test, prediction):
                            prediction_records.append({
                                "Dataset": name, "Repeat": repeat, "Fold": fold, "Method": method,
                                "ID": groups[row], "Outcome": y[row], "Prediction": value,
                            })
                tuning_records.append({
                    "Dataset": name, "Repeat": repeat, "Fold": fold, "Method": method,
                    "SelectedBudget": budget, "SelectedRegularization": regularization,
                    "InnerScores": json.dumps({f"k={k};r={r}": v for (k, r), v in tuning.items()}),
                    **ranking_result.metadata,
                })
                print(f"{name} repeat={repeat} fold={fold} {method} k={budget}", flush=True)
        if checkpoint_dir is not None:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(metric_records).to_csv(
                checkpoint_dir / f"{name}_metrics_checkpoint.csv", index=False,
            )
            pd.DataFrame(selection_records).to_csv(
                checkpoint_dir / f"{name}_selected_checkpoint.csv", index=False,
            )
    return tuple(map(pd.DataFrame, (metric_records, selection_records, prediction_records, tuning_records))), len(feature_names)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=["PE"])
    parser.add_argument("--methods", nargs="+", choices=DEFAULT_METHODS, default=list(DEFAULT_METHODS))
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument(
        "--repeat-start", type=int, default=1,
        help="First one-based repeat label; used by Slurm arrays to create distinct random splits.",
    )
    parser.add_argument("--budgets", nargs="+", type=int, default=[3, 5, 10])
    parser.add_argument("--output-dir", default=str(OUT))
    args = parser.parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    all_metrics, all_selected, all_predictions, all_tuning = [], [], [], []
    total_features = {}
    for dataset in args.datasets:
        tables, p = evaluate_dataset(
            dataset, args.methods, args.repeats, args.budgets,
            checkpoint_dir=output_dir, repeat_start=args.repeat_start,
        )
        metrics, selected, predictions, tuning = tables
        all_metrics.append(metrics)
        all_selected.append(selected)
        all_predictions.append(predictions)
        all_tuning.append(tuning)
        total_features[dataset] = p
    metrics = pd.concat(all_metrics, ignore_index=True)
    selected = pd.concat(all_selected, ignore_index=True)
    predictions = pd.concat(all_predictions, ignore_index=True)
    tuning = pd.concat(all_tuning, ignore_index=True)
    summary, comparisons = summarize(metrics)
    stability = stability_table(selected, total_features)
    metrics.to_csv(output_dir / "fold_metrics.csv", index=False)
    selected.to_csv(output_dir / "selected_features.csv", index=False)
    predictions.to_csv(output_dir / "outer_predictions.csv", index=False)
    tuning.to_csv(output_dir / "inner_tuning.csv", index=False)
    summary.to_csv(output_dir / "performance_summary.csv", index=False)
    comparisons.to_csv(output_dir / "paired_comparisons.csv", index=False)
    stability.to_csv(output_dir / "selection_stability.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
