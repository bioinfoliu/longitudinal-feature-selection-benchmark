"""Bidirectional locked validation between Stanford and Detroit SomaLogic PE cohorts.

The analysis matches assays by SomaId, restricts Stanford to late-onset PE, uses
early samples (<=22 weeks), and tunes only within the source cohort. Target
labels never affect preprocessing, feature selection, panel size, penalties, or
clinical-covariate handling.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from feature_selectors import rank_features
from run_expanded_benchmark import group_folds
from summarize_expanded_results import fast_auc


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "external" / "processed"
OUT = ROOT / "results" / "validation" / "external_validation"
OUT.mkdir(parents=True, exist_ok=True)

METHODS = (
    "Lasso", "ElasticNet", "StabilitySelection", "Stabl-RP",
    "LongGroupLasso", "LLSS", "RandomForest", "MutualInfo", "Boruta",
    "mRMR", "SNN-FS", "OutcomeSNN-FS",
)
CLINICAL = ("MaternalAge", "BMI", "Parity")
NON_PROTEIN = {
    "Cohort", "ID", "GestationalAge", "Phenotype", "Label", "LatePE",
    "ClinicalGroup", "MaternalAge", "BMI", "Race", "Gravidity", "Parity",
}


def load_cohort(name: str) -> pd.DataFrame:
    frame = pd.read_csv(DATA / f"{name}_SomaLogic_longitudinal_PE.csv")
    frame = frame[frame.GestationalAge <= 22].copy()
    if name == "Stanford":
        frame = frame[frame.Phenotype != "PE-early"].copy()
    frame["Outcome"] = frame.LatePE.astype(int)
    return frame


def percentile_rows(X: np.ndarray) -> np.ndarray:
    output = np.empty_like(X, dtype=float)
    for row in range(len(X)):
        valid = np.isfinite(X[row])
        output[row] = np.nan
        output[row, valid] = rankdata(X[row, valid], method="average") / (valid.sum() + 1.0)
    return output


def subject_metrics(y, prediction, groups):
    table = pd.DataFrame({"Outcome": y, "Prediction": prediction, "ID": groups})
    table = table.groupby("ID", as_index=False).agg(Outcome=("Outcome", "first"), Prediction=("Prediction", "mean"))
    return roc_auc_score(table.Outcome, table.Prediction), average_precision_score(table.Outcome, table.Prediction), table


def make_design(
    train_X, other_X, train_ga, other_ga, train_clinical, other_clinical,
    selected, adjusted, include_ga=False,
):
    parts_a, parts_b = [], []
    if len(selected):
        protein_imputer = SimpleImputer(strategy="median", keep_empty_features=True)
        A = protein_imputer.fit_transform(train_X[:, selected])
        B = protein_imputer.transform(other_X[:, selected])
        protein_scaler = StandardScaler().fit(A)
        parts_a.append(protein_scaler.transform(A))
        parts_b.append(protein_scaler.transform(B))

    if include_ga:
        ga_scaler = StandardScaler().fit(train_ga.reshape(-1, 1))
        parts_a.append(ga_scaler.transform(train_ga.reshape(-1, 1)))
        parts_b.append(ga_scaler.transform(other_ga.reshape(-1, 1)))
    if adjusted:
        clinical_imputer = SimpleImputer(strategy="median", keep_empty_features=True)
        C = clinical_imputer.fit_transform(train_clinical)
        D = clinical_imputer.transform(other_clinical)
        clinical_scaler = StandardScaler().fit(C)
        parts_a.append(clinical_scaler.transform(C))
        parts_b.append(clinical_scaler.transform(D))
    return np.column_stack(parts_a), np.column_stack(parts_b)


def visit_weights(groups):
    counts = pd.Series(groups).value_counts()
    return np.array([1.0 / counts[group] for group in groups], float)


def tune_source(method, X, y, groups, times, clinical, seed):
    budgets = (3, 5, 10)
    penalties = (0.1, 1.0, 10.0)
    scores = {
        adjusted: {(budget, penalty): [] for budget in budgets for penalty in penalties}
        for adjusted in (False, True)
    }
    for fold, (train, valid) in enumerate(group_folds(y, groups, "classification", 3, seed), 1):
        ranking = rank_features(
            method, X[train], y[train], groups[train], times[train], "classification", seed + fold * 101,
        ).ranking
        for adjusted in (False, True):
            for budget in budgets:
                selected = ranking[:budget]
                A, B = make_design(
                    X[train], X[valid], times[train], times[valid],
                    clinical[train], clinical[valid], selected, adjusted,
                )
                for penalty in penalties:
                    model = LogisticRegression(
                        C=penalty, class_weight="balanced", max_iter=4_000, random_state=seed,
                    ).fit(A, y[train], sample_weight=visit_weights(groups[train]))
                    prediction = model.predict_proba(B)[:, 1]
                    auc, _, _ = subject_metrics(y[valid], prediction, groups[valid])
                    scores[adjusted][(budget, penalty)].append(auc)
    best = {}
    means = {}
    for adjusted in (False, True):
        means[adjusted] = {key: float(np.mean(value)) for key, value in scores[adjusted].items()}
        best[adjusted] = max(means[adjusted], key=lambda key: (means[adjusted][key], -key[0], -key[1]))
    return best, means


def bootstrap_intervals(subject: pd.DataFrame, iterations: int, seed: int):
    rng = np.random.default_rng(seed)
    y = subject.Outcome.to_numpy(int)
    p = subject.Prediction.to_numpy(float)
    aucs, aprs = [], []
    for _ in range(iterations):
        sample = rng.integers(0, len(subject), len(subject))
        if np.unique(y[sample]).size < 2:
            continue
        aucs.append(fast_auc(y[sample], p[sample]))
        aprs.append(average_precision_score(y[sample], p[sample]))
    return (*np.quantile(aucs, [0.025, 0.975]), *np.quantile(aprs, [0.025, 0.975]))


def evaluate_direction(source_name, target_name, source, target, features, method_index):
    source_X = percentile_rows(source[features].apply(pd.to_numeric, errors="coerce").to_numpy(float))
    target_X = percentile_rows(target[features].apply(pd.to_numeric, errors="coerce").to_numpy(float))
    source_y = source.Outcome.to_numpy(int)
    target_y = target.Outcome.to_numpy(int)
    source_groups = source.ID.to_numpy()
    target_groups = target.ID.to_numpy()
    source_time = source.GestationalAge.to_numpy(float)
    target_time = target.GestationalAge.to_numpy(float)
    source_clinical = source[list(CLINICAL)].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    target_clinical = target[list(CLINICAL)].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    method = METHODS[method_index]
    seed = 33000 + method_index * 1000 + (0 if source_name == "Stanford" else 500)
    best, tuning = tune_source(method, source_X, source_y, source_groups, source_time, source_clinical, seed)
    ranking = rank_features(
        method, source_X, source_y, source_groups, source_time, "classification", seed + 777,
    ).ranking
    performance, panels, predictions, tuning_rows = [], [], [], []
    for adjusted in (False, True):
        budget, penalty = best[adjusted]
        selected = ranking[:budget]
        A, B = make_design(
            source_X, target_X, source_time, target_time,
            source_clinical, target_clinical, selected, adjusted,
        )
        model = LogisticRegression(
            C=penalty, class_weight="balanced", max_iter=4_000, random_state=seed,
        ).fit(A, source_y, sample_weight=visit_weights(source_groups))
        visit_prediction = model.predict_proba(B)[:, 1]
        auc, auprc, subject = subject_metrics(target_y, visit_prediction, target_groups)
        ci_auc_l, ci_auc_u, ci_ap_l, ci_ap_u = bootstrap_intervals(subject, 1000, seed + int(adjusted))
        variant = "ProteinPlusClinical" if adjusted else "ProteinOnly"
        performance.append({
            "TrainCohort": source_name, "TestCohort": target_name, "Method": method,
            "ModelVariant": variant, "SelectedFeatures": budget, "SharedAssays": len(features),
            "TrainParticipants": source.ID.nunique(), "TestParticipants": target.ID.nunique(),
            "AUROC": auc, "AUROC_CI95Lower": ci_auc_l, "AUROC_CI95Upper": ci_auc_u,
            "AUPRC": auprc, "AUPRC_CI95Lower": ci_ap_l, "AUPRC_CI95Upper": ci_ap_u,
            "SourceSelectedC": penalty,
        })
        panels.extend({
            "TrainCohort": source_name, "TestCohort": target_name, "Method": method,
            "ModelVariant": variant, "Rank": rank, "Feature": features[index],
        } for rank, index in enumerate(selected, 1))
        subject.insert(0, "ModelVariant", variant)
        subject.insert(0, "Method", method)
        subject.insert(0, "TestCohort", target_name)
        subject.insert(0, "TrainCohort", source_name)
        predictions.append(subject)
        tuning_rows.append({
            "TrainCohort": source_name, "TestCohort": target_name, "Method": method,
            "ModelVariant": variant, "SelectedFeatures": budget, "SelectedC": penalty,
            "MeanInnerAUROC": tuning[adjusted][(budget, penalty)],
            "AllInnerScores": json.dumps({f"k={k};C={c}": value for (k, c), value in tuning[adjusted].items()}),
        })
    return performance, panels, pd.concat(predictions, ignore_index=True), tuning_rows


def evaluate_clinical_baselines(source_name, target_name, source, target):
    source_y = source.Outcome.to_numpy(int)
    target_y = target.Outcome.to_numpy(int)
    source_groups = source.ID.to_numpy()
    target_groups = target.ID.to_numpy()
    source_time = source.GestationalAge.to_numpy(float)
    target_time = target.GestationalAge.to_numpy(float)
    source_clinical = source[list(CLINICAL)].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    target_clinical = target[list(CLINICAL)].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    seed = 47000 + (0 if source_name == "Stanford" else 500)
    records, prediction_tables, tuning_rows = [], [], []
    for variant, use_clinical, include_ga in (
        ("GestationalAgeOnly", False, True), ("ClinicalOnly", True, False)
    ):
        candidate_scores = {penalty: [] for penalty in (0.1, 1.0, 10.0)}
        for train, valid in group_folds(source_y, source_groups, "classification", 3, seed):
            empty_train = np.empty((len(train), 0))
            empty_valid = np.empty((len(valid), 0))
            A, B = make_design(
                empty_train, empty_valid, source_time[train], source_time[valid],
                source_clinical[train], source_clinical[valid], np.array([], dtype=int),
                use_clinical, include_ga,
            )
            for penalty in candidate_scores:
                model = LogisticRegression(
                    C=penalty, class_weight="balanced", max_iter=4_000, random_state=seed,
                ).fit(A, source_y[train], sample_weight=visit_weights(source_groups[train]))
                prediction = model.predict_proba(B)[:, 1]
                auc, _, _ = subject_metrics(source_y[valid], prediction, source_groups[valid])
                candidate_scores[penalty].append(auc)
        mean_scores = {penalty: float(np.mean(values)) for penalty, values in candidate_scores.items()}
        best_penalty = max(mean_scores, key=lambda value: (mean_scores[value], -value))
        empty_source = np.empty((len(source), 0))
        empty_target = np.empty((len(target), 0))
        A, B = make_design(
            empty_source, empty_target, source_time, target_time,
            source_clinical, target_clinical, np.array([], dtype=int), use_clinical, include_ga,
        )
        model = LogisticRegression(
            C=best_penalty, class_weight="balanced", max_iter=4_000, random_state=seed,
        ).fit(A, source_y, sample_weight=visit_weights(source_groups))
        visit_prediction = model.predict_proba(B)[:, 1]
        auc, auprc, subject = subject_metrics(target_y, visit_prediction, target_groups)
        ci_auc_l, ci_auc_u, ci_ap_l, ci_ap_u = bootstrap_intervals(subject, 1000, seed + int(use_clinical))
        records.append({
            "TrainCohort": source_name, "TestCohort": target_name, "Method": "ClinicalBaseline",
            "ModelVariant": variant, "SelectedFeatures": 0, "SharedAssays": 0,
            "TrainParticipants": source.ID.nunique(), "TestParticipants": target.ID.nunique(),
            "AUROC": auc, "AUROC_CI95Lower": ci_auc_l, "AUROC_CI95Upper": ci_auc_u,
            "AUPRC": auprc, "AUPRC_CI95Lower": ci_ap_l, "AUPRC_CI95Upper": ci_ap_u,
            "SourceSelectedC": best_penalty,
        })
        subject.insert(0, "ModelVariant", variant)
        subject.insert(0, "Method", "ClinicalBaseline")
        subject.insert(0, "TestCohort", target_name)
        subject.insert(0, "TrainCohort", source_name)
        prediction_tables.append(subject)
        tuning_rows.append({
            "TrainCohort": source_name, "TestCohort": target_name, "Method": "ClinicalBaseline",
            "ModelVariant": variant, "SelectedFeatures": 0, "SelectedC": best_penalty,
            "MeanInnerAUROC": mean_scores[best_penalty],
            "AllInnerScores": json.dumps({f"C={key}": value for key, value in mean_scores.items()}),
        })
    return records, pd.concat(prediction_tables, ignore_index=True), tuning_rows


def main():
    stanford = load_cohort("Stanford")
    detroit = load_cohort("Detroit")
    features = sorted((set(stanford.columns) - NON_PROTEIN - {"Outcome"}) & (set(detroit.columns) - NON_PROTEIN - {"Outcome"}))
    cohort_summary = pd.DataFrame([
        {"Cohort": name, "Visits": len(frame), "Participants": frame.ID.nunique(),
         "Cases": frame.loc[frame.Outcome == 1, "ID"].nunique(),
         "Controls": frame.loc[frame.Outcome == 0, "ID"].nunique(),
         "MinimumGA": frame.GestationalAge.min(), "MaximumGA": frame.GestationalAge.max()}
        for name, frame in (("Stanford", stanford), ("Detroit", detroit))
    ])
    records, panels, predictions, tuning = [], [], [], []
    for source_name, target_name, source, target in (
        ("Stanford", "Detroit", stanford, detroit),
        ("Detroit", "Stanford", detroit, stanford),
    ):
        for method_index in range(len(METHODS)):
            a, b, c, d = evaluate_direction(source_name, target_name, source, target, features, method_index)
            records.extend(a); panels.extend(b); predictions.append(c); tuning.extend(d)
            print(source_name, "->", target_name, METHODS[method_index], flush=True)
        baseline_records, baseline_predictions, baseline_tuning = evaluate_clinical_baselines(
            source_name, target_name, source, target
        )
        records.extend(baseline_records); predictions.append(baseline_predictions); tuning.extend(baseline_tuning)
    pd.DataFrame(records).to_csv(OUT / "transfer_performance.csv", index=False)
    pd.DataFrame(panels).to_csv(OUT / "locked_panels.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_csv(OUT / "subject_predictions.csv", index=False)
    pd.DataFrame(tuning).to_csv(OUT / "source_inner_tuning.csv", index=False)
    cohort_summary.to_csv(OUT / "cohort_summary.csv", index=False)
    print(pd.DataFrame(records).sort_values(["TrainCohort", "ModelVariant", "AUROC"], ascending=[True, True, False]).to_string(index=False))


if __name__ == "__main__":
    main()
