"""Leakage-safe feature rankers used in the expanded CTSNN benchmark.

The Stabl-RP implementation follows the public Stabl v1.0.1 lightweight
random-permutation algorithm (commit 2f048470): artificial columns are made
once, group subsamples are reused over a sparse-penalty path, and the FDP+
cutoff is estimated from the maximum artificial-feature selection frequency.
It is kept dependency-light because the official package's optional knockpy
extension does not build on every platform.  It is never labelled as the
official package in result files.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import subprocess
import tempfile

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.feature_selection import f_classif, f_regression, mutual_info_classif, mutual_info_regression
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from ctsnn import canonical_method_name, ctsnn_weights, prepare_distance_embedding, snn_weights
from longitudinal_lasso import LongitudinalLassoSelector


def _safe_scores(values: np.ndarray) -> np.ndarray:
    return np.nan_to_num(np.asarray(values, float), nan=-np.inf, posinf=np.inf, neginf=-np.inf)


def _rank_desc(scores: np.ndarray) -> np.ndarray:
    scores = _safe_scores(scores)
    return np.lexsort((np.arange(len(scores)), scores))[::-1]


def _signed_log_scale(X: np.ndarray) -> np.ndarray:
    X = np.sign(X) * np.log1p(np.abs(X))
    return StandardScaler().fit_transform(X)


def _prefilter(X: np.ndarray, y: np.ndarray, task: str, limit: int = 300) -> tuple[np.ndarray, np.ndarray]:
    if X.shape[1] <= limit:
        index = np.arange(X.shape[1])
    else:
        scores, _ = (f_regression(X, y) if task == "regression" else f_classif(X, y))
        index = np.sort(_rank_desc(scores)[:limit])
    return index, X[:, index]


def _sparse_scores(X: np.ndarray, y: np.ndarray, task: str, weights=None, l1_ratio: float = 1.0) -> np.ndarray:
    Xs = _signed_log_scale(X)
    if task == "regression":
        model = ElasticNet(
            alpha=0.03,
            l1_ratio=l1_ratio,
            max_iter=20_000,
            random_state=2026,
        ).fit(Xs, y, sample_weight=weights)
        return np.abs(model.coef_)
    penalty = "l1" if l1_ratio >= 0.999 else "elasticnet"
    solver = "liblinear" if penalty == "l1" else "saga"
    model = LogisticRegression(
        C=0.3,
        penalty=penalty,
        solver=solver,
        l1_ratio=None if penalty == "l1" else l1_ratio,
        class_weight="balanced",
        max_iter=5_000,
        random_state=2026,
    ).fit(Xs, y, sample_weight=weights)
    return np.abs(model.coef_).ravel()


def _glmmlasso_scores(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    task: str,
    seed: int,
) -> tuple[np.ndarray, float, float]:
    """Feature magnitudes from the official glmmLasso random-intercept model.

    The adapter operates on a prefiltered, transformed training fold. It asks
    the R package to choose from a small BIC-selected lambda grid and never
    falls back to a different selector if the GLMM fails.
    """
    rscript = os.environ.get("CTSNN_LONGFS_RSCRIPT", "Rscript")
    adapter = Path(__file__).resolve().parents[1] / "scripts" / "rank_glmmlasso.R"
    if not adapter.exists():
        raise FileNotFoundError(f"Missing glmmLasso adapter: {adapter}")
    names = [f"x{i:03d}" for i in range(X.shape[1])]
    with tempfile.TemporaryDirectory(prefix="ctsnn_glmmlasso_") as temp:
        temp_dir = Path(temp)
        source, destination = temp_dir / "input.csv", temp_dir / "scores.csv"
        frame = pd.DataFrame(_signed_log_scale(X), columns=names)
        frame.insert(0, "subject", np.asarray(groups).astype(str))
        frame.insert(0, "outcome", np.asarray(y, dtype=float))
        frame.to_csv(source, index=False)
        completed = subprocess.run(
            [rscript, str(adapter), str(source), str(destination), task, str(seed)],
            check=False, capture_output=True, text=True, timeout=300,
        )
        if completed.returncode != 0 or not destination.exists():
            detail = completed.stderr[-1000:] if completed.stderr else completed.stdout[-1000:]
            raise RuntimeError(f"glmmLasso adapter failed: {detail}")
        result = pd.read_csv(destination).set_index("Feature")
        scores = result.reindex(names)["Score"].to_numpy(float)
        if not np.isfinite(scores).all():
            raise RuntimeError("glmmLasso adapter returned non-finite feature scores.")
        return scores, float(result["Lambda"].iloc[0]), float(result["BIC"].iloc[0])


def _pgee_scores(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    task: str,
    seed: int,
) -> tuple[np.ndarray, float, float]:
    """Feature magnitudes from the official PGEE SCAD-penalized GEE package.

    PGEE's package-native three-fold CV chooses the SCAD penalty on each
    training partition. Its clustered estimating equation receives participant
    IDs but no outcome or held-out observations outside that partition.
    """
    rscript = os.environ.get("CTSNN_LONGFS_RSCRIPT", "Rscript")
    adapter = Path(__file__).resolve().parents[1] / "scripts" / "rank_pgee.R"
    if not adapter.exists():
        raise FileNotFoundError(f"Missing PGEE adapter: {adapter}")
    names = [f"x{i:03d}" for i in range(X.shape[1])]
    with tempfile.TemporaryDirectory(prefix="ctsnn_pgee_") as temp:
        temp_dir = Path(temp)
        source, destination = temp_dir / "input.csv", temp_dir / "scores.csv"
        frame = pd.DataFrame(_signed_log_scale(X), columns=names)
        # PGEE 1.5's legacy formula evaluator requires conventional `y` and
        # `id` names; neither is exposed outside this temporary adapter file.
        # PGEE 1.5 coerces cluster IDs to integer internally. Factorizing
        # inside this training-fold-only temporary file preserves membership
        # while supporting arbitrary cohort subject labels.
        frame.insert(0, "id", pd.factorize(np.asarray(groups).astype(str))[0] + 1)
        frame.insert(0, "y", np.asarray(y, dtype=float))
        frame.to_csv(source, index=False)
        environment = os.environ.copy()
        # An absolute Rscript path alone does not put the matching Conda
        # compilers/libraries on PATH. Keeping its bin directory first makes
        # this adapter work in the isolated server environment as well.
        rscript_directory = str(Path(rscript).expanduser().resolve().parent)
        environment["PATH"] = f"{rscript_directory}{os.pathsep}{environment.get('PATH', '')}"
        completed = subprocess.run(
            [rscript, str(adapter), str(source), str(destination), task, str(seed)],
            check=False, capture_output=True, text=True, timeout=180, env=environment,
        )
        if completed.returncode != 0 or not destination.exists():
            detail = completed.stderr[-1000:] if completed.stderr else completed.stdout[-1000:]
            raise RuntimeError(f"PGEE adapter failed: {detail}")
        result = pd.read_csv(destination).set_index("Feature")
        scores = result.reindex(names)["Score"].to_numpy(float)
        if not np.isfinite(scores).all():
            raise RuntimeError("PGEE adapter returned non-finite feature scores.")
        return scores, float(result["Lambda"].iloc[0]), float(result["CV"].iloc[0])


def _geeverse_scores(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    task: str,
    seed: int,
) -> tuple[np.ndarray, float, float, bool]:
    """Rank continuous outcomes with geeVerse's HBIC-tuned quantile GEE."""
    if task != "regression":
        raise ValueError("geeVerse qpgee is a registered regression-only comparator.")
    rscript = os.environ.get("CTSNN_LONGFS_RSCRIPT", "Rscript")
    adapter = Path(__file__).resolve().parents[1] / "scripts" / "rank_geeverse.R"
    if not adapter.exists():
        raise FileNotFoundError(f"Missing geeVerse adapter: {adapter}")
    names = [f"x{i:03d}" for i in range(X.shape[1])]
    with tempfile.TemporaryDirectory(prefix="ctsnn_geeverse_") as temp:
        temp_dir = Path(temp)
        source, destination = temp_dir / "input.csv", temp_dir / "scores.csv"
        frame = pd.DataFrame(_signed_log_scale(X), columns=names)
        frame.insert(0, "id", pd.factorize(np.asarray(groups).astype(str))[0] + 1)
        frame.insert(0, "y", np.asarray(y, dtype=float))
        frame.to_csv(source, index=False)
        environment = os.environ.copy()
        rscript_directory = str(Path(rscript).expanduser().resolve().parent)
        environment["PATH"] = f"{rscript_directory}{os.pathsep}{environment.get('PATH', '')}"
        completed = subprocess.run(
            [rscript, str(adapter), str(source), str(destination), task, str(seed)],
            check=False, capture_output=True, text=True, timeout=180, env=environment,
        )
        if completed.returncode != 0 or not destination.exists():
            detail = completed.stderr[-1000:] if completed.stderr else completed.stdout[-1000:]
            raise RuntimeError(f"geeVerse adapter failed: {detail}")
        result = pd.read_csv(destination).set_index("Feature")
        scores = result.reindex(names)["Score"].to_numpy(float)
        if not np.isfinite(scores).all():
            raise RuntimeError("geeVerse adapter returned non-finite feature scores.")
        return (
            scores,
            float(result["Lambda"].iloc[0]),
            float(result["HBIC"].iloc[0]),
            bool(result["Converged"].iloc[0]),
        )


def _participant_subsamples(groups: np.ndarray, fraction: float, n_bootstraps: int, seed: int):
    rng = np.random.default_rng(seed)
    unique = np.unique(groups)
    take = max(2, int(np.floor(fraction * len(unique))))
    for _ in range(n_bootstraps):
        chosen = rng.choice(unique, size=take, replace=False)
        yield np.flatnonzero(np.isin(groups, chosen))


def stability_scores(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    task: str,
    seed: int,
    n_bootstraps: int = 30,
) -> np.ndarray:
    """Participant-subsampled sparse-model selection frequencies."""
    Xs = _signed_log_scale(X)
    frequencies = np.zeros(X.shape[1])
    for rows in _participant_subsamples(groups, 0.65, n_bootstraps, seed):
        if task == "classification" and np.unique(y[rows]).size < 2:
            continue
        if task == "regression":
            model = ElasticNet(alpha=0.03, l1_ratio=0.8, max_iter=10_000, random_state=seed)
        else:
            model = LogisticRegression(
                C=0.3, penalty="l1", solver="liblinear", class_weight="balanced",
                max_iter=3_000, random_state=seed,
            )
        model.fit(Xs[rows], y[rows])
        coefficient = np.abs(model.coef_).ravel()
        frequencies += coefficient > 1e-8
    return frequencies / n_bootstraps


def stabl_random_permutation_scores(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    task: str,
    seed: int,
    n_bootstraps: int = 20,
) -> tuple[np.ndarray, float, float]:
    """Lightweight compatibility implementation of Stabl's RP idea.

    Each participant subsample is compared with column-wise permuted artificial
    features. This keeps the random-permutation null control used by Stabl-RP
    while avoiding an expensive sparse-model path inside every nested-CV split.
    """
    Xs = _signed_log_scale(X)
    rng = np.random.default_rng(seed)
    p = X.shape[1]
    frequencies = np.zeros(p)
    score_sum = np.zeros(p)
    artificial_selected = []
    subsamples = list(_participant_subsamples(groups, 0.5, n_bootstraps, seed + 19))
    valid = 0
    for rows in subsamples:
        if task == "classification" and np.unique(y[rows]).size < 2:
            continue
        artificial = Xs[rows].copy()
        for column in range(artificial.shape[1]):
            rng.shuffle(artificial[:, column])
        augmented = np.column_stack([Xs[rows], artificial])
        scores, _ = (f_regression(augmented, y[rows]) if task == "regression" else f_classif(augmented, y[rows]))
        scores = _safe_scores(scores)
        real = scores[:p]
        null = scores[p:]
        threshold = float(np.nanquantile(null[np.isfinite(null)], 0.95)) if np.isfinite(null).any() else np.inf
        selected = real > threshold
        frequencies += selected
        score_sum += real / max(threshold, 1e-12)
        artificial_selected.append(int(np.sum(null > threshold)))
        valid += 1
    if valid == 0:
        return np.zeros(p), 0.0, 1.0
    real_max = frequencies / valid + 1e-6 * score_sum / valid
    artificial_rate = np.mean(artificial_selected) / max(1, p)
    thresholds = np.arange(0.0, 1.0, 0.01)
    fdp = np.array([
        (1.0 + artificial_rate * p) / max(1, np.sum(real_max > threshold))
        for threshold in thresholds
    ])
    best = int(np.argmin(fdp))
    return real_max, float(thresholds[best]), float(fdp[best])


def _subject_trajectory_scores(X: np.ndarray, y: np.ndarray, groups: np.ndarray, times: np.ndarray, task: str) -> np.ndarray:
    """Rank features using subject means and within-subject linear slopes."""
    numeric_time = np.asarray(times, float)
    unique = np.unique(groups)
    means = np.zeros((len(unique), X.shape[1]))
    slopes = np.zeros_like(means)
    outcomes = np.zeros(len(unique))
    for row, subject in enumerate(unique):
        mask = groups == subject
        means[row] = np.nanmean(X[mask], axis=0)
        outcomes[row] = np.nanmean(y[mask])
        t = numeric_time[mask]
        if mask.sum() >= 2 and np.ptp(t) > 0:
            centered = t - t.mean()
            slopes[row] = centered @ X[mask] / max(centered @ centered, 1e-12)
    if task == "classification":
        mean_score, _ = f_classif(means, outcomes.astype(int))
        slope_score, _ = f_classif(slopes, outcomes.astype(int))
    else:
        mean_score, _ = f_regression(means, outcomes)
        slope_score, _ = f_regression(slopes, outcomes)
    return np.maximum(_safe_scores(mean_score), _safe_scores(slope_score))


def _unit_interval_scores(values: np.ndarray, floor: float = 0.0) -> np.ndarray:
    values = _safe_scores(values)
    finite = np.isfinite(values)
    if not finite.any():
        return np.ones_like(values, dtype=float)
    lo, hi = np.min(values[finite]), np.max(values[finite])
    if hi - lo <= 1e-12:
        return np.ones_like(values, dtype=float)
    scaled = np.zeros_like(values, dtype=float)
    scaled[finite] = floor + (1.0 - floor) * (values[finite] - lo) / (hi - lo)
    return scaled


def _association_scores(X: np.ndarray, y: np.ndarray, task: str) -> np.ndarray:
    scores, _ = f_regression(X, y) if task == "regression" else f_classif(X, y)
    return _safe_scores(scores)


def _boruta_shadow_scores(X: np.ndarray, y: np.ndarray, task: str, seed: int) -> np.ndarray:
    """Boruta-style all-relevant ranking using permuted shadow features.

    The published Boruta algorithm compares real-variable importance with
    permuted shadow variables.  We retain that comparison as a deterministic
    ranking so the common fixed panel budgets can be evaluated directly.
    """
    model_class = ExtraTreesRegressor if task == "regression" else ExtraTreesClassifier
    kwargs = dict(n_estimators=160, min_samples_leaf=3, max_features="sqrt", n_jobs=-1, random_state=seed)
    if task == "classification":
        kwargs["class_weight"] = "balanced"
    rng = np.random.default_rng(seed)
    shadow = X.copy()
    for column in range(shadow.shape[1]):
        rng.shuffle(shadow[:, column])
    model = model_class(**kwargs).fit(np.column_stack([X, shadow]), y)
    importance = np.asarray(model.feature_importances_, float)
    real, null = importance[:X.shape[1]], importance[X.shape[1]:]
    return real / max(float(np.max(null)), 1e-12)


def _mrmr_scores(X: np.ndarray, y: np.ndarray, task: str) -> np.ndarray:
    """Greedy mRMR relevance-minus-redundancy ranking.

    Relevance is the univariate association with the outcome and redundancy
    is the mean absolute correlation with already selected variables.  The
    resulting order is converted to a score so the benchmark can request
    panels of 3, 5, or 10 features without refitting a separate selector.
    """
    relevance = _unit_interval_scores(_association_scores(X, y, task), floor=0.0)
    if X.shape[1] == 1:
        return relevance
    scaled = StandardScaler().fit_transform(X)
    corr = np.nan_to_num(np.corrcoef(scaled, rowvar=False), nan=0.0)
    np.fill_diagonal(corr, 0.0)
    remaining = set(range(X.shape[1]))
    selected: list[int] = []
    order_scores = np.zeros(X.shape[1], dtype=float)
    for step in range(X.shape[1]):
        candidates = np.fromiter(remaining, dtype=int)
        if not selected:
            values = relevance[candidates]
        else:
            redundancy = np.mean(np.abs(corr[np.ix_(candidates, selected)]), axis=1)
            values = relevance[candidates] - redundancy
        best = int(candidates[int(np.argmax(values))])
        selected.append(best)
        remaining.remove(best)
        order_scores[best] = 1.0 - step / max(1, X.shape[1] - 1)
    return order_scores


def _knn_sets_1d(values: np.ndarray, k: int) -> list[set[int]]:
    values = np.asarray(values, float).reshape(-1, 1)
    n = len(values)
    if n <= 1:
        return [set() for _ in range(n)]
    k = int(np.clip(k, 1, n - 1))
    order = NearestNeighbors(n_neighbors=k + 1).fit(values).kneighbors(return_distance=False)
    return [set(row[row != i][:k]) for i, row in enumerate(order)]


def _temporal_snn_stability_scores(X: np.ndarray, y: np.ndarray, groups: np.ndarray, times: np.ndarray, task: str) -> np.ndarray:
    """Feature-level temporal preservation of one-dimensional SNN structure."""
    assoc = _unit_interval_scores(_association_scores(X, y, task), floor=0.0)
    numeric_time = np.asarray(times, float)
    unique_times = np.sort(np.unique(numeric_time[np.isfinite(numeric_time)]))
    stability = np.full(X.shape[1], 0.5, dtype=float)
    if len(unique_times) < 2:
        return assoc

    pair_records = []
    for left_time, right_time in zip(unique_times[:-1], unique_times[1:]):
        left = np.flatnonzero(numeric_time == left_time)
        right = np.flatnonzero(numeric_time == right_time)
        common = np.intersect1d(groups[left], groups[right])
        if len(common) < 4:
            continue
        left_lookup = {groups[i]: i for i in left}
        right_lookup = {groups[i]: i for i in right}
        left_rows = np.array([left_lookup[g] for g in common])
        right_rows = np.array([right_lookup[g] for g in common])
        pair_records.append((left_rows, right_rows))
    if not pair_records:
        return assoc

    for feature in range(X.shape[1]):
        overlaps = []
        for left_rows, right_rows in pair_records:
            m = len(left_rows)
            k = min(10, max(2, int(np.sqrt(m))))
            left_sets = _knn_sets_1d(X[left_rows, feature], k)
            right_sets = _knn_sets_1d(X[right_rows, feature], k)
            overlaps.extend(
                len(left_sets[i] & right_sets[i]) / max(k, 1)
                for i in range(m)
            )
        if overlaps:
            stability[feature] = float(np.mean(overlaps))
    return assoc * (1.0 + _unit_interval_scores(stability, floor=0.0))


def _graph_snn_smooth_scores(X: np.ndarray, y: np.ndarray, groups: np.ndarray, times: np.ndarray, task: str) -> np.ndarray:
    """Outcome association boosted when a feature is smooth on a temporal SNN graph."""
    assoc = _unit_interval_scores(_association_scores(X, y, task), floor=0.0)
    embedding = prepare_distance_embedding(X, robust=False)
    n = len(X)
    if n < 4:
        return assoc
    k = int(np.clip(int(0.3 * n), 2, min(70, n - 1)))
    indices = NearestNeighbors(n_neighbors=k + 1).fit(embedding).kneighbors(return_distance=False)
    numeric_time = np.asarray(times, float)
    span = float(np.nanmax(numeric_time) - np.nanmin(numeric_time))
    if not np.isfinite(span) or span <= 1e-12:
        temporal = np.ones(n)
        normalized_time = numeric_time
    else:
        normalized_time = (numeric_time - np.nanmin(numeric_time)) / span
        temporal = None
    roughness = np.zeros(X.shape[1], dtype=float)
    counts = 0
    for i, row in enumerate(indices):
        neighbours = row[row != i][:k]
        if temporal is None:
            weights = np.exp(-np.abs(normalized_time[neighbours] - normalized_time[i]) / 0.25)
        else:
            weights = np.ones(len(neighbours))
        if weights.sum() <= 1e-12:
            continue
        diffs = X[neighbours] - X[i]
        roughness += np.average(diffs * diffs, axis=0, weights=weights)
        counts += 1
    if counts:
        roughness /= counts
    variance = np.nanvar(X, axis=0) + 1e-12
    smoothness = 1.0 / (1.0 + roughness / variance)
    return assoc * (1.0 + _unit_interval_scores(smoothness, floor=0.0))


def _subject_summary_snn_scores(X: np.ndarray, y: np.ndarray, groups: np.ndarray, times: np.ndarray, task: str) -> np.ndarray:
    """Subject-level mean/slope association with an SNN stability boost."""
    base = _unit_interval_scores(_subject_trajectory_scores(X, y, groups, times, task), floor=0.0)
    unique = np.unique(groups)
    numeric_time = np.asarray(times, float)
    means = np.zeros((len(unique), X.shape[1]))
    slopes = np.zeros_like(means)
    outcomes = np.zeros(len(unique))
    for row, subject in enumerate(unique):
        mask = groups == subject
        means[row] = np.nanmean(X[mask], axis=0)
        outcomes[row] = np.nanmean(y[mask])
        t = numeric_time[mask]
        if mask.sum() >= 2 and np.ptp(t) > 0:
            centered = t - t.mean()
            slopes[row] = centered @ X[mask] / max(centered @ centered, 1e-12)
    summary = np.column_stack([means, slopes])
    if len(unique) < 4:
        return base
    embedding = StandardScaler().fit_transform(summary)
    local_support = snn_weights(embedding, k=min(20, max(2, len(embedding) - 2)), mutual=False)
    if task == "classification" and np.unique(outcomes.astype(int)).size >= 2:
        model = LogisticRegression(
            C=0.3, penalty="l1", solver="liblinear", class_weight="balanced",
            max_iter=3_000, random_state=2026,
        ).fit(StandardScaler().fit_transform(summary), outcomes.astype(int), sample_weight=local_support)
        weighted = np.maximum(np.abs(model.coef_).ravel()[:X.shape[1]], np.abs(model.coef_).ravel()[X.shape[1]:])
    else:
        model = ElasticNet(alpha=0.03, l1_ratio=0.8, max_iter=10_000, random_state=2026)
        model.fit(StandardScaler().fit_transform(summary), outcomes, sample_weight=local_support)
        weighted = np.maximum(np.abs(model.coef_)[:X.shape[1]], np.abs(model.coef_)[X.shape[1]:])
    return base * (1.0 + _unit_interval_scores(weighted, floor=0.0))


def _outcome_snn_weights(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Observation weights from SNN support and local outcome coherence."""
    embedding = prepare_distance_embedding(X, robust=False)
    spatial = snn_weights(embedding, mutual=False)
    n = len(X)
    if n < 4:
        return spatial
    k = int(np.clip(int(0.3 * n), 2, min(70, n - 1)))
    indices = NearestNeighbors(n_neighbors=k + 1).fit(embedding).kneighbors(return_distance=False)
    y = np.asarray(y, float)
    scale = float(np.nanstd(y)) + 1e-12
    coherence = np.zeros(n, dtype=float)
    for i, row in enumerate(indices):
        neighbours = row[row != i][:k]
        coherence[i] = np.exp(-np.mean(np.abs(y[neighbours] - y[i])) / scale)
    return _unit_interval_scores(spatial * coherence, floor=1e-4)


def _outcome_traj_snn_weights(
    X: np.ndarray,
    y: np.ndarray,
    times: np.ndarray,
    task: str,
) -> np.ndarray:
    """Outcome-aware SNN weights with a time-locality adjustment.

    The graph is built only from the molecular feature space.  Its support is
    then modulated by two training-fold quantities: local outcome coherence
    and temporal proximity.  No participant identity is used, so this remains
    valid when visits are sparse or when a participant has a single visit.
    """
    embedding = prepare_distance_embedding(X, robust=False)
    n = len(X)
    spatial = snn_weights(embedding, mutual=False)
    if n < 4:
        return _unit_interval_scores(spatial, floor=1e-4)
    k = int(np.clip(int(0.3 * n), 2, min(70, n - 1)))
    indices = NearestNeighbors(n_neighbors=k + 1).fit(embedding).kneighbors(return_distance=False)

    y = np.asarray(y, float)
    if task == "classification":
        # Local outcome agreement is the fraction of neighbours in the same
        # class; it is well-defined even when outcomes are binary labels.
        coherence = np.zeros(n, dtype=float)
        for i, row in enumerate(indices):
            neighbours = row[row != i][:k]
            coherence[i] = np.mean(y[neighbours] == y[i])
    else:
        scale = float(np.nanstd(y)) + 1e-12
        coherence = np.zeros(n, dtype=float)
        for i, row in enumerate(indices):
            neighbours = row[row != i][:k]
            coherence[i] = np.exp(-np.mean(np.abs(y[neighbours] - y[i])) / scale)

    times = np.asarray(times, float)
    span = float(np.nanmax(times) - np.nanmin(times))
    if not np.isfinite(span) or span <= 1e-12:
        temporal = np.ones(n, dtype=float)
    else:
        normalized = (times - np.nanmin(times)) / span
        temporal = np.zeros(n, dtype=float)
        for i, row in enumerate(indices):
            neighbours = row[row != i][:k]
            delta_t = np.abs(normalized[neighbours] - normalized[i])
            temporal[i] = np.mean(np.exp(-delta_t / 0.25))
    combined = spatial * _unit_interval_scores(coherence, floor=1e-4) * _unit_interval_scores(temporal, floor=1e-4)
    return _unit_interval_scores(combined, floor=1e-4)


def _rank_aggregation_scores(components: list[np.ndarray], weights: list[float]) -> np.ndarray:
    """Weighted rank aggregation where larger component scores are better."""
    if not components:
        raise ValueError("At least one component is required for rank aggregation.")
    p = len(components[0])
    combined = np.zeros(p, dtype=float)
    weight_sum = 0.0
    for values, weight in zip(components, weights):
        values = _safe_scores(values)
        # Convert each component to a comparable [0, 1] rank score. Ties are
        # resolved deterministically by feature index through _rank_desc.
        order = _rank_desc(values)
        rank_score = np.zeros(p, dtype=float)
        if p == 1:
            rank_score[order] = 1.0
        else:
            rank_score[order] = 1.0 - np.arange(p, dtype=float) / (p - 1)
        combined += float(weight) * rank_score
        weight_sum += float(weight)
    return combined / max(weight_sum, 1e-12)


def _long_snn_scores(X: np.ndarray, y: np.ndarray, groups: np.ndarray, times: np.ndarray, task: str) -> np.ndarray:
    """Stabilized longitudinal SNN feature score.

    This optimized variant keeps strong supervised association in the score
    while using SNN-derived longitudinal evidence as a stabilizer. The static
    weights avoid dataset-specific switching:

    - sparse supervised association: protects against losing simple signals;
    - subject-summary SNN: captures subject-level longitudinal profiles;
    - outcome-coherent SNN: captures local outcome-consistent neighborhoods;
    - graph smoothness: weak temporal/SNN regularization.
    """
    sparse = _sparse_scores(X, y, task, l1_ratio=0.8)
    subject = _subject_summary_snn_scores(X, y, groups, times, task)
    outcome_weights = _outcome_snn_weights(X, y)
    outcome = _sparse_scores(X, y, task, weights=outcome_weights, l1_ratio=0.8)
    graph = _graph_snn_smooth_scores(X, y, groups, times, task)
    return _rank_aggregation_scores(
        [sparse, subject, outcome, graph],
        [0.40, 0.25, 0.25, 0.10],
    )


def _multiscale_snn_scores(X: np.ndarray, y: np.ndarray, task: str) -> np.ndarray:
    """SNN weighting averaged over several graph resolutions."""
    embedding = prepare_distance_embedding(X, robust=False)
    n = len(X)
    if n < 4:
        return _sparse_scores(X, y, task, l1_ratio=0.8)
    ks = sorted(set(int(np.clip(frac * n, 2, min(70, n - 1))) for frac in (0.10, 0.20, 0.30)))
    weights = np.mean([snn_weights(embedding, k=k, mutual=False) for k in ks], axis=0)
    return _sparse_scores(X, y, weights=_unit_interval_scores(weights, floor=1e-4), task=task, l1_ratio=0.8)


def _cross_time_snn_weights(X: np.ndarray, times: np.ndarray, k: int = 10) -> np.ndarray:
    """Shared-neighbor support using only neighbors from a different visit stratum."""
    embedding = prepare_distance_embedding(X, robust=False)
    n = len(X)
    if n < 4 or len(np.unique(times)) < 2:
        return snn_weights(embedding, k=min(k, max(2, n - 1)))
    dist = NearestNeighbors(n_neighbors=n).fit(embedding)
    order = dist.kneighbors(n_neighbors=max(2, n - 1), return_distance=False)
    neighborhoods = []
    for i, row in enumerate(order):
        eligible = [j for j in row if j != i and times[j] != times[i]][:k]
        neighborhoods.append(set(eligible))
    support = np.array([
        np.mean([len(neighborhoods[i] & neighborhoods[j]) / max(k, 1) for j in neighborhoods[i]])
        if neighborhoods[i] else 0.0
        for i in range(n)
    ])
    return _unit_interval_scores(support, floor=1e-4)


def _delta_snn_scores(X: np.ndarray, y: np.ndarray, groups: np.ndarray, times: np.ndarray, task: str) -> np.ndarray:
    """SNN graph on within-participant change from the first observed visit."""
    delta = np.zeros_like(X, dtype=float)
    for subject in np.unique(groups):
        rows = np.flatnonzero(groups == subject)
        rows = rows[np.argsort(times[rows])]
        if len(rows) > 1:
            delta[rows] = X[rows] - X[rows[0]]
    embedding = prepare_distance_embedding(delta, robust=False)
    weights = snn_weights(embedding, mutual=False)
    return _sparse_scores(X, y, weights=_unit_interval_scores(weights, floor=1e-4), task=task, l1_ratio=0.8)


def _prototype_snn_scores(X: np.ndarray, y: np.ndarray, task: str) -> np.ndarray:
    """Select graph-central prototype observations before supervised ranking."""
    embedding = prepare_distance_embedding(X, robust=False)
    support = snn_weights(embedding, mutual=False)
    n = len(X)
    n_proto = max(8, int(np.ceil(0.35 * n)))
    if task == "classification" and np.unique(y).size >= 2:
        # Preserve both outcome classes while retaining the most central points.
        per_class = max(2, n_proto // 2)
        selected = []
        for cls in np.unique(y):
            cls_rows = np.flatnonzero(y == cls)
            selected.extend(cls_rows[np.argsort(support[cls_rows])[-min(per_class, len(cls_rows)):]])
        prototypes = np.asarray(selected, dtype=int)
    else:
        prototypes = np.argsort(support)[-min(n_proto, n):]
    # Keep the prototype selection unsupervised; only the final sparse model sees y.
    return _sparse_scores(X[prototypes], y[prototypes], task=task, l1_ratio=0.8)


def _consensus_snn_scores(X: np.ndarray, y: np.ndarray, groups: np.ndarray, task: str, seed: int) -> np.ndarray:
    """Bootstrap consensus of SNN support, reducing dependence on one graph."""
    embedding = prepare_distance_embedding(X, robust=False)
    rng = np.random.default_rng(seed)
    unique = np.unique(groups)
    accumulated = np.zeros(len(X), dtype=float)
    counts = np.zeros(len(X), dtype=float)
    for _ in range(12):
        chosen = rng.choice(unique, size=max(2, int(0.75 * len(unique))), replace=False)
        rows = np.flatnonzero(np.isin(groups, chosen))
        if len(rows) < 4:
            continue
        support = snn_weights(embedding[rows], mutual=False)
        accumulated[rows] += support
        counts[rows] += 1.0
    weights = accumulated / np.maximum(counts, 1.0)
    weights[counts == 0] = np.mean(weights[counts > 0]) if np.any(counts > 0) else 1.0
    return _sparse_scores(X, y, weights=_unit_interval_scores(weights, floor=1e-4), task=task, l1_ratio=0.8)


def _longitudinal_group_lasso_scores(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    task: str,
    iterations: int = 10,
) -> np.ndarray:
    """Paired between/within group-lasso path solved by proximal gradient."""
    Xs = _signed_log_scale(X)
    between = np.empty_like(Xs)
    for subject in np.unique(groups):
        mask = groups == subject
        between[mask] = Xs[mask].mean(axis=0)
    within = Xs - between
    Z = np.column_stack([between, within])
    Z = StandardScaler().fit_transform(Z)
    n, p2 = Z.shape
    p = p2 // 2
    target = y.astype(float)
    if task == "regression":
        target = target - target.mean()
        gradient_zero = np.column_stack([Z[:, :p], Z[:, p:]]).T @ target / n
        lip_factor = 1.0
    else:
        target = target.astype(float)
        gradient_zero = Z.T @ (target - target.mean()) / n
        lip_factor = 0.25
    lambda_max = max(
        np.hypot(gradient_zero[:p], gradient_zero[p:]).max(), 1e-6
    )
    lambdas = np.geomspace(lambda_max, lambda_max * 0.10, 3)
    # Power iteration avoids a full SVD of each longitudinal design.
    vector = np.ones(p2) / np.sqrt(p2)
    for _ in range(5):
        vector = Z.T @ (Z @ vector)
        vector /= max(np.linalg.norm(vector), 1e-12)
    lipschitz = lip_factor * np.linalg.norm(Z @ vector) ** 2 / n + 1e-8
    step = 1.0 / lipschitz
    beta = np.zeros(p2)
    scores = np.zeros(p)
    for penalty in lambdas:
        for _ in range(iterations):
            linear = Z @ beta
            if task == "regression":
                gradient = Z.T @ (linear - target) / n
            else:
                probability = 1.0 / (1.0 + np.exp(-np.clip(linear, -30, 30)))
                gradient = Z.T @ (probability - target) / n
            proposal = beta - step * gradient
            norm = np.hypot(proposal[:p], proposal[p:])
            shrink = np.maximum(0.0, 1.0 - step * penalty / np.maximum(norm, 1e-12))
            updated = np.r_[proposal[:p] * shrink, proposal[p:] * shrink]
            if np.linalg.norm(updated - beta) <= 1e-6 * (1 + np.linalg.norm(beta)):
                beta = updated
                break
            beta = updated
        active_norm = np.hypot(beta[:p], beta[p:])
        scores = np.maximum(scores, (penalty / lambda_max) + 1e-3 * active_norm)
        scores[active_norm <= 1e-8] = np.minimum(scores[active_norm <= 1e-8], 0.0)
    return scores


@dataclass
class RankingResult:
    ranking: np.ndarray
    scores: np.ndarray
    metadata: dict[str, float | str]


def rank_features(
    method: str,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    times: np.ndarray,
    task: str,
    seed: int,
    prefilter_limit: int = 300,
) -> RankingResult:
    """Return a full feature ranking for one training set."""
    method = canonical_method_name(method)
    original_p = X.shape[1]
    method_limit = min(prefilter_limit, 40) if method in {"LongGroupLasso", "glmmLasso", "PGEE", "geeVerse"} else prefilter_limit
    method_limit = min(method_limit, 220) if method == "Stabl-RP" else method_limit
    index, local_X = _prefilter(X, y, task, limit=method_limit)
    metadata: dict[str, float | str] = {"prefilter_features": int(len(index))}

    if method in {"Lasso", "Equal"}:
        local_scores = _sparse_scores(local_X, y, task, l1_ratio=1.0)
    elif method == "ElasticNet":
        local_scores = _sparse_scores(local_X, y, task, l1_ratio=0.5)
    elif method == "RandomForest":
        model_class = ExtraTreesRegressor if task == "regression" else ExtraTreesClassifier
        kwargs = dict(n_estimators=250, min_samples_leaf=3, max_features="sqrt", n_jobs=-1, random_state=seed)
        if task == "classification":
            kwargs["class_weight"] = "balanced"
        local_scores = model_class(**kwargs).fit(local_X, y).feature_importances_
    elif method == "MutualInfo":
        function = mutual_info_regression if task == "regression" else mutual_info_classif
        local_scores = function(local_X, y, random_state=seed)
    elif method == "Boruta":
        local_scores = _boruta_shadow_scores(local_X, y, task, seed)
    elif method == "mRMR":
        local_scores = _mrmr_scores(local_X, y, task)
    elif method == "StabilitySelection":
        local_scores = stability_scores(local_X, y, groups, task, seed)
    elif method == "Stabl-RP":
        local_scores, threshold, min_fdp = stabl_random_permutation_scores(local_X, y, groups, task, seed)
        metadata.update(stabl_threshold=threshold, stabl_min_fdp=min_fdp)
    elif method == "LongGroupLasso":
        local_scores = _longitudinal_group_lasso_scores(local_X, y, groups, task)
    elif method == "glmmLasso":
        local_scores, lambda_selected, bic_selected = _glmmlasso_scores(local_X, y, groups, task, seed)
        metadata.update(glmmLasso_lambda=lambda_selected, glmmLasso_bic=bic_selected)
    elif method == "PGEE":
        local_scores, lambda_selected, cv_selected = _pgee_scores(local_X, y, groups, task, seed)
        metadata.update(PGEE_lambda=lambda_selected, PGEE_cv=cv_selected)
    elif method == "geeVerse":
        local_scores, lambda_selected, hbic_selected, converged = _geeverse_scores(local_X, y, groups, task, seed)
        metadata.update(geeVerse_lambda=lambda_selected, geeVerse_hbic=hbic_selected, geeVerse_converged=converged)
    elif method == "TrajectoryScore":
        local_scores = _subject_trajectory_scores(local_X, y, groups, times, task)
    elif method == "TSNN-Stability":
        local_scores = _temporal_snn_stability_scores(local_X, y, groups, times, task)
    elif method == "GraphSNN-FS":
        local_scores = _graph_snn_smooth_scores(local_X, y, groups, times, task)
    elif method == "SubjectSummarySNN":
        local_scores = _subject_summary_snn_scores(local_X, y, groups, times, task)
    elif method == "OutcomeSNN-FS":
        weights = _outcome_snn_weights(local_X, y)
        local_scores = _sparse_scores(local_X, y, task, weights=weights, l1_ratio=0.8)
        metadata.update(weight_mean=float(np.mean(weights)), weight_sd=float(np.std(weights)))
    elif method == "OutcomeTrajSNN-FS":
        weights = _outcome_traj_snn_weights(local_X, y, times, task)
        local_scores = _sparse_scores(local_X, y, task, weights=weights, l1_ratio=0.8)
        metadata.update(weight_mean=float(np.mean(weights)), weight_sd=float(np.std(weights)))
    elif method == "LongSNN-FS":
        local_scores = _long_snn_scores(local_X, y, groups, times, task)
    elif method == "MultiScaleSNN-FS":
        local_scores = _multiscale_snn_scores(local_X, y, task)
    elif method == "CrossTimeSNN-FS":
        weights = _cross_time_snn_weights(local_X, times)
        local_scores = _sparse_scores(local_X, y, weights=weights, task=task, l1_ratio=0.8)
        metadata.update(weight_mean=float(np.mean(weights)), weight_sd=float(np.std(weights)))
    elif method == "DeltaSNN-FS":
        local_scores = _delta_snn_scores(local_X, y, groups, times, task)
    elif method == "PrototypeSNN-FS":
        local_scores = _prototype_snn_scores(local_X, y, task)
    elif method == "ConsensusSNN-FS":
        local_scores = _consensus_snn_scores(local_X, y, groups, task, seed)
    elif method == "LLSS":
        selector = LongitudinalLassoSelector(
            task=task, n_features=len(index), n_resamples=8,
            prefilter_features=len(index), random_state=seed,
        ).fit(local_X, y, groups)
        local_scores = selector.selection_frequency_ + 1e-6 * selector.mean_magnitude_
    elif method in {"SNN-FS", "TrajSNN"}:
        weights = ctsnn_weights(local_X, groups, method, times=times)
        local_scores = _sparse_scores(local_X, y, task, weights=weights, l1_ratio=0.8)
        metadata.update(weight_mean=float(np.mean(weights)), weight_sd=float(np.std(weights)))
    else:
        raise ValueError(f"Unknown feature-ranking method: {method}")

    scores = np.full(original_p, -np.inf)
    scores[index] = _safe_scores(local_scores)
    return RankingResult(_rank_desc(scores), scores, metadata)
