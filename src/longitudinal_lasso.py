"""Longitudinal LASSO-inspired stable feature selection.

This module implements a practical, leakage-safe adaptation of Longitudinal
LASSO (Xu et al., KDD 2015) for repeated-measurement biomarker data.  Each
candidate biomarker is represented by two components: a subject-level
(``between``) component and a visit-specific deviation (``within``) component.
Between/within scores are repeatedly estimated after resampling *subjects*, not
rows; a biomarker is ranked by its selection frequency across resamples.

The original paper models lagged temporal contingency.  The data in this
repository have irregular and short visit histories, so this implementation
uses the paper's core decomposition and structured sparsity idea without
inventing unavailable lags.  It is therefore described in the manuscript as a
"Longitudinal-LASSO-inspired stability selector", rather than a reproduction
of the original estimator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
from sklearn.feature_selection import f_classif, f_regression
from sklearn.preprocessing import StandardScaler


Task = Literal["regression", "classification"]


@dataclass
class LongitudinalLassoSelector:
    """Stable selector that respects repeated measurements during resampling."""

    task: Task
    n_features: int = 20
    n_resamples: int = 8
    subject_fraction: float = 0.75
    prefilter_features: int = 300
    random_state: int = 2026

    def _prefilter(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Keep an outcome-associated screening set before sparse fitting."""
        if X.shape[1] <= self.prefilter_features:
            return np.arange(X.shape[1])
        score, _ = (
            f_regression(X, y) if self.task == "regression" else f_classif(X, y)
        )
        score = np.nan_to_num(score, nan=-np.inf, neginf=-np.inf, posinf=np.inf)
        return np.argsort(score)[-self.prefilter_features :]

    @staticmethod
    def _longitudinal_design(X: np.ndarray, groups: np.ndarray) -> np.ndarray:
        """Return between-subject and within-subject components for each feature."""
        between = np.empty_like(X, dtype=float)
        for subject in np.unique(groups):
            mask = groups == subject
            between[mask] = X[mask].mean(axis=0)
        within = X - between
        return np.hstack((between, within))

    def fit(
        self,
        X: np.ndarray,
        y: Sequence[float],
        groups: Sequence[object],
        sample_weight: Sequence[float] | None = None,
    ):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        groups = np.asarray(groups)
        sample_weight = (
            np.ones(len(y), dtype=float)
            if sample_weight is None
            else np.asarray(sample_weight, dtype=float)
        )
        self.prefilter_idx_ = self._prefilter(X, y)
        X = X[:, self.prefilter_idx_]
        self.scaler_ = StandardScaler().fit(X)
        X = self.scaler_.transform(X)
        design = self._longitudinal_design(X, groups)

        n_groups = np.unique(groups).size
        if n_groups < 2:
            raise ValueError("At least two subjects are required for longitudinal selection.")
        rng = np.random.default_rng(self.random_state)
        unique_groups = np.unique(groups)
        counts = np.zeros(X.shape[1], dtype=float)
        magnitudes = np.zeros(X.shape[1], dtype=float)
        take = max(2, int(np.ceil(self.subject_fraction * n_groups)))

        for _ in range(self.n_resamples):
            selected_groups = rng.choice(unique_groups, size=take, replace=False)
            mask = np.isin(groups, selected_groups)
            if self.task == "classification" and np.unique(y[mask]).size < 2:
                continue
            score, _ = (
                f_regression(design[mask], y[mask])
                if self.task == "regression"
                else f_classif(design[mask], y[mask])
            )
            score = np.nan_to_num(score, nan=0.0, neginf=0.0, posinf=np.inf)
            feature_strength = np.hypot(score[: X.shape[1]], score[X.shape[1] :])
            selected = np.argsort(feature_strength)[-min(self.n_features, X.shape[1]) :]
            active = np.zeros(X.shape[1], dtype=bool)
            active[selected] = True
            counts += active
            magnitudes += feature_strength

        self.selection_frequency_ = counts / self.n_resamples
        self.mean_magnitude_ = magnitudes / self.n_resamples
        # Frequency is primary; magnitude gives deterministic tie-breaking.
        ranking = np.lexsort((self.mean_magnitude_, self.selection_frequency_))[::-1]
        local = ranking[: min(self.n_features, len(ranking))]
        self.selected_indices_ = np.sort(self.prefilter_idx_[local])
        self.selected_frequency_ = self.selection_frequency_[local]
        return self

    def get_support(self, indices: bool = False) -> np.ndarray:
        if indices:
            return self.selected_indices_
        support = np.zeros(int(self.prefilter_idx_.max()) + 1, dtype=bool)
        support[self.selected_indices_] = True
        return support
