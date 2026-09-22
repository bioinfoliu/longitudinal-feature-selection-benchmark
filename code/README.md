# CTSNN longitudinal feature-selection benchmark

This repository is the reproducible code and evidence package for a benchmark
of feature selection in longitudinal biomedical data. The current analysis
compares **15 methods** across **14 cohort–outcome tasks** from **nine source
cohorts**, covering pregnancy proteomics, type 2 diabetes, COVID-19 immune
profiling, multiple sclerosis, influenza vaccination, and type 1 diabetes.

The benchmark includes 13 published/general-purpose or longitudinal-aware
selectors plus two project methods: `SNN-FS` and `OutcomeSNN-FS`. Historical
CTSNN, STAR, Hybrid, and CTSNN-R experiments are retained only as archived
provenance; they are not part of the final benchmark claim.

## Final evaluation design

- 50 repeated participant-level outer train/test splits per task.
- Three participant-level inner folds for selection of panel size and predictor
  regularization.
- Shared candidate panel sizes: 3, 5, and 10 features.
- Downstream models: balanced L2-penalized logistic regression for
  classification and ridge regression for continuous outcomes.
- No participant appears in both training and test data. Transformations,
  prefiltering, selection, and tuning are training-fold operations.

## Main result

No method is universally best. Using mean paired-repeat ranks across tasks,
`glmmLasso` had the best overall mean rank (4.77), followed by `PGEE` (5.52)
and `LongGroupLasso` (6.36). The overall Friedman test showed rank
heterogeneity (14 methods common to all 14 tasks; chi-square 40.94,
P < 0.001), but task winners switched substantially. The classification-only
comparison was not significant (P = 0.147); the regression-only comparison was
significant (15 methods, nine tasks; chi-square 33.02, P = 0.003).

The reproducible interpretation, selected features, repeat-level metrics,
method ranks, statistical tests, and main figures are written to:

`/home/zliu/ctsnn/results/main_benchmark/final_benchmark/`

## Rebuild final results

The completed 50-repeat runs must be present. Rebuild the combined results and
figures without refitting models:

```bash
cd /home/zliu/ctsnn/project
MPLBACKEND=Agg python3 src/build_final_benchmark.py
```

This integrates the standard benchmark runs and completed official-package
extensions for `glmmLasso` and `PGEE`.

## Project layout

- `src/build_final_benchmark.py`: canonical result integration, task ranks,
  Friedman/Wilcoxon summaries, selected-feature frequency tables, and main plots.
- `src/feature_selectors.py`, `src/ctsnn.py`: project SNN-based selectors.
- `scripts/rank_glmmlasso.R`, `scripts/rank_pgee.R`: official R-package adapters.
- `results/main_benchmark/final_benchmark/`: final 15-method evidence package.
- `docs/benchmark_methods.md`: method provenance and longitudinal handling.
- `docs/results_summary.md`: final results and interpretation guardrails.

## Scope

This is a benchmark study, not a claim that a particular biomarker panel is
clinically validated. The locked Stanford-to-Detroit PE transfer analysis is
kept separately and is not pooled with the repeated internal benchmark.
