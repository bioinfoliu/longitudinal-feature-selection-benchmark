# Final benchmark results

## Scope

The final benchmark contains 15 feature-selection methods evaluated on 14
cohort–outcome tasks from nine longitudinal biomedical source cohorts. Every
task used 50 repeated participant-level outer splits, with grouped inner
tuning and common candidate panel sizes of 3, 5, and 10 features.

## Overall rank results

Mean paired-repeat rank is first computed within each task and then averaged
over tasks (lower is better):

| Method | Mean rank | Rank SD | Task win rate |
|---|---:|---:|---:|
| glmmLasso | 4.77 | 1.47 | 16.1% |
| PGEE | 5.52 | 2.25 | 14.7% |
| LongGroupLasso | 6.36 | 1.70 | 5.6% |
| LLSS | 6.99 | 3.20 | 7.0% |
| Stabl-RP | 7.17 | 3.17 | 10.9% |

The all-task Friedman test used the 14 methods available on all 14 tasks and
showed overall rank heterogeneity (chi-square = 40.94, P = 9.75e-05).
Classification alone did not show a significant omnibus difference (five
tasks; P = 0.147). Regression did (15 methods across nine tasks; chi-square =
33.02, P = 0.00286).

## Task winners

- Classification: LLSS led `GSE41849_MS`, `OLINK_COVID`, and
  `PE_LOPE_MATCHED`; Boruta led `GSE41848_MS`; LongGroupLasso led `PE`.
- Regression: PGEE led `BRIST1D` and `FPG_KARE`; Stabl-RP led `FPG` and `TRA`;
  geeVerse led `GSE48023_H1N1`; OutcomeSNN-FS led `IGG`; glmmLasso led `IGM`
  and `TRB`; ElasticNet led `KARE_STATUS`.

## Interpretation

The results support task-dependent method selection, not universal superiority.
`glmmLasso` has the best average predictive rank, while `LongGroupLasso` has
the strongest average selected-panel stability. `SNN-FS` and `OutcomeSNN-FS`
remain project comparators with task-specific strengths, not universal winners.

Compact reproducible outputs are in `results/summary/`. The compressed complete
selected-feature audit is `results/summary/selected_features_all_tasks.csv.gz`.
