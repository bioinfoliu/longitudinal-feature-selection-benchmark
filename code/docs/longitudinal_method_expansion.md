# Longitudinal-aware method expansion protocol

## Why the comparator set is being expanded

The initial benchmark mixed generic feature selectors with a small number of
longitudinal-structured comparators. That is useful as a baseline but is not
sufficient for a claim about the comparative performance of methods designed
for repeated-measure data. The expanded benchmark retains generic methods and
adds published methods that explicitly model subject-level correlation or
temporal structure.

## Methods to add

| Method | Published implementation | Design handled | Eligible outcomes | Benchmark status |
|---|---|---|---|---|
| `PGEE` | Wang, Zhou and Qu; archived CRAN `PGEE` | SCAD-penalized GEE for high-dimensional longitudinal data | Gaussian, binary | official archive package installed; synthetic Gaussian/binary adapter tests passed; FPG nested smoke running |
| `geeVerse` | CRAN `geeVerse` | computationally efficient penalized quantile GEE for high-/ultra-high-dimensional longitudinal data | continuous outcomes; regression-only adapter | official package installed; synthetic tests and FPG 1-repeat/5-fold nested smoke passed |
| `glmmLasso` | Groll and Tutz; CRAN `glmmLasso` | random-intercept GLMM with L1 fixed-effect selection | binary, Gaussian | official package and synthetic smoke test passed; first FPG nested real-data smoke exceeded the per-fold viability window, so it is not yet admitted |
| `glmmPen` | Rashid et al.; CRAN `glmmPen` | high-dimensional penalized GLMM; fixed and random effect selection | binary, Gaussian | installation pending |
| `TemporalForest` | Shao, Moore and Ramirez; CRAN/GitHub `TemporalForest` | network-guided temporal feature selection | aligned repeated-time designs | software comparator only: its package citation currently describes the method manuscript as submitted, so it is not counted as a published-method comparator |

`LongGroupLasso` and `LLSS` remain in the benchmark, but their implementation
status will be described precisely. `LLSS` is a longitudinal-LASSO-inspired
stability selector and must not be presented as the official implementation of
Xu et al.'s Longitudinal LASSO.

The initial full longitudinal comparator set will therefore include both
marginal-correlation methods (`PGEE`, `geeVerse`) and conditional mixed-model
methods (`glmmLasso`, `glmmPen`) where they pass the same real-data feasibility
check.

## Methods intentionally not pooled into the FS comparison

- **BiMM forest** is a longitudinal binary prediction model. Its current public
  `bimm` package does not include variable selection; it can be a secondary
  end-to-end PE prediction comparator but is not a like-for-like feature
  selector.
- **timeOmics** is a longitudinal multi-omics temporal-signature and clustering
  framework, not a supervised outcome-prediction feature selector. It should
  not be forced into the same AUROC/RMSE benchmark.
- **GEE-TGDR** and the original **Longitudinal LASSO** are methodologically
  relevant, but no maintained official implementation has yet been verified.
  They will be cited as screened methods, not silently recreated and called
  official software.

## Fair evaluation protocol

1. Participant-level outer splits remain unchanged.
2. All imputation, screening, method-specific tuning, and feature ranking occur
   only in an outer-training set.
3. The candidate panel sizes remain 3, 5, and 10 for selectors that return a
   ranking; the selected budget is determined inside outer training data.
4. `TemporalForest` is evaluated only where a common usable visit grid can be
   constructed without outcome leakage. Its coverage is reported, rather than
   treating non-applicability as poor performance.
5. A method is not included in the final 50-repeat benchmark until it passes a
   synthetic Gaussian and/or binary random-intercept smoke test and produces a
   deterministic, machine-readable selected-feature table.
6. For `glmmLasso`, the wrapper uses the package's L1-penalized random-intercept
   fit with \(\lambda \in \{1,3,10,30\}\) selected by BIC. To make repeated
   nested evaluation feasible, its registered controls are `steps = 150`,
   `maxIter = 35`, and `epsilon = 1e-3`; these settings will be reported with
   the benchmark and applied identically in every outer-training fit.
   Its initial FPG nested smoke still did not finish one inner fold in 162
   seconds, so it is currently a screened but non-admitted comparator rather
   than a scored method.

## References

- Groll A, Tutz G. Variable selection for generalized linear mixed models by
  L1-penalized estimation. *Statistics and Computing*. 2014.
- Wang L, Zhou J, Qu A. Penalized generalized estimating equations for
  high-dimensional longitudinal data analysis. *Biometrics*. 2012.
- Rashid N, et al. High dimensional penalized generalized linear mixed models.
  *Journal of the American Statistical Association*. 2020.
- Shao S, Moore JH, Ramirez C. Network-guided temporal forests for feature
  selection in high-dimensional longitudinal data. `TemporalForest` package.
