# Longitudinal feature-selection benchmark methods

The final benchmark compares 13 published/general-purpose or
longitudinal-aware method families with two project SNN-based variants.
Methods are evaluated under a common participant-level nested design; repeated
visits from one participant never cross an outer or inner split boundary.

| Benchmark label | Method family | Implementation used | Longitudinal handling |
|---|---|---|---|
| `Lasso` | L1-penalized regression/classification | scikit-learn | Equal visit weights; participant-level splitting |
| `ElasticNet` | Elastic net | scikit-learn | Equal visit weights; participant-level splitting |
| `StabilitySelection` | Stability selection | compatible project implementation | Participant-subsampled resampling |
| `Stabl-RP` | Stabl random-permutation selection | compatible project implementation | Participant-subsampled resampling |
| `LongGroupLasso` | Grouped between/within effects | project proximal-gradient implementation | Repeated visits represented by grouped effects |
| `LLSS` | Longitudinal-LASSO-inspired stable selection | project implementation | Participant-level resampling |
| `RandomForest` | Tree-ensemble importance | scikit-learn ExtraTrees under benchmark label | Equal visit weights; participant-level splitting |
| `MutualInfo` | Mutual-information filter | scikit-learn | Outcome association only |
| `Boruta` | Shadow-feature all-relevant selection | compatible project implementation | Outcome-aware shadow comparison |
| `mRMR` | Minimum-redundancy maximum-relevance | compatible project implementation | Outcome relevance minus redundancy |
| `geeVerse` | Penalized quantile GEE | official CRAN package adapter | Correlated-observation GEE; regression only |
| `glmmLasso` | L1-penalized generalized linear mixed model | official R package adapter | Random-intercept clustered outcome model |
| `PGEE` | SCAD-penalized GEE | official R package adapter | Participant-clustered GEE |
| `SNN-FS` | Shared-nearest-neighbor weighting | project method | Local shared-neighbor support before ranking |
| `OutcomeSNN-FS` | Outcome-coherent SNN weighting | project method | SNN support modulated by local outcome coherence |

`SNN-FS` and `OutcomeSNN-FS` are the two project methods. Historical CTSNN,
STAR, Hybrid, and CTSNN-R names describe earlier exploratory analyses and are
not used as final benchmark methods.

## Key references

- Tibshirani (1996), Lasso; Zou and Hastie (2005), Elastic Net.
- Meinshausen and Bühlmann (2010), stability selection; Hédou et al. (2024), Stabl.
- Yuan and Lin (2006), grouped lasso; Xu, Sun, and Bi (2015), Longitudinal LASSO.
- Peng, Long, and Ding (2005), mRMR; Kursa and Rudnicki (2010), Boruta.
- Wang, Zhou, and Qu (2012), penalized GEE; Groll and Tutz (2014), glmmLasso.
