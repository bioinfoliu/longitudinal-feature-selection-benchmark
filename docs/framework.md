# Benchmarking framework

This repository treats the benchmarking pipeline itself as a methodological
contribution: a leakage-controlled and participant-aware framework for
longitudinal biomedical feature selection.

## Common evaluation contract

1. **Grouped outer split.** All visits from one participant remain together.
   The outer test participants are used only once for final evaluation.
2. **Grouped inner tuning.** Panel size and downstream regularization are tuned
   using participant-grouped inner folds inside the outer training data.
3. **Common panel budgets.** Methods are compared at the same candidate panel
   sizes (3, 5, and 10), preventing a method from winning simply by retaining
   more variables.
4. **Identical downstream predictors.** Balanced L2 logistic regression is used
   for classification and ridge regression for continuous outcomes.
5. **Multiple evidence axes.** The framework reports predictive performance,
   selection stability, paired-repeat ranks, cohort-level uncertainty, and
   computational-resource evidence when it is prospectively available.
6. **Complete audit trail.** Every outer split stores the selected variables,
   their ranks, the chosen budget, tuning scores, predictions, and metrics.

No feature-selection method is fitted on outer-test participants. Preprocessing,
feature ranking, budget selection, and predictor tuning are fitted using outer
training data only.

