#!/usr/bin/env Rscript

# Rank prefiltered features with the official glmmLasso random-intercept model.
# Input CSV columns: outcome, subject, x001, x002, ...
# Output CSV columns: Feature, Score, Lambda, BIC.

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 4) {
  stop("Usage: rank_glmmlasso.R INPUT.csv OUTPUT.csv TASK{classification|regression} SEED")
}

input_path <- args[[1]]
output_path <- args[[2]]
task <- args[[3]]
seed <- as.integer(args[[4]])

suppressPackageStartupMessages(library(glmmLasso))
set.seed(seed)
d <- read.csv(input_path, check.names = FALSE)
if (!all(c("outcome", "subject") %in% names(d))) stop("Input must contain outcome and subject columns.")
features <- setdiff(names(d), c("outcome", "subject"))
if (!length(features)) stop("Input contains no feature columns.")
d$subject <- factor(d$subject)

formula_fixed <- as.formula(paste("outcome ~", paste(features, collapse = " + ")))
family_value <- if (task == "classification") binomial(link = "logit") else gaussian()

# Variables are standardized before this adapter is called. The small fixed grid
# is selected by the model's BIC within the outer-training set.  The package
# defaults (1,000 line-search steps and 200 EM iterations per lambda) are not
# feasible inside repeated nested resampling.  These convergence limits are
# therefore part of the registered benchmark wrapper, rather than an
# undocumented runtime shortcut.
lambda_grid <- c(1, 3, 10, 30)
fit_control <- glmmLassoControl(
  steps = 150,
  maxIter = 35,
  epsilon = 1e-3,
  print.iter = FALSE,
  print.iter.final = FALSE
)
fits <- lapply(lambda_grid, function(lambda) {
  tryCatch(
    glmmLasso(
      fix = formula_fixed,
      rnd = list(subject = ~1),
      data = d,
      lambda = lambda,
      family = family_value,
      final.re = FALSE,
      switch.NR = FALSE,
      control = fit_control
    ),
    error = function(e) NULL
  )
})
valid <- which(!vapply(fits, is.null, logical(1)))
if (!length(valid)) stop("glmmLasso failed for all candidate lambda values.")
bics <- vapply(fits[valid], function(model) as.numeric(model$bic)[1], numeric(1))
best_position <- valid[[which.min(bics)]]
best <- fits[[best_position]]
coefficient <- coef(best)
scores <- abs(as.numeric(coefficient[features]))
scores[!is.finite(scores)] <- 0

write.csv(
  data.frame(Feature = features, Score = scores, Lambda = lambda_grid[[best_position]], BIC = as.numeric(best$bic)[1]),
  output_path,
  row.names = FALSE
)
