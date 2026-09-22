#!/usr/bin/env Rscript

# Rank prefiltered features with the official glmmLasso random-intercept model.
# Input columns: outcome, subject, x001, x002, ...

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 4) stop("Usage: rank_glmmlasso.R INPUT.csv OUTPUT.csv TASK SEED")
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
if (task == "regression") {
  outcome_sd <- sd(d$outcome)
  if (is.finite(outcome_sd) && outcome_sd > 0) d$outcome <- (d$outcome - mean(d$outcome)) / outcome_sd
}

formula_fixed <- as.formula(paste("outcome ~", paste(features, collapse = " + ")))
family_value <- if (task == "classification") binomial(link = "logit") else gaussian()
lambda_grid <- c(0.1, 0.5, 2, 10)
fit_control <- glmmLassoControl(
  steps = 80,
  maxIter = 25,
  epsilon = 1e-3,
  print.iter = FALSE,
  print.iter.final = FALSE
)
fits <- lapply(lambda_grid, function(lambda) {
  tryCatch(
    suppressWarnings(glmmLasso(
      fix = formula_fixed,
      rnd = list(subject = ~1),
      data = d,
      lambda = lambda,
      family = family_value,
      final.re = FALSE,
      switch.NR = FALSE,
      control = fit_control
    )),
    error = function(e) NULL
  )
})
valid <- which(!vapply(fits, is.null, logical(1)))
if (!length(valid)) stop("glmmLasso failed for all candidate lambda values.")
bics <- vapply(fits[valid], function(model) as.numeric(model$bic)[1], numeric(1))
finite <- is.finite(bics)
if (!any(finite)) stop("glmmLasso returned no finite BIC values.")
valid <- valid[finite]
bics <- bics[finite]
best_position <- valid[[which.min(bics)]]
best <- fits[[best_position]]
coefficient <- coef(best)
scores <- abs(as.numeric(coefficient[features]))
scores[!is.finite(scores)] <- 0

write.csv(
  data.frame(
    Feature = features,
    Score = scores,
    Lambda = lambda_grid[[best_position]],
    BIC = as.numeric(best$bic)[1]
  ),
  output_path,
  row.names = FALSE
)
