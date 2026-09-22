# Final benchmark interpretation

## Scope

The primary evidence comprises 14 cohort–outcome tasks from 9 source cohorts, evaluated with 15 feature-selection methods and 50 repeated subject-level outer splits per task.
The locked Stanford-to-Detroit transfer analysis is retained separately because it does not share the repeated internal-split design.

## Main finding
The task-level Friedman test for the 14 methods available across all 14 tasks gave p = 9.75e-05, indicating an overall difference in average ranks. However, the winner changed across tasks and source cohorts, so this does not establish a universally dominant method.
The supported conclusion is task-dependent performance rather than universal superiority; a non-winning method on average may still be best for a particular data structure.

## Best method by task

| Task | Outcome type | Best method | Mean | SD |
|---|---|---:|---:|---:|
| GSE41848_MS | classification (AUROC) | Boruta | 0.8394 | 0.0297 |
| GSE41849_MS | classification (AUROC) | LLSS | 0.8014 | 0.0333 |
| OLINK_COVID | classification (AUROC) | LLSS | 0.8874 | 0.0439 |
| PE | classification (AUROC) | LongGroupLasso | 0.8127 | 0.0622 |
| PE_LOPE_MATCHED | classification (AUROC) | LLSS | 0.7113 | 0.0680 |
| BRIST1D | regression (RMSE) | PGEE | 1.7494 | 0.0511 |
| FPG | regression (RMSE) | Stabl-RP | 27.5791 | 0.1402 |
| FPG_KARE | regression (RMSE) | PGEE | 11.5504 | 0.0711 |
| GSE48023_H1N1 | regression (RMSE) | geeVerse | 1.3572 | 0.0291 |
| IGG | regression (RMSE) | OutcomeSNN-FS | 0.7621 | 0.0100 |
| IGM | regression (RMSE) | glmmLasso | 1.0449 | 0.0149 |
| KARE_STATUS | regression (RMSE) | ElasticNet | 0.7238 | 0.0038 |
| TRA | regression (RMSE) | Stabl-RP | 1.0460 | 0.0109 |
| TRB | regression (RMSE) | glmmLasso | 1.1959 | 0.0170 |

## Interpretation guardrails

- Repeat-level Friedman and Wilcoxon tests characterize paired split-to-split differences; repeated splits are not fully independent biological replications.
- The structural-stratum table is descriptive only. With 14 tasks, it supports hypothesis generation rather than causal statements about why a method performs best.
- BrisT1D has only 20 participants despite many daily observations; it should be interpreted as a dense-within-subject stress test, not as broad population validation.
- SNN-FS and OutcomeSNN-FS are retained as proposed comparators. The results do not support claiming that either is uniformly superior.
