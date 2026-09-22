# Benchmark data inventory

The repeated benchmark contains 14 cohort–outcome tasks from nine longitudinal
biomedical source cohorts. A tenth cohort, Detroit PE, is reserved for locked
external validation and is not pooled with the repeated internal benchmark.

| Task | Source cohort | Outcome | Type |
|---|---|---|---|
| `PE` | Stanford PE | PE versus term control | classification |
| `PE_LOPE_MATCHED` | Stanford PE | late-onset PE versus term control | classification |
| `FPG` | Japan T2D | fasting plasma glucose | regression |
| `FPG_KARE` | Korea KARE | fasting plasma glucose | regression |
| `KARE_STATUS` | Korea KARE | longitudinal status score | regression |
| `IGG` | Korea COVID-19 (CODA) | BCR-IGG clonal diversity | regression |
| `IGM` | Korea COVID-19 (CODA) | BCR-IGM clonal diversity | regression |
| `TRA` | Korea COVID-19 (CODA) | TCR-alpha clonal diversity | regression |
| `TRB` | Korea COVID-19 (CODA) | TCR-beta clonal diversity | regression |
| `OLINK_COVID` | Olink COVID | fatal disease | classification |
| `BRIST1D` | BrisT1D | daily mean blood glucose | regression |
| `GSE41848_MS` | GSE41848 | MS versus healthy control | classification |
| `GSE41849_MS` | GSE41849 | MS versus healthy control | classification |
| `GSE48023_H1N1` | GSE48023 | H1N1 HAI response, Day 14 minus Day 0 | regression |

## Counting convention

A **task** is one cohort–outcome evaluation. Multiple outcomes from the same
source cohort are not counted as independent datasets in prose. Thus IGG, IGM,
TRA, and TRB are four tasks from one Korea COVID-19 cohort; FPG_KARE and
KARE_STATUS are two tasks from Korea KARE; and PE and PE_LOPE_MATCHED are two
tasks from Stanford PE.

The repeated benchmark therefore has:

- 14 tasks;
- 9 source cohorts;
- 5 classification tasks and 9 regression tasks;
- 50 repeated participant-level evaluations per eligible method–task pair.

Detroit PE is the tenth study cohort in the broader project. Its selector is
trained in Stanford and transferred without using Detroit to fit or tune the
feature selector. It is reported separately as locked external validation.

## Longitudinal safeguards

All observations from one participant remain in the same outer and inner split.
The number of observations is never treated as the number of independent
participants. Dataset-specific participant counts, visit counts, candidate
features, and visit-frequency summaries are provided in
`results/summary/dataset_longitudinal_audit.csv`.

## Data redistribution

The repository includes only a synthetic example. Biomedical data must be
obtained under the original providers' access and licensing terms, then mapped
to the processed schema expected by `src/run_expanded_benchmark.py`.

