# Benchmark data inventory

The current local inventory supports nine reproducible cohort–outcome tasks.
We count the locked Detroit PE cohort as the tenth external task because it is
not used to fit the feature selector; it is evaluated with panels discovered
in Stanford. This distinction is kept explicit in the analysis rather than
calling two cohorts one pooled dataset.

| Task | Source | Outcome | Longitudinal structure | Role |
|---|---|---|---|---|
| `PE` | Stanford clinical/protein table | PE versus term control | three gestational visits | internal PE task |
| `PE_LOPE_MATCHED` | Stanford SomaLogic table | late-onset PE | up to two visits; 1,116 matched SomaId assays | primary matched-assay task |
| `FPG` | Japan T2D Figshare table | fasting plasma glucose | 0M, 1M, 3M, 12M | metabolic regression |
| `FPG_KARE` | Korean Genome and Epidemiology Study-derived table | fasting plasma glucose | three visits | independent metabolic regression |
| `KARE_STATUS` | Korean Genome and Epidemiology Study-derived table | ordinal status code | three visits | secondary ordinal/continuous benchmark |
| `IGG` | CODA COVID-19 protein/TCR table | BCR-IGG clonal diversity | admission-to-discharge visits | immune regression |
| `IGM` | CODA COVID-19 protein/TCR table | BCR-IGM clonal diversity | admission-to-discharge visits | immune regression |
| `TRA` | CODA COVID-19 protein/TCR table | TCR-α clonal diversity | admission-to-discharge visits | immune regression |
| `TRB` | CODA COVID-19 protein/TCR table | TCR-β clonal diversity | admission-to-discharge visits | immune regression |
| `PE_DETROIT_EXTERNAL` | Detroit SomaLogic table | late-onset PE | up to two visits; matched assay panel | locked external task |

The first nine tasks are run with subject-level outer splits. The Detroit task
uses a locked external protocol: feature ranking is learned in Stanford only,
then the selected assays are transferred to Detroit without refitting the
selector on Detroit.

## Added public longitudinal tasks

| Task | Source | Outcome | Longitudinal structure | Role |
|---|---|---|---|---|
| `OLINK_COVID` | Olink COVID proteomics | fatal disease | 52 participants, 246 samples, up to 10 visits, 436 protein candidates | classification benchmark |
| `BRIST1D` | BrisT1D wearable/device data | daily mean blood glucose | 20 participants, 3,655 participant-days, 12 device-derived candidates | dense longitudinal regression stress test |

`OLINK_COVID` derives from the public plasma subcohort released with Gisby et
al., *Longitudinal proteomic profiling of dialysis patients with COVID-19
reveals markers of severity and predictors of death*, *eLife* (2021), DOI:
10.7554/eLife.64827 (CC BY 4.0). `BRIST1D` uses the open processed-device-data
component of the BrisT1D dataset, DOI: 10.5523/bris.33z5jc8fa6tob21ptrugzqog08
(CC BY 4.0). For BrisT1D, the large number of daily rows must not be confused
with a large number of independent participants; the analysis retains
participant-level splitting and reports the participant count explicitly.

## Counting convention

The final repeated benchmark has 11 cohort--outcome tasks from 6 source
cohorts. IGG, IGM, TRA, and TRB are four outcomes from the same Korea COVID-19
source cohort; FPG and KARE status are two outcomes from KARE; and PE and
PE_LOPE_MATCHED are two outcome definitions from Stanford PE. The locked
Detroit transfer is reported separately, rather than pooled with the repeated
internal-split tasks.

Potential future public additions are OASIS-2 longitudinal MRI and MIRIAD
longitudinal MRI, but they are not silently mixed into the protein/clinical
benchmark because they require a separate imaging feature-extraction and
outcome definition.
