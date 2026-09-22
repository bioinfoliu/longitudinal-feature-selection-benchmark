# Project inventory

## Canonical code

- `src/ctsnn.py`: SNN, STAR, Hybrid, and CTSNN-R observation weights.
- `src/feature_selectors.py`: Lasso, Stabl-RP, longitudinal group Lasso, LLSS,
  trajectory, and CTSNN-family rankings.
- `src/prepare_figshare_pe_multicohort.py`: immutable workbook to cohort tables;
  assay identity is matched by SomaId.
- `src/run_expanded_benchmark.py`: repeated nested participant-level evaluation.
- `src/run_multicohort_external_validation.py`: locked bidirectional transfer.
- `src/summarize_expanded_results.py`, `src/summarize_multicohort_external.py`,
  and `src/summarize_secondary_longitudinal.py`: participant-level inference.
- `src/run_simulation_study.py`: ground-truth stress tests.
- `src/make_publication_figures.py`: deterministic manuscript figures.

## Canonical data

- `data/processed/PE_final.csv`: supplied Stanford matrix, verified against the
  public registry.
- `data/external/raw/figshare_7962998_registry.xlsx`: immutable CC BY 4.0 source.
- `data/external/processed/Stanford_SomaLogic_longitudinal_PE.csv` and
  `Detroit_SomaLogic_longitudinal_PE.csv`: derived visit-level matrices.
- `data/external/processed/Stanford_clinical.csv` and `Detroit_clinical.csv`:
  derived participant covariates.
- `data/external/processed/figshare_multicohort_report.json`: counts, checksum,
  license, and 1,116-assay overlap.
- `data/processed/FPG_final.csv`, `IGG_final.csv`, `IGM_final.csv`,
  `TRA_final.csv`, and `TRB_final.csv`: secondary longitudinal benchmarks.

## Canonical results

- `results/main_benchmark/benchmark_full50/`: primary 50-repeat benchmark.
- `results/validation/external_validation/`: transfer metrics, locked panels,
  subject predictions, source-only tuning, and paired bootstrap comparisons.
- `results/main_benchmark/benchmark_with_new/`: benchmark including new datasets.
- `results/main_benchmark/final_benchmark/`: merged 11-task package, formal
  statistics, data-structure audit, selected-feature audit, and final figures.
- `results/new_datasets/`: Olink and BrisT1D benchmark outputs.
- `results/archive/`: older exploratory outputs retained for provenance.

Every benchmark directory retains fold metrics, selected features, OOF or
target predictions, inner tuning, and uncertainty summaries. Values reported in
the manuscript are traceable to these tables.

## Supplemental and historical material

- `results/main_benchmark/benchmark_with_new/`: the wider benchmark including
  the newly added datasets.
- `docs/archive/invalidated_275_feature_external_validation/`: invalidated
  string-mapped external analysis and its source artifacts.
- `src/legacy/`, `notebooks/legacy/`, and `results/legacy/`: noncanonical
  historical implementations and outputs.

Files under an archive or legacy directory must not be cited as current results.
