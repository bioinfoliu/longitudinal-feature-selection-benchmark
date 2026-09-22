# Invalidated 275-feature external analysis

This directory preserves the obsolete external-validation outputs for audit
only. They must not be cited as current results.

The old pipeline parsed display labels into gene symbols and described the
Stanford discovery matrix as Olink. Inspection of the public registry proved
that `PE_final.csv` is the Stanford SomaLogic matrix and that Detroit is also
SomaLogic. Direct matching by the original SomaId identifiers yields 1,116
shared assays, not 275. The canonical replacement is:

- preparation: `src/prepare_figshare_pe_multicohort.py`
- validation: `src/run_multicohort_external_validation.py`
- results: `results/multicohort_external_validation/`

The historical scripts are retained under `src/legacy/` to document how the
invalidated result was produced.
