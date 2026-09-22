# Version 1.0.0

Initial reproducible release of the longitudinal biomedical feature-selection
benchmark.

## Included

- installable Python package and pinned Python environment;
- exact R and official-package versions for geeVerse, glmmLasso, and PGEE;
- synthetic longitudinal example data and one-command smoke test;
- leakage-controlled participant-level nested benchmark pipeline;
- 15-method implementation-status audit;
- compact completed 50-repeat results for 14 tasks;
- compressed complete selected-feature audit trail;
- publication table and figure regeneration command;
- panel-size sensitivity summaries with explicit evidence-level labels;
- retrospective Slurm resource audit with comparability qualifications;
- strict fixed-budget rerun script for prospective sensitivity analysis.

## Before publishing the release

1. Replace `USERNAME` in `pyproject.toml` and `CITATION.cff` with the GitHub
   account or organization name.
2. Upload this folder to a public GitHub repository.
3. Create a GitHub release tagged `v1.0.0` and attach the release ZIP if desired.
4. Connect the repository to Zenodo and archive the GitHub release.
5. Add the resulting DOI to `CITATION.cff`, the README, and the manuscript.

Raw biomedical data are not redistributed. Users must obtain them under the
original providers' licenses and access conditions.

