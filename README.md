# Longitudinal Feature-Selection Benchmark

A leakage-controlled and participant-aware benchmark of feature-selection
methods for longitudinal biomedical data. The repeated benchmark compares 15
methods on 14 cohort–outcome tasks from 9 source cohorts, with a tenth cohort
reserved for locked external validation, and retains a complete audit
trail of selected variables across 50 repeated participant-level evaluations.

This repository is both a method implementation and a benchmarking framework.
Its main contribution is a common evaluation contract: grouped outer splits,
grouped inner tuning, common panel budgets, identical downstream predictors,
selection-stability analysis, cohort-level uncertainty, and auditable selected
features.

## Quick start

```bash
conda env create -f environment.yml
conda activate longfs-benchmark
python -m pip install -e ".[test]"
make smoke
```

The smoke test uses only the synthetic, non-identifiable dataset in
`data/example/minimal_longitudinal.csv` and runs Lasso, LongGroupLasso, SNN-FS,
and OutcomeSNN-FS.

## Rebuild publication outputs

Compact completed 50-repeat metrics are included. Regenerate the main tables and
figures with one command:

```bash
make publication
```

Outputs are written to `results/reproduced/`. If the full completed run archive
is available beside the project (the server layout uses `../results`), rebuild
the complete statistical package with:

```bash
make full-results
```

## Run a benchmark

The runner keeps all visits from one participant in the same split and performs
grouped inner tuning inside each outer training set.

```bash
python src/run_expanded_benchmark.py \
  --datasets PE \
  --methods Lasso LongGroupLasso SNN-FS OutcomeSNN-FS \
  --repeats 1 \
  --budgets 3 5 10 \
  --output-dir results/runs/example
```

Biomedical source datasets are not redistributed. Place authorized processed
tables under `data/processed/`; expected schemas and provenance are documented
in `docs/benchmark_datasets.md`.

## Panel-size sensitivity

The completed runs save inner-validation scores for all three candidate budgets
and outer-test results for the budget selected within each outer fold. On the
server, generate the valid retrospective analysis with:

```bash
python src/analyze_panel_sensitivity.py \
  --results-root /home/zliu/ctsnn/results \
  --output-dir results/panel_sensitivity
```

This produces budget-selection frequencies, inner-validation curves,
outer-test performance conditional on the selected budget, and conditional
Jaccard stability. A strict fixed-budget outer-test comparison cannot be
reconstructed from the existing files; submit the supplied rerun script instead:

```bash
METHOD=Lasso SLUG=lasso sbatch scripts/run_fixed_budget_sensitivity.sbatch
```

Repeat that submission for each method to obtain genuine fixed-budget 3/5/10
outer-test curves and stability estimates. The distinction is deliberate: the
repository never labels conditional or inner-validation results as fixed-budget
test performance.

## Methods and implementation status

The 15 benchmark labels are separated into five implementation classes:

- **Official package adapters:** geeVerse, glmmLasso, PGEE.
- **Library implementations:** Lasso, ElasticNet, RandomForest benchmark label
  (implemented with ExtraTrees), MutualInfo.
- **Compatible reimplementations:** StabilitySelection, Stabl-RP, Boruta, mRMR.
- **Project adaptations:** LongGroupLasso, LLSS.
- **Proposed methods:** SNN-FS, OutcomeSNN-FS.

See `docs/method_implementation_status.csv` for exact qualifications. In
particular, compatible reimplementations are not claimed to be byte-identical
runs of official software.

## Computational efficiency

File timestamps are not used as runtime estimates. The completed official R
adapters retain real Slurm accounting (elapsed time, CPU time, and peak RSS),
reported as a retrospective resource audit because job granularity differed
between adapters. The original Python benchmark was not timed per method, so a
complete controlled 15-method speed comparison requires a prospective rerun.
See `docs/computational_efficiency.md`.

## Reproducibility and data releases

- Python versions are pinned in `environment.yml` and `requirements.txt`.
- R versions are recorded in `docs/r_environment.md`.
- Compact publication results are under `results/summary/`.
- The compressed complete selected-feature audit should be attached to the
  GitHub release and archived with the data/results bundle on Zenodo.
- After creating a GitHub release, archive it on Zenodo, obtain the DOI, and
  replace the placeholder repository URL in `pyproject.toml` and `CITATION.cff`.

## Repository layout

```text
data/example/      synthetic smoke-test data
docs/              framework, datasets, methods, and implementation audit
examples/          minimal runnable example
scripts/           Slurm and official-R-package adapters
src/               selectors, evaluation pipeline, and analyses
tests/             automated smoke test
results/summary/   compact completed publication results
```

## License

MIT. Dataset licenses and access conditions remain those of the original data
providers and are not superseded by this software license.

