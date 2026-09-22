# longitudinal feature-selection benchmark — release package

This package is a curated, portable snapshot of the final benchmark analysis.

Contents:

- `code/`: Python and R source code, Slurm scripts, dependency files, and current documentation.
- `results/final_benchmark/`: 15-method, 14-task final summary tables, repeat-level metrics, selected-feature records, statistical tests, and classification/regression figures.
- `manuscript/`: the current Word manuscript.

Raw and processed study data are deliberately not bundled because of size and data-access restrictions. Dataset provenance and preparation code are documented in `code/docs/` and `code/src/`.

To recreate final summary outputs after restoring the required data and completed run directories:

```bash
cd code
MPLBACKEND=Agg python3 src/build_final_benchmark.py
```

The final analysis uses 50 repeated participant-level outer splits, 3-fold grouped inner tuning, and candidate panels of 3, 5, and 10 features. It compares 15 methods across 14 cohort–outcome tasks.
