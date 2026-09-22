# R environment used by official-package adapters

The completed official-package extensions used:

- R 4.3.3
- `geeVerse` 0.3.1
- `glmmLasso` 1.6.4
- `PGEE` 1.5

The Python benchmark calls these packages through the scripts in `scripts/`.
They are **official-package adapters**: the feature ranking is obtained from the
official R package, while participant-level splitting, panel-budget tuning, and
downstream prediction are controlled by the common Python benchmark.

The remaining labels must not be described as official reproductions unless
their implementation status in `docs/method_implementation_status.csv` says so.

