# Computational-efficiency evidence

The original Python benchmark was not prospectively instrumented per method.
File timestamps are therefore **not** used as runtime estimates: they confound
queue delay, parallel execution, aggregation, and file copying.

Real Slurm accounting was retained for the completed official-package adapters
(`geeVerse`, `glmmLasso`, and `PGEE`). The repository reports that evidence as a
retrospective resource audit, including job wall time, CPU time, peak resident
memory, allocated CPUs, and requested memory. The jobs had different units of
work, so their raw wall times are not a controlled head-to-head speed test.

For future runs, invoke the benchmark through the supplied Slurm scripts and
archive `sacct` output. A fully controlled efficiency comparison should use the
same task, split, panel budget, CPU allocation, and software environment for all
methods.

