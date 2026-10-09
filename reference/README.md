# Reference output

`quick/` was generated from `configs/quick.yaml`. It is a deterministic smoke-test
run that covers every supported model; its short horizon and small Monte Carlo
sample are not intended for scientific comparison. The full thesis settings are in
`configs/thesis.yaml` and should be run before replacing these files with release
figures.

Replicate files are ignored by Git. The committed material consists of the resolved
manifest, compact aggregates, summary tables, and figures regenerated with the
observation standard deviations recorded in the configuration.

`benchmark/` contains a separate 30-record paired parameter-estimation
comparison at horizon 250. Its results, configuration, methods, and limitations
are documented in [`docs/benchmark.md`](../docs/benchmark.md). It is not the
source of the longer thesis illustrations in the main README.

`comparison/` archives the completed checkpoint study: 40 independent records
per observation model at horizons 250, 500, and 1000. It contains the original
manifest, estimate/cost/summary/sensitivity tables, and all 120 compact record
JSON files, including saved score and likelihood surfaces. Raw observation
arrays are not stored; the manifest and recorded seeds define their simulation.
This study is separate from both `benchmark/` and the thesis illustrations.

The [audited report](comparison/analysis/report.md), figures (PNG, PDF, SVG),
and input checksums are in `comparison/analysis/`. The analysis independently
checks saved estimates, pairing, accuracy metrics, and bootstrap intervals.
From the repository root with the package installed, regenerate it without
running any filters:

```bash
python scripts/analyze_comparison.py --input reference/comparison
```

The [study design](../docs/comparison.md) documents the update schedules,
numerical checks, and commands for a new run. The archived results support the
comparison on the main README.
