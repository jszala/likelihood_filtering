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
