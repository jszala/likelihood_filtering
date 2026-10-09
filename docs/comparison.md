# Checkpoint comparison

This experiment compares parameter estimates from identical synthetic OU
observation records at times 250, 500, and 1000. The full profile has 40
independent records per observation model, step 0.01, and true parameter 0.37.
It uses a new master seed, 20261008. The older T=250 benchmark remains a
separate experiment; its results are not pooled with this one.

## What each method computes

| Method | State representation | Parameter candidates | Updates and reported estimates |
| --- | --- | --- | --- |
| Augmented-EnKF score root | 75 augmented state/score/information members per candidate for linear Gaussian; 150 for the other models | 19 fixed candidates; interpolate a score root between candidates | Filter and score update at every observation; continuation defines an estimate at every step. The runner computes candidate paths in batches and saves estimates at the three checkpoints. |
| Kalman likelihood | Scalar Gaussian mean and variance; no state particles | Numerical optimization over [0.1, 1.0] | Each likelihood evaluation filters observations sequentially. At each checkpoint, optimization reruns the filter on that observation prefix. Parameter optimization runs at checkpoints, rather than at every step. |
| Bootstrap particle likelihood | 8,192 hidden-state particles **per candidate**, with adaptive systematic resampling | 19 fixed candidates, spacing 0.05; choose the largest accumulated likelihood | State distributions and cumulative likelihoods update at every observation; extract the grid maximum at each checkpoint. The likelihood bank could also supply a grid estimate at every step without refiltering. |

The candidate banks are fixed parameter grids, not joint ensembles that learn a
parameter posterior. The primary particle bank has 19 × 8,192 state particles.
Candidates share forecast noise and resampling uniforms within each method to
reduce variability in comparisons between candidate parameters. The EnKF and
particle methods have separate random streams. All methods share the same
simulated observations and the same known signal and sensor constants.

The Kalman likelihood is exact for the Euler-discretized, right-endpoint linear
Gaussian model; its maximization is numerical. The bootstrap filter uses exact
per-particle observation densities and Monte Carlo integration over the hidden
state. The augmented EnKF uses its ensemble approximation, including the
Poisson ensemble update. None of the baselines uses future observations or a
smoother. Gaussian observations are evaluated at the right endpoint, and
Poisson observations at the left endpoint.

At each checkpoint, only its observation prefix enters the estimate. The
score-root continuation considers all earlier roots, including intermediate
steps between checkpoints. Missing roots use the existing `hold_previous`
policy starting at the grid midpoint 0.55; those estimates remain in the
accuracy analysis, with missing-root rates reported separately.

## Fixed evaluation plan

The primary metric is the paired difference in MAE at T=1000 for each model:
`MAE(score root) - MAE(baseline)`. Positive values favor the baseline.
The two earlier checkpoints are secondary. Report MAE, RMSE, signed bias,
empirical standard deviation, boundary rates, and missing-root rates. No
replicates are removed for poor estimates. A nonfinite estimate or a worker
failure prevents a completed summary; saved records remain resumable.

Resample **records**, preserving the two estimates within each pair. The
50,000-resample percentile bootstrap yields pointwise 95% intervals for the
paired MAE differences. `family_ci_low/high` use Bonferroni-adjusted percentile
levels across all nine model/checkpoint comparisons, giving approximate 95%
family coverage. These intervals are deliberately wider. Bootstrap coverage
is approximate, particularly with a finite sample; no significance claims
should rely on selecting the most favorable checkpoint. `paired_delta_mcse`
is the empirical standard error of the mean paired absolute-error difference.

Each primary record has one independent filter realization. Across-record
uncertainty therefore reflects the performance of these randomized
implementations under simulated data and independently seeded filters. It
does not establish closeness to the infinite-particle likelihood, account for
numerical bias, or provide a parameter confidence interval for an individual
record. The three checkpoints from a record are dependent and must never be
counted as three independent replicates.

On the first **five records** per model, the runner repeats the score-root
filter with an independent seed. For the two nonlinear models it also:

- Repeats the primary particle filter with an independent seed.
- Doubles the particle count to 16,384 with the primary seed.
- Refines the particle parameter grid to 37 candidates, spacing 0.025, at 8,192 particles.

The sensitivity subset is fixed by record index. It is not pooled with the
primary records and does not increase the effective sample size. Particle
count checks use the same seed but are not guaranteed to be nested samples
or a confidence bound. The finer grid contains every primary grid point and
uses the same random-number coordinates. A stable estimate on this small
subset is evidence about that subset, not proof of convergence elsewhere.
Filter-seed checks include the EnKF because its numerical variability also
matters. The saved surfaces permit inspection of close or competing maxima.

The primary particle estimator is restricted to spacing 0.05, while the
score root interpolates and the Kalman optimizer is continuous. This is a
material resolution difference at long horizons. If the grid or particle
checks change estimates appreciably, report it and limit claims about method
accuracy. Do not reinterpret this implementation as a definitive comparison
with every modern particle-based parameter estimator.

This design covers one true parameter, one sensor/noise configuration per
model, and one discretization step. It does not establish continuous-time
accuracy, asymptotic efficiency, or broad superiority. Changes made after
inspecting primary results must be documented and evaluated separately.

## Completed analysis

The study subsequently finished by resuming the saved records: all 120 planned
records are complete. The [full audited report](../reference/comparison/analysis/report.md)
and [archived results](../reference/README.md) are included in the repository.
Regenerate the accuracy tables, numerical checks, and scientific figures from
the archive without rerunning the filters:

```bash
python scripts/analyze_comparison.py --input reference/comparison
```

The output is `reference/comparison/analysis/report.md`, with PNG, PDF, and SVG
figures, input checksums, and CSV tables. The report separates the planned
comparison from a post hoc diagnostic that rounds score-root estimates to the
particle grid. Its interpretation is specific to the completed 20261008 study;
the script rejects a different manifest.

## Start the runs on Linux

From the repository directory, install once using Python 3.12:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements/comparison-py312.txt
.venv/bin/python -m pip install --no-deps --no-build-isolation -e .
```

The lock file records the environment used for verification. The ordinary
package dependencies in `pyproject.toml` remain available for other supported
environments; resume requires matching software versions and source files.

For the full comparison:

```bash
bash scripts/run_comparison.sh full
```

The launcher uses 24 processes and one BLAS thread per process. The original
70–90 minute forecast from short-horizon checks was too optimistic for the
full workload. The first full attempt stopped with 51 of 120 records saved
and no completed summary. Completed primary nonlinear records took roughly
20–22 minutes each, and completed sensitivity records took 77–85 minutes.
Completion of this full profile within two hours has therefore **not been
verified**. The scheduler starts expensive particle sensitivity records first
to reduce the final queue tail. A **115-minute time limit**, with
at most 20 seconds to terminate remaining processes, keeps a launch below
two hours. If time expires, the launcher returns failure and labels the run
incomplete. It never publishes partial results as a completed comparison.
The launcher records timestamps and its exit status in the log; the first
attempt's launcher logged computation output without the final exit status,
so that attempt's precise stop reason is unavailable.

Quick verification and an optional, separate-seed pilot:

```bash
bash scripts/run_comparison.sh smoke
bash scripts/run_comparison.sh pilot
```

The smoke run uses two tiny records per model and is not scientific evidence.
The pilot uses three records per model through T=250, seed 20261009, and runs
all numerical checks on those records. Use it for computational stability,
not to choose whichever settings make one method look best. Its 30-minute
limit is separate from the full launch. The full profile does not reuse its
records. Thread count can be changed with `COMPARISON_WORKERS=16`; slower
throughput may exhaust the time limit.

After interruption, resume the same profile:

```bash
bash scripts/run_comparison.sh full --resume
```

Resume starts another time-limited launch, so combined elapsed time can exceed
two hours. It recomputes only unsaved records. It refuses changes in the
experiment settings, source digest, or dependency versions. Each record is
saved atomically when its computation finishes, regardless of other workers'
completion order. A lock prevents concurrent launcher writes to the same
output. Existing output directories require `--resume`.

## Artifacts and runtime interpretation

Results are written to `runs/comparison-full/`:

- `manifest.json`: resolved design, source hash, seeds, and software versions.
- `records/<case>/<replicate>.json`: estimates, checkpoint score/likelihood surfaces, observation checksum, seeds, and costs, saved atomically.
- `estimates.csv`: all primary and sensitivity estimates, with explicit variant identifiers.
- `summary.csv`: primary accuracy metrics and paired intervals only.
- `sensitivity.csv`: changes under numerical checks; never pooled into primary summaries.
- `costs.csv`: CPU and wall time for each method/variant's entire record workload.

The console log is `runs/comparison-full.log`. Raw observations are regenerated
from semantic random-number coordinates, with a checksum to verify pairing.
Keep the manifests and individual records when sharing results.

CPU cost excludes data simulation and artifact writing. The score-root method
computes a running root path at every step; the particle bank accumulates one
stream of likelihoods and extracts three grid maxima; Kalman optimization
refilters prefixes at each checkpoint. Timing compares these implemented
workloads, not equivalent per-observation parameter optimization or online
latency. `costs.csv` separates sensitivity work from primary estimation.
The runner records no invented per-checkpoint CPU cost for the score path.
