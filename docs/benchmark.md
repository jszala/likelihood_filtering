# Paired parameter-estimation benchmark

This is a finite-horizon synthetic comparison of the online augmented-EnKF
score-root estimator with two independent likelihood-based benchmarks. It is
separate from the longer thesis illustrations shown in the README.

## Design

The three cases in [`configs/benchmark.yaml`](../configs/benchmark.yaml) use the
same OU signal parameters, observation maps, noise levels, parameter range, and
EnKF ensemble sizes as the thesis profile. Each case has 30 independently
simulated observation records with true parameter `0.37`, step `0.01`, and
horizon `250`. Both methods for a case receive exactly the same observation
increments or counts. The score-root estimator uses 19 candidate parameters
and the same continuation rule as the package's normal experiment runner.
The master seed is `20260820`; simulation, EnKF, and particle streams are
derived from semantic case, replicate, and time-step coordinates by
[`seeded_rng`](../src/likelihood_filtering/randomness.py). Raw observations are
regenerated from these streams rather than stored in the result directory.

The linear Gaussian benchmark maximizes the **exact marginal likelihood for
the repository's discrete model**. It uses a scalar Kalman filter for

\[
X_{n+1}=(1-\theta\Delta t)X_n+\sigma\sqrt{\Delta t}\,\xi_n,
\qquad
\Delta Y_n=cX_{n+1}\Delta t+\sigma_{\rm obs}\sqrt{\Delta t}\,\varepsilon_n.
\]

The bounded scalar optimizer also evaluates both parameter bounds. This is an
exact numerical benchmark for the discretized linear Gaussian model, not for
an exactly sampled continuous-time OU process.

The nonlinear Gaussian and Poisson benchmarks use a bootstrap particle filter
with 4,096 particles. Each candidate parameter receives the exact Gaussian or
Poisson observation density, and the estimator maximizes the estimated
marginal likelihood over the same 19-point grid. The nonlinear Gaussian
case forecasts before applying its right-endpoint observation. The Poisson
case applies its left-endpoint observation before forecasting. We accumulate
likelihoods in log space and resample systematically when effective sample
size drops below half the particle count. The first five records in each
nonlinear case are repeated with 8,192 particles to expose particle-count
sensitivity. Residual differences between 4,096 and 8,192 particles remain
part of the reported limitation.
Candidate parameters share forecast random draws and resampling offsets, which
reduces Monte Carlo variation across the estimated likelihood surface.

The existing EnKF predictive log-likelihood surface is recorded only as an
internal diagnostic. It shares the EnKF approximation with the score-root
estimator and is not an independent benchmark.

## Metrics and interpretation

[`reference/benchmark/per_record.csv`](../reference/benchmark/per_record.csv)
contains every final parameter estimate, signed and absolute error, CPU time,
and boundary or missing-root status. [`summary.csv`](../reference/benchmark/summary.csv)
reports mean absolute error (MAE), bias, median CPU seconds per record, and failure
rates. A paired bootstrap over the 30 shared records gives a 95% interval for
`MAE(score-root) - MAE(baseline)`; negative values favor the score-root method.
The bootstrap measures variation across simulated records, not all sources of
model or Monte Carlo uncertainty. Particle counts and sensitivity checks are
reported in [`particle_check.csv`](../reference/benchmark/particle_check.csv).

CPU time includes parameter estimation, but not simulation or artifact writing.
It is measured with `process_time()` separately inside each worker and should
be read as an approximate computational tradeoff. The score-root run computes
the whole online estimate path, while the likelihood benchmarks return a final
estimate. The comparison uses one true parameter, one noise setting per sensor,
and a finite horizon. The particle MLE is restricted to the 0.05-spaced
parameter grid, while the score-root method interpolates between grid points.
It cannot establish asymptotic efficiency or broad superiority over other
inference methods.

## Regenerate

From an installed checkout of this repository:

```bash
python -m likelihood_filtering.benchmark \
  --config configs/benchmark.yaml --output runs/paired-ou-benchmark --workers 4
```

The runner writes a resolved manifest and appends per-record results as they
finish. If interrupted, rerun the command with `--resume` and the same output
path. To verify the full code path quickly, add `--smoke` and use a separate
output path. The committed results use the command above with
`--output reference/benchmark --workers 4`; the manifest records the resolved settings and
software versions. The full run is intentionally outside CI.
