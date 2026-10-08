# likelihood_filtering

**Estimate a hidden system's mean-reversion rate from noisy measurements.** This
Python package studies a process whose state changes continuously but cannot be
observed directly. It filters the observations and estimates the unknown parameter
as data arrive.

The main examples share a one-dimensional Ornstein–Uhlenbeck (OU) signal:

$$
dX_t = -\theta X_t\,dt + \sigma\,dW_t.
$$

The signal diffusion amplitude is $\sigma=0.15$. Here, $X_t$ is the **hidden
signal** and $\theta$ controls how quickly it returns toward zero. The
estimator sees only one of these observation streams, not $X_t$:

| Observation model | What is recorded |
| --- | --- |
| Linear Gaussian | Noisy continuous measurements proportional to $X_t$ |
| Nonlinear Gaussian | Noisy continuous measurements through a sigmoid sensor |
| Poisson | Event counts whose rate depends on $X_t$ |

## Paired parameter-estimation benchmark

At $T=250$, both estimators received the **same 30 simulated observation
records per sensor** ($\theta_0=0.37$, $\Delta t=0.01$). The online EnKF
score-root estimator is compared with an exact discrete-model Kalman likelihood
MLE for linear Gaussian observations and a 4,096-particle likelihood MLE for the
nonlinear and Poisson observations.

| Observation model | Score-root MAE (bias) | Independent MLE MAE (bias) | MAE difference, 95% paired interval | Median CPU s, score / MLE |
| --- | ---: | ---: | ---: | ---: |
| Linear Gaussian · Kalman | 0.0535 (+0.0045) | 0.0429 (−0.0051) | +0.0106 [+0.0005, +0.0196] | 61.8 / 0.23 |
| Nonlinear Gaussian · particle filter | 0.0608 (+0.0287) | 0.0820 (+0.0300) | −0.0212 [−0.0452, +0.0032] | 72.0 / 66.1 |
| Poisson · particle filter | 0.0510 (+0.0271) | 0.0617 (+0.0317) | −0.0106 [−0.0235, +0.0018] | 71.1 / 80.2 |

The difference is score-root MAE minus baseline MAE, so positive values favor
the baseline. Kalman had lower error in the linear case; the nonlinear and
Poisson intervals include zero. Every final score estimate had a root, and no
method reached a parameter bound (0/30 records for each method and sensor).
The particle MLE is approximate: among five preselected records per nonlinear
sensor, doubling particles to 8,192 changed a grid estimate by up to 0.15 for
nonlinear Gaussian and 0.05 for Poisson. These are finite-horizon synthetic
results at one true parameter, not a general performance claim. The particle
MLE uses a 0.05-spaced grid, while the score root interpolates; CPU times
compare a full online path with a final-only likelihood estimate.

The [benchmark design and regeneration command](docs/benchmark.md),
[configuration](configs/benchmark.yaml), [resolved manifest](reference/benchmark/manifest.json),
[per-record estimates](reference/benchmark/per_record.csv),
[full summary](reference/benchmark/summary.csv), and
[particle-count checks](reference/benchmark/particle_check.csv) are committed.
The longer thesis trajectories below are separate experiments.

![Four-panel plot showing one hidden Ornstein–Uhlenbeck signal and the observed increments from linear Gaussian, nonlinear Gaussian, and Poisson sensors driven by that same signal](docs/assets/signal_and_observations.png)

*What is observed:* The upper-left panel is one simulated hidden signal path.
The other panels show the corresponding observed increments $\Delta Y_n$
(discrete samples of $dY_t$) over a short window with $\Delta t=0.01$.
The dark curves are conditional mean increments given the hidden signal, shown
only to explain the observation models; the estimator receives the noisy
increments, not the signal or those means. Generate this illustration with
`python scripts/plot_readme_observations.py`.

**Input:** Gaussian observation increments or Poisson counts over time. **Output:**
a running estimate $\hat\theta_t$ of the mean-reversion rate, with score and
information diagnostics. The command-line experiments generate simulated data;
the Python filtering API also accepts supplied observation increments.

![Estimator trajectories for linear Gaussian, nonlinear Gaussian, and Poisson observations, concentrating around the true parameter over time](docs/assets/estimator_paths.png)

*Thesis numerical experiment:* Each panel shows 150 independently simulated
observation records for the same hidden OU model, with true parameter
$\theta_0=0.37$ and horizon $T=1000$. Pale blue lines are online estimates, the
black line is their median, and the red dashed line is the true value. The
estimates concentrate near $\theta_0$ as more observations arrive. These are
simulation results, reproduced from the numerical experiments chapter of the
author's PhD thesis. The original long-run artifacts are not committed here;
`configs/thesis.yaml` records their settings. The committed `reference/quick`
artifacts come from the much shorter smoke-test profile and do not reproduce
this figure.

## How the estimate is computed

1. For each candidate $\theta$ on a grid, run an **augmented ensemble Kalman
   filter**. Each ensemble member carries a possible hidden state together with
   the complete-data score and information.
2. Condition those quantities on the observations to approximate the **marginal
   likelihood score** for each candidate parameter.
3. Find where the score crosses zero, interpolate between neighboring grid
   points, and repeat as observations arrive to obtain $\hat\theta_t$.

![Estimated likelihood score across candidate parameters at four time horizons for the three observation models; each score crosses zero near the true parameter](docs/assets/score_roots.png)

*Thesis numerical experiment:* Each panel shows the estimated score across the
parameter grid at observation horizons of 250, 500, 750, and 1000 time units
for one representative observation record. The horizontal line is zero; its
crossing gives the score-root estimate. The dotted vertical line marks
$\theta_0=0.37$.

The package also includes a one-dimensional finite-element stochastic heat
equation with ten noisy spatial sensors. It is a small spatial proof of concept;
the OU examples above are the main demonstration. The single-filter
Louis/Newton estimator is available in `likelihood_filtering.experimental` as
a less robust research approximation.

## Install and run

Python 3.11 or newer is required. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

Run the quick profile:

```bash
likelihood-filter run configs/quick.yaml --output runs/quick
```

This is a **smoke test**, with two replicates and a short horizon for each OU
case. Its plots are not performance evidence. The long-run settings corresponding
to the thesis examples are recorded separately and are substantially more
expensive:

```bash
likelihood-filter run configs/thesis.yaml --output runs/thesis --workers 4
```

`--workers` parallelizes independent Monte Carlo replicates only. Inspect the
approximate number of ensemble steps before running the full profile:

```bash
likelihood-filter run configs/thesis.yaml --output runs/thesis --dry-run
```

Each output directory contains a resolved `manifest.json` and `summary.csv`.
Individual case directories contain `metrics.csv`, compact `aggregates.npz`
files, figures, and compressed replicate artifacts on a reduced time grid.
Ensemble histories are not saved. The notebook in `notebooks/` reads these
artifacts without rerunning the filters.

## Model and implementation notes

For the scalar OU Gaussian cases, $\gamma$ denotes observation variance per
unit time. The configured Gaussian profiles evaluate the observation map at
the right endpoint of each step:

$$
dY_t=h(X_t)\,dt+\sqrt{\gamma}\,dV_t,
\qquad
\Delta Y_n=h(X_{t_{n+1}})\Delta t+\sqrt{\gamma\Delta t}\,Z_n.
$$

Configuration files use `observation_std` $=\sqrt{\gamma}$, so the variance
of an increment is $\gamma\Delta t$ or `observation_std**2 * dt`. The linear
and nonlinear OU profiles have `observation_std` values `0.068526` and
`0.0117758`, respectively; their corresponding $\gamma$ values are the
squares of those numbers. Ambiguous legacy names are rejected by the
configuration loader. A full covariance matrix can be supplied to
`GaussianObservation` through the Python API as `observation_covariance`.

For the heat equation, `state_noise_variance: 0.015` is the state-driving
variance; its amplitude is the square root of that value. This is separate
from observation noise.

The estimated parameter is scalar. The EnKF uses stochastic
perturbed-observation updates, and Poisson counts use a Gaussian ensemble
approximation. The Euler and finite-element discretizations are intended to
reproduce the thesis experiments. This is a focused inference package rather
than a general SDE or SPDE library. Custom finite-dimensional or spatial
signals can implement `ParametricSignal` with arrays shaped
`(ensemble, *state_shape)`.

The companion [SPDE_Poisson_filtering](https://github.com/jszala/SPDE_Poisson_filtering)
repository studies Poisson filtering for spatial models. This project follows
the same small-package, YAML-profile, and saved-artifact layout while keeping
its likelihood and score calculations independent.

Further details are in [the numerical methods](docs/numerical_methods.md), the
[thesis-to-code map](docs/thesis_to_code.md), and the
[reference output notes](reference/README.md).

## Development

```bash
python -m pip install -e '.[dev]'
ruff check .
ruff format --check .
pytest
python -m build
```

The source is released under the BSD 3-Clause license. Citation metadata is in
`CITATION.cff`.
