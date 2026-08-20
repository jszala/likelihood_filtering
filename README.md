# Likelihood estimation and filtering

This repository contains the numerical examples from the likelihood-estimation
chapter of my PhD thesis. The main examples are Ornstein--Uhlenbeck signals with
linear Gaussian, nonlinear Gaussian, and Poisson observations. A one-dimensional
finite-element heat equation is included as a small spatial example.

The filter augments each ensemble member with the complete-data score and
information. Parameter estimates are obtained by interpolating a score root on a
fixed grid. The recursive Louis/Newton calculation is kept in
`likelihood_filtering.experimental` because it is less robust than the grid method.

## Observation convention

Gaussian observations use one convention throughout:

\[
dY_t=h(X_t)\,dt+\sigma_{\mathrm{obs}}\,dV_t,
\qquad
\Delta Y_n=h(X_{t_n})\Delta t+\sigma_{\mathrm{obs}}\sqrt{\Delta t}\,Z_n.
\]

Configuration files therefore use `observation_std`. The covariance of an
increment is `observation_std**2 * dt`. The linear and nonlinear OU experiments use
the thesis values `0.068526` and `0.0117758`, respectively. Ambiguous legacy names
are rejected by the configuration loader. A full covariance matrix can be supplied
to `GaussianObservation` through the Python API as `observation_covariance`.

For the heat equation, `state_noise_variance: 0.015` is the state-driving variance;
the corresponding amplitude is its square root. This is separate from observation
noise.

## Installation and use

The package requires Python 3.11 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

Run the small profile from the repository root:

```bash
likelihood-filter run configs/quick.yaml --output runs/quick
```

The full thesis settings are recorded separately and are substantially more
expensive:

```bash
likelihood-filter run configs/thesis.yaml --output runs/thesis --workers 4
```

`--workers` only parallelizes independent Monte Carlo replicates. A dry run reports
the approximate number of ensemble steps without starting the calculation:

```bash
likelihood-filter run configs/thesis.yaml --output runs/thesis --dry-run
```

Each output directory contains a resolved `manifest.json` and `summary.csv`.
Individual case directories contain `metrics.csv`, compact `aggregates.npz` files,
figures, and compressed replicate artifacts on a reduced time grid. Ensemble
histories are not saved. The notebook in `notebooks/` only reads these artifacts; it
does not run the filters.

## Scope

The estimated parameter is scalar. The EnKF uses stochastic perturbed-observation
updates, and Poisson counts use the usual Gaussian ensemble approximation. The
Euler and finite-element discretizations are intended to reproduce the thesis
experiments, not to provide a general SDE or SPDE library. Custom finite-dimensional
or spatial signals can implement `ParametricSignal` directly with arrays shaped as
`(ensemble, *state_shape)`.

The companion [SPDE_Poisson_filtering](https://github.com/jszala/SPDE_Poisson_filtering)
repository studies Poisson filtering for spatial models. This project follows the
same small-package, YAML-profile, and saved-artifact layout, while keeping its
likelihood and score calculations independent.

Further details are in [the numerical methods](docs/numerical_methods.md) and the
[thesis-to-code map](docs/thesis_to_code.md).

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
