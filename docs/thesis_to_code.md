# Thesis-to-code map

| Thesis object | Implementation |
| --- | --- |
| OU signal and complete-data derivatives | `signals.OrnsteinUhlenbeckSignal` |
| 1D stochastic heat equation | `signals.Heat1DFEMSignal` |
| Gaussian increment model | `observations.GaussianObservation` |
| Poisson count model | `observations.PoissonObservation` |
| Augmented EnKF | `filtering.run_augmented_filter` and `enkf` |
| Fisher--Louis information | `diagnostics.fisher_louis_information` |
| Grid score-root estimate | `estimation.GridScoreRootEstimator` |
| Recursive Newton calculation | `experimental.RecursiveLouisEstimator` |
| Monte Carlo studies | `experiment.run_experiment` |
| Canonical numerical settings | `configs/thesis.yaml` |

The linear Gaussian, nonlinear Gaussian, and Poisson OU studies share the signal
parameters and parameter grid from the thesis. The recursive study is explicitly
marked experimental in both its case name and import path. The heat example uses 20
interior finite-element degrees of freedom and ten point sensors.

Historical particle-filter variants, exact Kalman comparisons, two-dimensional
solvers, spectral experiments, real-data analyses, and cluster scripts are not
mapped into this repository.
