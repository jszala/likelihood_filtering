# Numerical methods

## Signal and complete-data derivatives

The finite-dimensional examples use

\[
dX_t=-\theta\mu X_t\,dt+\sigma\,dW_t.
\]

Euler--Maruyama is used at the observation timestep. For every ensemble member the
forecast also accumulates the complete-data score and information increments

\[
\Delta S=-\frac{\mu}{\sigma}X_n\Delta W_n,
\qquad
\Delta B=\frac{\mu^2}{\sigma^2}X_n^2\Delta t.
\]

The augmented state is `(X, S, B)`. An ordinary stochastic EnKF analysis is applied
to this full vector. The filtered score is the ensemble mean of `S`. The observed
information estimate uses the Fisher--Louis identity

\[
I_t=E[B_t\mid Y]-\operatorname{Var}(S_t\mid Y).
\]

## Observations

For Gaussian data, the observation increment covariance passed to the EnKF is
\(R\Delta t\), where \(R=\sigma_{\mathrm{obs}}^2I\). This is also the convention
used by simulation and by the manifest. The nonlinear map is

\[
h(x)=\{1+\exp[-17.2047(x-0.15)]\}^{-1}.
\]

The Poisson intensity is evaluated at the left endpoint:

\[
\lambda(x)=0.25+\frac{3394.91}{1+\exp[-8(x-0.04)]},
\qquad
\Delta Y_n\sim\operatorname{Poisson}(\lambda(X_{t_n})\Delta t).
\]

## Parameter estimates

A separate augmented filter is run at each candidate parameter, with common random
numbers for its forecast and analysis perturbations. At each stored time the code
finds sign changes in the estimated score and linearly interpolates a root. If
several roots exist, the one nearest the previous estimate is chosen. The default
fallback keeps the previous estimate when no bracket is present.

The experimental recursive method runs one filter. It takes a clipped Newton step
from the filtered score and Louis information, then transports the score ensemble
to the new parameter by the first-order update `S <- S - delta_theta * B`.

## Heat equation

The spatial example uses continuous piecewise-linear finite elements on `(0, 0.1)`
with homogeneous Dirichlet boundary conditions. The drift is stepped implicitly:

\[
(M+\theta\Delta t K)u_{n+1}=Mu_n+\sqrt{q\Delta t}\,M^{1/2}Z_n,
\]

where `q = state_noise_variance = 0.015`. Point sensors are constructed by
evaluating the finite-element basis. No two-dimensional or spectral solver is part
of this repository.

## Reproducibility

Random generators are built from a master seed and semantic coordinates for the
case, replicate, source, and timestep. NumPy fills the ensemble and observation
coordinates in a fixed order. Candidate parameters deliberately share the same
forecast and analysis draws. Replicate artifacts are written independently, so
their results do not depend on the worker count.
