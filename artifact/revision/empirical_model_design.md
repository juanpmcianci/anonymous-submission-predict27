# Frozen empirical conditional-mixture specification

Frozen on 28 September 2026 under `revision/empirical_protocol.md`, before this implementer accessed any empirical dataset, validation score, or test score. The implementer used small constructed arrays for numerical tests only. This addendum fixes the empirical model independently of the earlier synthetic generating law. It does not change that simulation code or its results.

## Inputs and training quantities

A row contains logit increments `z[n,t]` and aligned actual interval volumes `volume[n,t]`. The caller constructs increments from consecutive transaction closes using the protocol's price clipping rule; the model itself neither clips increments nor constructs volume proxies. The first retained candle supplies the initial price, so the volume paired with the first increment belongs to the second retained candle. Fit each horizon and each declared clipping sensitivity separately, on its training subset only.

Pool the training increments to compute the centered median absolute deviation. The common scale is

\[
s=\max\{1.4826\operatorname{median}|z-\operatorname{median}(z)|,\ 0.05\}.
\]

The floor is 0.05 logit units. Define \(m_v\) as the median strictly positive training interval volume and \(r_t=v_t/m_v\). Require finite increments, finite nonnegative volumes, at least one increment per path, and at least one positive training volume. Neither predictions nor serialization update these fitted quantities. The protocol's stronger requirement of a trade in every retained candle remains the caller's responsibility.

## Catalogue and normalized densities

One index \(j\in\{1,\ldots,12\}\) selects a pair \((w_j,a_j)\) for an entire path. The fixed Cartesian catalogue uses \(w\in\{0.2,0.5,0.8\}\) and \(a\in\{0,0.1,0.3,0.6\}\), ordered first by weight and then by amplitude. There is no new catalogue draw at each increment. The component identity can vary by increment and is integrated inside each one-step density.

Write \(q=2y-1\), \(g(r)=2r/(1+r)\), and \(\mu_j=a_js\). For each observation, normalize the positive gate values

\[
\bigl(w_j(1+r)^{1/2},\ (1-w_j)(1+r)^{3/10}/2,\ (1-w_j)/2\bigr)
\]

to obtain \((\alpha_{+,j},\alpha_{-,j},\alpha_{0,j})\). The component laws, in that same order, are

\[
N\!\left(q\mu_jg(r),\frac{s^2}{1+0.25r}\right),\qquad
N\!\left(-0.5q\mu_jg(r),s^2\right),\qquad
t_5\!\left(0,s\sqrt{3/5}\right).
\]

The Student distribution uses a location/scale parameterization; its variance is \(s^2\). Both Gaussian variance arguments above denote variances, not standard deviations. All three distributions integrate to one. Their weighted sum defines \(f_{y,j}(z_t\mid v_t)\). The labels “positive,” “opposing,” and “centered” describe statistical components; they are not estimates of trader intent.

The volume-free ablation sets \(r_t=1\) in every gate, mean, and variance during both fitting and prediction. It otherwise repeats the same training procedure. In particular, volume does not enter the ablation's fitted mixture weights.

## Fitting and prediction

Conditional on a shared state, multiply the increment densities:

\[
L_{ij}=\prod_t f_{y_i,j}(z_{it}\mid v_{it}),\qquad
p(z_i\mid y_i,v_i)=\sum_j\pi_jL_{ij}.
\]

Fit the global catalogue masses \(\pi\) using only training outcomes and training paths. Start at \(\pi_j=1/12\). Each EM iteration computes the exact responsibilities and updates

\[
R_{ij}=\frac{\pi_jL_{ij}}{\sum_k\pi_kL_{ik}},\qquad
\pi_j^{\mathrm{new}}=\frac{\sum_iR_{ij}+1/12}{n+1}.
\]

This update maximizes the conditional training log likelihood plus \((1/12)\sum_j\log\pi_j\), the Dirichlet MAP objective with parameter \(1+1/12\) per state. The total added mass is one. Stop when the absolute objective increment is at most \(10^{-9}(1+|\text{previous objective}|)\), or after 2,000 iterations. Save the convergence flag and both likelihood traces; reaching the iteration cap does not trigger a new model choice.

Use the Jeffreys-smoothed training label frequency \(p_Y=(\sum_i y_i+0.5)/(n+1)\). This explicit smoothing keeps the outcome prior finite even for a single-class training subset. For an unseen path, compute

\[
\log B(z,v)=\operatorname{LSE}_j\left(\log\pi_j+\sum_t\log f_{1,j}\right)
-\operatorname{LSE}_j\left(\log\pi_j+\sum_t\log f_{0,j}\right),
\quad
\widehat p=\operatorname{expit}\{\operatorname{logit}(p_Y)+\log B\}.
\]

The model returns one probability per path. It does not use the market price as a prior. Any validation-fitted price/likelihood combination belongs to the separate pipeline specified in the protocol.

## Numerical checks and limits

The implementation evaluates gates, component densities, path products, and catalogue sums in the log domain. The Student tail uses a log-domain squared residual to avoid overflow on extreme finite inputs. Tests cover density normalization, manual whole-path catalogue enumeration, outcome reflection, the no-signal states, volume-ablation invariance, unchanged training scalers at prediction, EM objective monotonicity, serialization, and finite predictions at extreme inputs.

Observed volumes are a conditioning design here; this model does not assert that real trading volume is exogenous. Conditional independence within a path is a modeling assumption that real market histories may violate. Neither supervised catalogue fitting nor a positive test comparison establishes trader identification or a causal account of price changes.

Implementation: `code/empirical_mixture.py`; checks: `code/test_empirical_mixture.py`. The fitting artifact records exact configuration, catalogue, weights, training scale, training volume median, outcome prior, training dimensions and label count, convergence diagnostics, and the implementation SHA-256 hash. No validation or test arrays enter `fit` unless a caller violates the frozen protocol.

## Fixed Gaussian history comparator

Added on 28 September 2026 under the protocol amendment authorized before held-out scoring. No empirical data or scores informed these constants. This comparator models within-path covariance directly; it has no volume input.

For each asset separately, fit a mean vector \(m_0\) and \(m_1\) from the corresponding training outcome classes. Both classes must occur in the asset's training subset. Let \(R\) contain training residual vectors after subtracting their own class means and let \(n\) denote that asset's training count. The shared covariance is

\[
S=R^\top R/n,\qquad
\Sigma=0.9S+0.1\operatorname{diag}(\operatorname{diag}S).
\]

Replace each diagonal entry below \(10^{-4}\) by \(10^{-4}\). If the smallest eigenvalue of the resulting symmetric matrix falls below \(10^{-8}\), replace eigenvalues below \(10^{-8}\) by that floor and reconstruct the covariance. Record the number of diagonal floors, whether the emergency eigenvalue floor ran, and the smallest eigenvalues before and after that repair. These constants do not depend on validation or test performance.

The two class laws use the same covariance: \(z\mid Y=y,A\sim N(m_{y,A},\Sigma_A)\). Therefore the exact full-path log Bayes factor simplifies to

\[
\log B_A(z)=(z-(m_{1,A}+m_{0,A})/2)^\top\Sigma_A^{-1}(m_{1,A}-m_{0,A}).
\]

Combine this value with the asset-specific Jeffreys training prior \((n_{1,A}+0.5)/(n_A+1)\). The model requires the same path dimension at training and prediction; unknown assets raise an error. The artifact stores means, covariances, discriminant coefficients, priors, class counts, numerical diagnostics, fixed configuration, and source hash.

`GaussianHistory` lives in the same module and exposes `fit(z, asset, y)`, `log_bayes_factor(z, asset)`, `predict_proba(z, asset)`, and JSON-compatible `to_dict`/`from_dict`. Constructed two-dimensional tests compare its discriminant with direct multivariate Gaussian densities, check that correlations alter the result, verify outcome-label symmetry, and exercise the covariance floor. The pipeline owns any validation-fitted combination with the latest market price.
