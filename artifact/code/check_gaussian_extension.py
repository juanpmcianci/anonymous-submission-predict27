"""Deterministic numerical checks for the Gaussian history-value extension.

Run from any directory:
    python3 code/check_gaussian_extension.py

This script uses constructed model parameters and quadrature only. It never
loads empirical markets, outcomes, predictions, or scoring artifacts. Its
default output is revision/qa/gaussian_extension_check.json. Numerical checks
support implementation and illustration reproducibility; the manuscript's
analytic arguments establish the propositions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import scipy
from scipy.integrate import quad
from scipy.special import expit, logsumexp, roots_hermitenorm, xlogy
from scipy.stats import multivariate_normal, norm


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "revision/qa/gaussian_extension_check.json"
HERMITE_ORDER = 160
NODES, WEIGHTS = roots_hermitenorm(HERMITE_ORDER)
WEIGHTS = WEIGHTS / np.sqrt(2 * np.pi)


def require(condition, message):
    """Checks remain active when Python is invoked with optimization enabled."""
    if not bool(condition):
        raise AssertionError(message)


def close(actual, expected, tolerance, message):
    require(
        np.isfinite(actual) and np.isfinite(expected)
        and abs(actual - expected) <= tolerance,
        f"{message}: actual={actual!r}, expected={expected!r}, tolerance={tolerance}",
    )


def entropy(q):
    return -xlogy(q, q) - xlogy(1 - q, 1 - q)


def scalar_metrics(gamma, amplitudes, masses, prior):
    """Return Brier Bayes risk, outcome information, and its I-MMSE derivative.

    Conditional on Y, the channel input is +(amplitude) or -(amplitude).
    Each Gaussian mixture component is integrated over standard-normal noise
    by 160-point Gauss-Hermite quadrature. Class-conditional means are formed
    with log-normalized weights, avoiding division by a posterior near zero.
    """
    amplitudes, masses = np.asarray(amplitudes), np.asarray(masses)
    signed = np.r_[-amplitudes, amplitudes]
    joint_mass = np.r_[(1 - prior) * masses, prior * masses]
    observations = np.sqrt(gamma) * signed[:, None] + NODES[None, :]
    component_logs = (
        np.log(joint_mass)[None, None, :]
        - 0.5 * (observations[..., None] - np.sqrt(gamma) * signed) ** 2
    )
    split = len(amplitudes)
    log0, log1 = component_logs[..., :split], component_logs[..., split:]
    sum0, sum1 = logsumexp(log0, axis=-1), logsumexp(log1, axis=-1)
    q = expit(sum1 - sum0)

    def average(values):
        return np.sum(joint_mass[:, None] * WEIGHTS[None, :] * values)

    brier = average(q * (1 - q))
    information = entropy(prior) - average(entropy(q))
    mean0 = np.sum(np.exp(log0 - sum0[..., None]) * signed[:split], axis=-1)
    mean1 = np.sum(np.exp(log1 - sum1[..., None]) * signed[split:], axis=-1)
    derivative = 0.5 * average(q * (1 - q) * (mean1 - mean0) ** 2)
    return float(brier), float(information), float(derivative)


def known_amplitude_adaptive(gamma):
    """Independent adaptive quadrature, for A=1 and the equal outcome prior."""
    def integrand(n, logarithmic):
        q = expit(2 * gamma + 2 * np.sqrt(gamma) * n)
        loss = entropy(q) if logarithmic else q * (1 - q)
        return norm.pdf(n) * loss

    # Symmetry permits integrating conditional on Y=1 rather than both classes.
    brier, brier_error = quad(
        lambda n: integrand(n, False), -np.inf, np.inf,
        epsabs=1e-12, epsrel=1e-12,
    )
    log_loss, log_error = quad(
        lambda n: integrand(n, True), -np.inf, np.inf,
        epsabs=1e-12, epsrel=1e-12,
    )
    return {
        "brier": float(brier), "log_loss": float(log_loss),
        "information_nats": float(np.log(2) - log_loss),
        "quadrature_error_estimates": {
            "brier": float(brier_error), "log_loss": float(log_error),
        },
    }


def run_checks():
    # A single reproducible covariance fixture, not a Monte Carlo experiment.
    rng = np.random.default_rng(184)
    matrix = rng.normal(size=(4, 4))
    covariance = matrix @ matrix.T + np.eye(4)
    response = np.array([0.2, 1.1, -0.3, 0.7])
    one = np.ones(4)
    precision_response = np.linalg.solve(covariance, response)
    c, m = one @ covariance @ one, one @ response
    jh, je = response @ precision_response, m * m / c
    delta = jh - je
    residual = response - (m / c) * covariance @ one
    close(delta, residual @ np.linalg.solve(covariance, residual), 1e-12,
          "Discarded strength equals the covariance-weighted residual norm")
    require(delta > 0, "Fixture must have strictly positive discarded strength")
    residual_weights = precision_response - (m / c) * one
    close(residual_weights @ covariance @ one, 0, 1e-12,
          "Residual statistic has zero conditional covariance with endpoint")
    close(residual_weights @ covariance @ residual_weights, delta, 1e-12,
          "Residual statistic has variance Delta")
    sufficient_response = 0.3 * covariance @ one
    sufficient_jh = sufficient_response @ np.linalg.solve(covariance, sufficient_response)
    sufficient_je = (one @ sufficient_response) ** 2 / c
    close(sufficient_jh, sufficient_je, 1e-12, "Endpoint-sufficient response has Delta zero")

    prior = 0.3
    amplitudes = np.array([0.0, 0.4, 1.3])
    masses = np.array([0.15, 0.35, 0.50])
    z = rng.normal(size=4)
    u = precision_response @ z
    log_g_plus = logsumexp(np.log(masses) + amplitudes * u - amplitudes ** 2 * jh / 2)
    log_g_minus = logsumexp(np.log(masses) - amplitudes * u - amplitudes ** 2 * jh / 2)
    q_reduced = expit(np.log(prior / (1 - prior)) + log_g_plus - log_g_minus)
    log_full = [
        logsumexp([
            np.log(weight) + multivariate_normal.logpdf(
                z, mean=sign * amp * response, cov=covariance,
            ) for amp, weight in zip(amplitudes, masses)
        ]) for sign in [-1, 1]
    ]
    q_direct = expit(np.log(prior / (1 - prior)) + log_full[1] - log_full[0])
    close(q_reduced, q_direct, 1e-12, "Scalar posterior equals direct Gaussian-mixture posterior")

    zero_brier, zero_information, zero_derivative = scalar_metrics(0, amplitudes, masses, prior)
    close(zero_brier, prior * (1 - prior), 1e-12, "Zero-strength Brier risk")
    close(zero_information, 0, 1e-12, "Zero-strength information")
    close(zero_derivative, 2 * prior * (1 - prior) * (masses @ amplitudes) ** 2,
          1e-12, "Zero-strength conditional I-MMSE derivative")

    derivative_checks = []
    step = 1e-4
    for gamma in [0.05, 0.4, 2.0]:
        left = scalar_metrics(gamma - step, amplitudes, masses, prior)[1]
        right = scalar_metrics(gamma + step, amplitudes, masses, prior)[1]
        numerical = (right - left) / (2 * step)
        analytic = scalar_metrics(gamma, amplitudes, masses, prior)[2]
        close(numerical, analytic, 2e-7, f"Conditional I-MMSE derivative at gamma={gamma}")
        derivative_checks.append({
            "gamma": gamma, "finite_difference": numerical, "formula": analytic,
            "absolute_error": abs(numerical - analytic), "tolerance": 2e-7,
        })

    metrics_e = scalar_metrics(je, amplitudes, masses, prior)
    metrics_h = scalar_metrics(jh, amplitudes, masses, prior)
    gain_log, gain_brier = metrics_h[1] - metrics_e[1], metrics_e[0] - metrics_h[0]
    gain_integral, integral_error = quad(
        lambda g: scalar_metrics(g, amplitudes, masses, prior)[2], je, jh,
        epsabs=1e-11, epsrel=1e-11,
    )
    require(gain_log > 0 and gain_brier > 0, "Unknown-amplitude score gains must be positive")
    close(gain_log, gain_integral, 1e-10, "Unknown-amplitude integrated I-MMSE identity")
    upper_bound = delta * np.sum(masses * amplitudes ** 2) / 2
    require(gain_log <= upper_bound + 1e-12, "Unknown-amplitude information upper bound")

    b_e, i_e, _ = scalar_metrics(je, [1], [1], prior)
    b_h, i_h, _ = scalar_metrics(jh, [1], [1], prior)
    known_integral = 2 * quad(
        lambda g: scalar_metrics(g, [1], [1], prior)[0], je, jh,
        epsabs=1e-11, epsrel=1e-11,
    )[0]
    close(i_h - i_e, known_integral, 1e-10, "Known-amplitude integral equals twice integrated Brier risk")
    require(2 * delta * b_h <= i_h - i_e <= 2 * delta * b_e,
            "Known-amplitude bounds must bracket the information gain")

    # Manuscript's equal-total-volume illustration: a(v)=v and Sigma=I.
    designs = [np.array([1., 1.]), np.array([.5, 1.5])]
    strengths = [
        {"volumes": v.tolist(), "J_endpoint": float(v.sum() ** 2 / 2),
         "J_history": float(v @ v), "discarded_strength": float(v @ v - v.sum() ** 2 / 2)}
        for v in designs
    ]
    close(strengths[0]["J_endpoint"], strengths[1]["J_endpoint"], 0,
          "Both volume designs have the same endpoint strength")
    close(strengths[0]["discarded_strength"], 0, 0, "Equal volumes discard no strength")
    close(strengths[1]["discarded_strength"], .5, 0, "Unequal volumes discard strength one-half")
    endpoint = known_amplitude_adaptive(2.)
    history = known_amplitude_adaptive(2.5)
    for gamma, values in [(2., endpoint), (2.5, history)]:
        gh_brier, gh_information, _ = scalar_metrics(gamma, [1], [1], .5)
        close(gh_brier, values["brier"], 1e-8, "Hermite and adaptive Brier quadrature agree")
        close(gh_information, values["information_nats"], 1e-8,
              "Hermite and adaptive information quadrature agree")
    example_log_gain = endpoint["log_loss"] - history["log_loss"]
    example_brier_gain = endpoint["brier"] - history["brier"]
    close(example_log_gain, .04953220178350804, 1e-10, "Published illustration log-score gain")
    close(example_brier_gain, .015556299212988885, 1e-10, "Published illustration Brier gain")
    example_integral = 2 * quad(
        lambda g: known_amplitude_adaptive(g)["brier"], 2., 2.5,
        epsabs=1e-11, epsrel=1e-11,
    )[0]
    close(example_log_gain, example_integral, 1e-10,
          "Illustration gain agrees with independently adaptive integrated Brier risk")

    return {
        "purpose": "Constructed-parameter quadrature checks; no empirical data or scoring.",
        "checks_passed": True,
        "software": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "settings": {"geometry_seed": 184, "hermite_order": HERMITE_ORDER,
                     "derivative_step": step, "derivative_tolerance": 2e-7,
                     "integral_identity_tolerance": 1e-10,
                     "illustration_adaptive_epsabs": 1e-12, "illustration_adaptive_epsrel": 1e-12},
        "geometry": {"covariance": covariance.tolist(), "response": response.tolist(),
                     "observation": z.tolist(), "J_history": float(jh),
                     "J_endpoint": float(je), "discarded_strength": float(delta),
                     "posterior_direct_minus_reduced": float(q_direct - q_reduced)},
        "unknown_amplitude": {"prior_outcome": prior, "amplitudes": amplitudes.tolist(),
                              "amplitude_masses": masses.tolist(),
                              "conditional_immse_checks": derivative_checks,
                              "log_score_gain": gain_log, "brier_gain": gain_brier,
                              "integrated_derivative": gain_integral,
                              "quadrature_error_estimate": integral_error,
                              "log_score_upper_bound": float(upper_bound)},
        "known_amplitude_identity": {"log_score_gain": i_h - i_e,
                                     "twice_integrated_brier_risk": known_integral,
                                     "bounds": [float(2 * delta * b_h), float(2 * delta * b_e)]},
        "same_endpoint_volume_example": {
            "prior_outcome": .5, "amplitude": 1., "noise_variance": 1.,
            "designs": strengths, "endpoint_either_design": endpoint,
            "history_equal_volumes": endpoint, "history_unequal_volumes": history,
            "log_score_gain": example_log_gain, "brier_gain": example_brier_gain,
            "twice_integrated_brier_risk": example_integral,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = run_checks()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    example = result["same_endpoint_volume_example"]
    print("All Gaussian extension checks passed.")
    print(f"Unknown-amplitude log gain: {result['unknown_amplitude']['log_score_gain']:.12f}")
    print(f"Same-endpoint volume example: Brier gain={example['brier_gain']:.12f}, "
          f"log gain={example['log_score_gain']:.12f} nats")
    print(f"Verification artifact: {args.output.resolve()}")


if __name__ == "__main__":
    main()
