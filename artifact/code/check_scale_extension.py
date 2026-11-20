#!/usr/bin/env python3
"""Deterministic checks of the common-Gamma-precision endpoint extension.

Run from any directory: python code/check_scale_extension.py
Only the JSON report under revision/qa is written; no market data are read.
The two-dimensional score integral uses polar coordinates for a Student t_2
noise vector: its angle is uniform and r^2/(2+r^2) is uniform on (0, 1).
All scores below are Bayes risks in the stated model, not empirical estimates.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.integrate import quad
from scipy.special import expit, gammaln, xlogy
from scipy.stats import multivariate_t


def log_marginal(z, a, sigma0, alpha, beta, sign):
    z, a, sigma0 = map(np.asarray, (z, a, sigma0))
    delta = z - sign * a
    q = float(delta @ np.linalg.solve(sigma0, delta))
    dim = z.size
    return float(
        alpha * np.log(beta) + gammaln(alpha + dim / 2) - gammaln(alpha)
        - dim / 2 * np.log(2 * np.pi) - .5 * np.linalg.slogdet(sigma0)[1]
        - (alpha + dim / 2) * np.log(beta + q / 2)
    )


def log_bf(z, a, sigma0, alpha, beta):
    z, a, sigma0 = map(np.asarray, (z, a, sigma0))
    km = np.linalg.solve(sigma0, z - a)
    q1 = float((z - a) @ km)
    u = float(a @ np.linalg.solve(sigma0, z))
    return float((alpha + z.size / 2) * np.log1p(2 * u / (beta + q1 / 2)))


def endpoint_log_bf(s, m, c, alpha, beta):
    return (alpha + .5) * np.log1p((2 * s * m / c) / (beta + (s - m)**2 / (2 * c)))


def entropy(q):
    return -xlogy(q, q) - xlogy(1 - q, 1 - q)


def aligned_score_risks(radial_order, angular_order):
    """Equal priors, a=(1,1), Sigma0=I, Gamma(1,1); bounded integrands.

    By reflection symmetry, the prior-predictive expectation of either
    posterior entropy or posterior variance equals its expectation under Y=1.
    Integrate the latter with a deterministic polar cubature, in radial chunks.
    """
    nodes, weights = leggauss(radial_order)
    radii = np.sqrt(2 * (1 + nodes) / (1 - nodes))
    angles = 2 * np.pi * (np.arange(angular_order) + .5) / angular_order
    cosine, sine = np.cos(angles), np.sin(angles)
    totals = np.zeros(4)
    for radius, weight in zip(radii, weights / 2):
        z1, z2 = 1 + radius * cosine, 1 + radius * sine
        s = z1 + z2
        q1 = (z1 - 1)**2 + (z2 - 1)**2
        qh = expit(2 * np.log1p(2 * s / (1 + q1 / 2)))
        qe = expit(endpoint_log_bf(s, 2, 2, 1, 1))
        totals += weight * np.array([
            np.mean(qh * (1 - qh)), np.mean(qe * (1 - qe)),
            np.mean(entropy(qh)), np.mean(entropy(qe)),
        ])
    return dict(zip(("history_brier", "endpoint_brier", "history_log", "endpoint_log"), map(float, totals)))


def main():
    sigma0 = np.array([[1.2, .2], [.2, .8]])
    a, z = np.array([.7, -.3]), np.array([.6, 1.1])
    alpha, beta = 1.7, .9
    comparisons = []
    for sign in (-1, 1):
        closed = log_marginal(z, a, sigma0, alpha, beta, sign)
        scipy_value = multivariate_t.logpdf(z, loc=sign * a, shape=beta / alpha * sigma0, df=2 * alpha)
        delta = z - sign * a
        q = float(delta @ np.linalg.solve(sigma0, delta))
        log_constant = alpha * np.log(beta) - gammaln(alpha) - np.log(2 * np.pi) - .5 * np.linalg.slogdet(sigma0)[1]
        integrated, error = quad(lambda precision: np.exp(log_constant + alpha * np.log(precision) - (beta + q / 2) * precision), 0, np.inf, epsabs=1e-12, epsrel=1e-12)
        assert abs(closed - scipy_value) < 2e-12
        assert abs(closed - np.log(integrated)) < 2e-10
        comparisons.append({"outcome_sign": sign, "closed_log_density": closed,
                            "scipy_log_density": float(scipy_value), "quadrature_log_density": float(np.log(integrated)),
                            "quadrature_absolute_error_estimate": float(error)})

    aligned_a, identity = np.ones(2), np.eye(2)
    records = []
    for record, expected in (([1., 1.], 25.), ([2., 0.], 9.)):
        bf = float(np.exp(log_bf(record, aligned_a, identity, 1, 1)))
        assert abs(bf - expected) < 1e-12
        records.append({"z": record, "endpoint": sum(record), "history_bf": bf, "history_posterior": bf / (1 + bf)})
    endpoint_bf = float(np.exp(endpoint_log_bf(2, 2, 2, 1, 1)))
    assert abs(endpoint_bf - 5**1.5) < 1e-12

    limit = []
    lambda0 = 1.3
    target = 2 * lambda0 * 2
    for shape in (10., 100., 1000., 10000., 1000000.):
        values = [log_bf(row["z"], aligned_a, identity, shape, shape / lambda0) for row in records]
        limit.append({"alpha": shape, "log_bfs": values, "target": target})
    assert max(abs(value - target) for value in limit[-1]["log_bfs"]) < 2e-5

    coarse = aligned_score_risks(256, 2048)
    fine = aligned_score_risks(512, 4096)
    errors = {key: abs(fine[key] - coarse[key]) for key in fine}
    assert max(errors.values()) < 2e-7, errors
    gain_brier = fine["endpoint_brier"] - fine["history_brier"]
    gain_log = fine["endpoint_log"] - fine["history_log"]
    assert gain_brier > 100 * max(errors.values())
    assert gain_log > 100 * max(errors.values())
    # Identical decision rule under equal priors: sign(U)=sign(S) here.
    report = {
        "status": "passed",
        "model": "Known amplitude 1; one common Gamma(shape, rate) precision independent of outcome",
        "density_checks": comparisons,
        "same_endpoint_example": {"records": records, "endpoint_bf": endpoint_bf,
                                  "endpoint_posterior": endpoint_bf / (1 + endpoint_bf)},
        "known_scale_limit": limit,
        "deterministic_score_cubature": {
            "parameters": {"a": [1, 1], "Sigma0": [[1, 0], [0, 1]], "alpha": 1, "beta": 1, "prior_y1": .5},
            "coarse_order": [256, 2048], "fine_order": [512, 4096],
            "coarse": coarse, "fine": fine, "absolute_order_differences": errors,
            "brier_gain": gain_brier, "log_gain_nats": gain_log,
            "classification_gain": 0.,
            "error_note": "Order comparison is a numerical convergence diagnostic, not a certified quadrature error bound. Classification equality is algebraic.",
        },
    }
    path = Path(__file__).resolve().parents[1] / "revision" / "qa" / "scale_extension_check.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "report": str(path), "brier_gain": gain_brier,
                      "log_gain_nats": gain_log, "max_order_difference": max(errors.values())}, indent=2))


if __name__ == "__main__":
    main()
