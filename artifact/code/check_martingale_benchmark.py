"""Data-free quadrature checks of the ideal-price conditioning identities.

Conditional on Y, Z has mean (2Y-1)*variance/2 and the stated variance.
Its outcome log likelihood ratio is exactly Z. The posterior price is a
martingale under the prior-predictive law, despite outcome-conditional drift.
The manuscript proves these facts; quadrature is an independent check.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.special import expit, logit, roots_hermitenorm
from scipy.stats import norm


def check():
    nodes, weights = roots_hermitenorm(192)
    weights = weights / np.sqrt(2 * np.pi)
    records = []
    for prior in (0.15, 0.5, 0.85):
        for variance in (0.2, 1.0, 2.0):
            conditional_means, second_moments = [], []
            for outcome in (0, 1):
                z = (2 * outcome - 1) * variance / 2 + np.sqrt(variance) * nodes
                llr = norm.logpdf(z, variance / 2, np.sqrt(variance)) - norm.logpdf(z, -variance / 2, np.sqrt(variance))
                np.testing.assert_allclose(llr, z, atol=3e-12, rtol=1e-12)
                posterior = expit(logit(prior) + z)
                conditional_means.append(float(weights @ posterior))
                second_moments.append(float(weights @ posterior**2))
            unconditional_mean = (1 - prior) * conditional_means[0] + prior * conditional_means[1]
            price_variance = (1 - prior) * second_moments[0] + prior * second_moments[1] - prior**2
            actual_drift = np.array(conditional_means) - prior
            predicted_drift = np.array([-price_variance / (1 - prior), price_variance / prior])
            assert abs(unconditional_mean - prior) < 2e-12
            np.testing.assert_allclose(actual_drift, predicted_drift, atol=2e-12, rtol=1e-11)
            assert actual_drift[0] < 0 < actual_drift[1]
            records.append({"initial_probability": prior, "increment_variance": variance,
                            "unconditional_price_mean": unconditional_mean,
                            "price_variance": price_variance,
                            "conditional_drifts_y0_y1": actual_drift.tolist(),
                            "maximum_identity_error": float(np.max(abs(actual_drift - predicted_drift)))})
    root = Path(__file__).resolve().parents[1]
    report = {"status": "passed", "quadrature_order": 192, "examples": records,
              "checker_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    destination = root / "revision" / "qa" / "martingale_benchmark_check.json"
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": "passed", "examples": len(records),
                      "max_error": max(r["maximum_identity_error"] for r in records)}))


if __name__ == "__main__":
    check()
