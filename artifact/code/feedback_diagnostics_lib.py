"""Feature map for the matched endpoint logistic E_log (frozen 18 Nov 2026)."""
import numpy as np
from scipy.special import logit


def endpoint_design(prices, volumes, asset, eps):
    """x_0, x_T, mean log(1+v), least-squares slope of log(1+v) on centred time, asset."""
    x = logit(np.clip(prices, eps, 1 - eps))
    lv = np.log1p(volumes)
    t = np.arange(lv.shape[1], dtype=float)
    t -= t.mean()
    return np.column_stack([x[:, 0], x[:, -1], lv.mean(1), lv @ t / (t @ t), asset])
