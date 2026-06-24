from __future__ import annotations

import math

import numpy as np


def var_es(pnl: np.ndarray, var_level: float = 0.99, es_level: float = 0.975) -> dict[str, float]:
    pnl = np.asarray(pnl, dtype=float)
    var_threshold = np.quantile(pnl, 1 - var_level)
    es_threshold = np.quantile(pnl, 1 - es_level)
    tail = pnl[pnl <= es_threshold]
    return {
        f"VaR_{int(var_level*100)}": float(-var_threshold),
        f"ES_{es_level:.1%}": float(-tail.mean()) if len(tail) else float("nan"),
    }


def chi2_sf_df1(x: float) -> float:
    if not np.isfinite(x) or x < 0:
        return float("nan")
    return math.erfc(math.sqrt(x / 2))


def chi2_sf_df2(x: float) -> float:
    if not np.isfinite(x) or x < 0:
        return float("nan")
    return math.exp(-x / 2)


def kupiec_test(exceptions: np.ndarray, alpha: float = 0.01) -> dict[str, float]:
    exceptions = np.asarray(exceptions, dtype=bool)
    n = len(exceptions)
    x = int(exceptions.sum())
    if n == 0:
        return {"observations": 0, "exceptions": 0, "exception_rate": np.nan, "LR_uc": np.nan, "p_value": np.nan}
    phat = x / n
    if x == 0:
        lr = -2 * (n * math.log(1 - alpha) - n * math.log(1))
    elif x == n:
        lr = -2 * (n * math.log(alpha) - n * math.log(1))
    else:
        log_l0 = (n - x) * math.log(1 - alpha) + x * math.log(alpha)
        log_l1 = (n - x) * math.log(1 - phat) + x * math.log(phat)
        lr = -2 * (log_l0 - log_l1)
    return {"observations": n, "exceptions": x, "exception_rate": phat, "LR_uc": lr, "p_value": chi2_sf_df1(lr)}


def christoffersen_test(exceptions: np.ndarray, alpha: float = 0.01) -> dict[str, float]:
    exceptions = np.asarray(exceptions, dtype=int)
    if len(exceptions) < 2:
        return {"LR_ind": np.nan, "p_ind": np.nan, "LR_cc": np.nan, "p_cc": np.nan}
    n00 = n01 = n10 = n11 = 0
    for prev, curr in zip(exceptions[:-1], exceptions[1:]):
        if prev == 0 and curr == 0:
            n00 += 1
        elif prev == 0 and curr == 1:
            n01 += 1
        elif prev == 1 and curr == 0:
            n10 += 1
        else:
            n11 += 1

    def xlogp(n: int, p: float) -> float:
        if n == 0:
            return 0.0
        p = min(max(p, 1e-12), 1 - 1e-12)
        return n * math.log(p)

    pi = (n01 + n11) / max(1, n00 + n01 + n10 + n11)
    pi0 = n01 / max(1, n00 + n01)
    pi1 = n11 / max(1, n10 + n11)
    log_l0 = xlogp(n00 + n10, 1 - pi) + xlogp(n01 + n11, pi)
    log_l1 = xlogp(n00, 1 - pi0) + xlogp(n01, pi0) + xlogp(n10, 1 - pi1) + xlogp(n11, pi1)
    lr_ind = -2 * (log_l0 - log_l1)
    uc = kupiec_test(exceptions.astype(bool), alpha=alpha)
    lr_cc = uc["LR_uc"] + lr_ind
    return {
        "n00": n00,
        "n01": n01,
        "n10": n10,
        "n11": n11,
        "LR_ind": lr_ind,
        "p_ind": chi2_sf_df1(lr_ind),
        "LR_cc": lr_cc,
        "p_cc": chi2_sf_df2(lr_cc),
    }

