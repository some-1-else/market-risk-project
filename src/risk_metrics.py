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



# ===========================================================================
# Дополнительные тесты валидации VaR из списка задания.
# ===========================================================================


def dq_test(exceptions: np.ndarray, var: np.ndarray, alpha: float = 0.01, lags: int = 4) -> dict[str, float]:
    """Dynamic Quantile test (Engle & Manganelli, 2004).

    Регрессия центрированного индикатора пробоя Hit_t = I_t - alpha на константу,
    лагированные Hit_{t-1..t-p} и текущий VaR_t. При корректной условной модели
    все коэффициенты равны нулю. Статистика DQ = (объяснённая сумма квадратов) /
    (alpha*(1-alpha)) ~ chi2(k), где k — число регрессоров. Тест инвариантен к
    линейному масштабированию регрессоров (используется проекция), поэтому масштаб
    VaR в рублях значения не имеет.
    """
    from scipy import stats as sps

    I = np.asarray(exceptions, dtype=float)
    var = np.asarray(var, dtype=float)
    n = len(I)
    hit = I - alpha
    if n <= lags + 2:
        return {"DQ_stat": float("nan"), "DQ_p_value": float("nan"), "DQ_lags": lags}

    rows, y = [], []
    for t in range(lags, n):
        reg = [1.0] + [hit[t - l] for l in range(1, lags + 1)] + [var[t]]
        rows.append(reg)
        y.append(hit[t])
    X = np.asarray(rows, dtype=float)
    Y = np.asarray(y, dtype=float)
    # объяснённая сумма квадратов = ||проекция Y на столбцы X||^2
    beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
    fitted = X @ beta
    stat = float(fitted @ fitted / (alpha * (1 - alpha)))
    k = X.shape[1]
    p = float(sps.chi2.sf(stat, k))
    return {"DQ_stat": stat, "DQ_p_value": p, "DQ_lags": lags}


def duration_test(exceptions: np.ndarray, alpha: float = 0.01) -> dict[str, float]:
    """Duration-based test (Christoffersen & Pelletier, 2004).

    При корректном VaR пробои образуют пуассоновский поток, а интервалы между ними
    (durations) не имеют памяти — распределены экспоненциально (Weibull с shape b=1).
    Гипотеза независимости проверяется через LR: Weibull(b) против экспоненты (b=1).
    Первый и последний интервалы цензурированы. Малое число пробоев => низкая мощность.
    """
    from scipy import optimize, stats as sps

    I = np.asarray(exceptions, dtype=bool)
    n = len(I)
    idx = np.where(I)[0]
    n_exc = len(idx)
    if n_exc < 2:
        return {"duration_LR": float("nan"), "duration_p_value": float("nan"),
                "duration_weibull_b": float("nan"), "n_durations": n_exc}

    # длительности между пробоями + цензурированные хвосты (в торговых днях)
    full = np.diff(idx).astype(float)              # полностью наблюдаемые
    cens = []
    if idx[0] > 0:
        cens.append(float(idx[0]))                 # от начала до первого пробоя
    if idx[-1] < n - 1:
        cens.append(float(n - 1 - idx[-1]))        # от последнего пробоя до конца
    cens = np.asarray(cens, dtype=float)
    d_unc = np.maximum(full, 1.0)
    d_cen = np.maximum(cens, 1.0) if len(cens) else np.array([])

    def neg_ll_weibull(theta):
        a, b = theta
        if a <= 0 or b <= 0:
            return 1e12
        ll = 0.0
        if len(d_unc):
            ll += np.sum(np.log(b) + b * np.log(a) + (b - 1) * np.log(d_unc) - (a * d_unc) ** b)
        if len(d_cen):
            ll += np.sum(-((a * d_cen) ** b))
        return -ll if np.isfinite(ll) else 1e12

    # экспонента (b=1): a по MLE с цензурой = (#полных) / (сумма всех длительностей)
    total_time = d_unc.sum() + (d_cen.sum() if len(d_cen) else 0.0)
    a_exp = len(d_unc) / total_time if total_time > 0 else alpha
    ll_exp = -neg_ll_weibull((a_exp, 1.0))

    res = optimize.minimize(neg_ll_weibull, x0=[a_exp, 1.0], method="Nelder-Mead",
                            options={"maxiter": 2000, "xatol": 1e-8, "fatol": 1e-8})
    ll_weib = -res.fun
    b_hat = float(res.x[1])
    lr = float(2 * (ll_weib - ll_exp))
    lr = max(lr, 0.0)
    p = float(sps.chi2.sf(lr, 1))
    return {"duration_LR": lr, "duration_p_value": p, "duration_weibull_b": b_hat, "n_durations": len(d_unc)}
