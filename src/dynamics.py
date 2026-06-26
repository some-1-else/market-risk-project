"""Stochastic dynamics models for risk factors with maximum-likelihood estimation.

Этот модуль закрывает требование задания (п.3): для риск-факторов выбрать
стохастическую модель динамики, обосновать выбор и оценить параметры методом
максимального правдоподобия (MLE).

Что здесь реализовано:

- ``fit_normal_mle``            — MLE для нормального распределения (baseline);
- ``fit_student_t_mle``        — MLE для распределения Стьюдента (тяжёлые хвосты);
- ``fit_garch11``              — GARCH(1,1), MLE через scipy (нормальные/t остатки);
- ``ewma_vol``                 — RiskMetrics EWMA условная волатильность (lambda=0.94);
- ``mv_t_standardized``        — генерация многомерного t с единичной дисперсией компонент;
- ``select_distribution``      — таблица выбора модели Normal vs Student-t по AIC;
- ``estimate_mvt_dof``         — MLE числа степеней свободы многомерного t.

Ключевая идея для оценки риска: волатильность непостоянна во времени. Если оценивать
ковариацию факторов по всей истории 2021-2025 (включая шок февраля-марта 2022 г.),
то оценка дневного риска на спокойный декабрь 2025 г. оказывается сильно завышенной.
Поэтому мы переходим к условной волатильности (EWMA/GARCH), которая отражает текущий
режим рынка, а зависимость факторов и тяжесть хвостов задаём многомерным t-Стьюдента.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import optimize, stats
from scipy.special import gammaln


# ---------------------------------------------------------------------------
# Безусловные распределения (для модельного выбора и описания хвостов)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FitResult:
    name: str
    params: dict[str, float]
    loglik: float
    n_params: int
    n_obs: int

    @property
    def aic(self) -> float:
        return 2 * self.n_params - 2 * self.loglik

    @property
    def bic(self) -> float:
        return self.n_params * np.log(self.n_obs) - 2 * self.loglik


def fit_normal_mle(x: np.ndarray) -> FitResult:
    """MLE нормального распределения.

    Для нормального распределения MLE среднего и дисперсии совпадают с выборочными
    моментами (дисперсия со смещением, делитель n), поэтому здесь это явно так
    и считается, а не берётся «на глаз».
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    mu = float(x.mean())
    sigma = float(np.sqrt(((x - mu) ** 2).mean()))
    loglik = float(np.sum(stats.norm.logpdf(x, loc=mu, scale=sigma)))
    return FitResult("normal", {"mu": mu, "sigma": sigma}, loglik, 2, n)


def fit_student_t_mle(x: np.ndarray) -> FitResult:
    """MLE распределения Стьюдента (location-scale) с оценкой числа степеней свободы."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    df, loc, scale = stats.t.fit(x)
    loglik = float(np.sum(stats.t.logpdf(x, df, loc=loc, scale=scale)))
    return FitResult("student_t", {"df": float(df), "loc": float(loc), "scale": float(scale)}, loglik, 3, n)


def select_distribution(x: np.ndarray) -> dict[str, float]:
    """Сравнить Normal и Student-t по логарифму правдоподобия / AIC."""
    normal = fit_normal_mle(x)
    student = fit_student_t_mle(x)
    best = "student_t" if student.aic < normal.aic else "normal"
    return {
        "normal_loglik": normal.loglik,
        "normal_aic": normal.aic,
        "student_df": student.params["df"],
        "student_loglik": student.loglik,
        "student_aic": student.aic,
        "aic_gain_t_over_normal": normal.aic - student.aic,
        "selected": best,
    }


# ---------------------------------------------------------------------------
# Условная волатильность: EWMA и GARCH(1,1) (MLE)
# ---------------------------------------------------------------------------

def ewma_vol(x: np.ndarray, lam: float = 0.94) -> tuple[np.ndarray, float]:
    """RiskMetrics EWMA условная дисперсия.

    Возвращает (sigma_t по истории, прогноз sigma на следующий день).
    EWMA — это GARCH(1,1) с omega=0 и alpha+beta=1, классический выбор RiskMetrics.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    mu = x.mean()
    eps2 = (x - mu) ** 2
    var = np.empty_like(eps2)
    var[0] = eps2.mean()
    for t in range(1, len(eps2)):
        var[t] = lam * var[t - 1] + (1 - lam) * eps2[t - 1]
    sigma = np.sqrt(var)
    sigma_next = float(np.sqrt(lam * var[-1] + (1 - lam) * eps2[-1]))
    return sigma, sigma_next


@dataclass(frozen=True)
class GarchFit:
    omega: float
    alpha: float
    beta: float
    nu: float | None
    mu: float
    sigma: np.ndarray          # условная sigma_t по истории
    sigma_next: float          # прогноз sigma на следующий день
    loglik: float
    n_params: int
    n_obs: int
    dist: str
    converged: bool

    @property
    def persistence(self) -> float:
        return self.alpha + self.beta

    @property
    def aic(self) -> float:
        return 2 * self.n_params - 2 * self.loglik


def _garch_recursion(eps: np.ndarray, omega: float, alpha: float, beta: float) -> np.ndarray:
    var = np.empty_like(eps)
    var[0] = eps.var() if eps.var() > 0 else 1e-10
    for t in range(1, len(eps)):
        var[t] = omega + alpha * eps[t - 1] ** 2 + beta * var[t - 1]
    return var


def fit_garch11(x: np.ndarray, dist: str = "normal") -> GarchFit:
    """GARCH(1,1) методом максимального правдоподобия.

    sigma_t^2 = omega + alpha * eps_{t-1}^2 + beta * sigma_{t-1}^2,
    eps_t = x_t - mu. Остатки нормальные (dist='normal') или Стьюдента (dist='t').
    Оптимизация — численная максимизация лог-правдоподобия (scipy Nelder-Mead).
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    mu = float(x.mean())
    eps = x - mu
    uncond_var = float(eps.var())
    if uncond_var <= 0:
        uncond_var = 1e-10

    def neg_loglik(theta: np.ndarray) -> float:
        if dist == "t":
            log_omega, logit_a, logit_b, log_nu_m2 = theta
            nu = 2.0 + np.exp(log_nu_m2)
        else:
            log_omega, logit_a, logit_b = theta
            nu = None
        omega = np.exp(log_omega)
        ea, eb = np.exp(logit_a), np.exp(logit_b)
        denom = 1.0 + ea + eb
        alpha = ea / denom
        beta = eb / denom
        var = _garch_recursion(eps, omega, alpha, beta)
        if np.any(var <= 0) or not np.all(np.isfinite(var)):
            return 1e12
        if dist == "t":
            scale = np.sqrt(var * (nu - 2) / nu)
            ll = np.sum(stats.t.logpdf(eps, nu, loc=0.0, scale=scale))
        else:
            ll = np.sum(stats.norm.logpdf(eps, loc=0.0, scale=np.sqrt(var)))
        if not np.isfinite(ll):
            return 1e12
        return -ll

    a0, b0 = 0.08, 0.90
    log_o0 = np.log(max(uncond_var * (1 - a0 - b0), 1e-12))
    la0 = np.log(a0 / (1 - a0 - b0))
    lb0 = np.log(b0 / (1 - a0 - b0))
    x0 = [log_o0, la0, lb0] + ([np.log(6.0)] if dist == "t" else [])

    res = optimize.minimize(neg_loglik, x0, method="Nelder-Mead",
                            options={"maxiter": 4000, "xatol": 1e-8, "fatol": 1e-8})
    theta = res.x
    omega = float(np.exp(theta[0])); ea, eb = np.exp(theta[1]), np.exp(theta[2])
    nu = float(2.0 + np.exp(theta[3])) if dist == "t" else None
    denom = 1.0 + ea + eb
    alpha = float(ea / denom)
    beta = float(eb / denom)
    var = _garch_recursion(eps, omega, alpha, beta)
    sigma = np.sqrt(var)
    var_next = omega + alpha * eps[-1] ** 2 + beta * var[-1]
    sigma_next = float(np.sqrt(var_next))
    loglik = float(-res.fun)
    n_params = 4 if dist == "t" else 3
    return GarchFit(omega, alpha, beta, nu, mu, sigma, sigma_next,
                    loglik, n_params, n, dist, bool(res.success))


# ---------------------------------------------------------------------------
# Многомерные инновации: t-Стьюдента с единичной дисперсией компонент
# ---------------------------------------------------------------------------

def _nearest_corr(corr: np.ndarray) -> np.ndarray:
    """Ближайшая валидная корреляционная матрица (клиппинг собственных значений)."""
    corr = (corr + corr.T) / 2
    vals, vecs = np.linalg.eigh(corr)
    vals = np.clip(vals, 1e-8, None)
    fixed = vecs @ np.diag(vals) @ vecs.T
    d = np.sqrt(np.clip(np.diag(fixed), 1e-12, None))
    fixed = fixed / np.outer(d, d)
    np.fill_diagonal(fixed, 1.0)
    return fixed


def estimate_mvt_dof(standardized: np.ndarray, dof_grid: np.ndarray | None = None) -> float:
    """Оценка общего числа степеней свободы многомерного t по стандартизованным остаткам.

    standardized: (T, k) остатки с примерно нулевым средним и единичной дисперсией.
    Профильное правдоподобие по nu при корреляции, оценённой из данных.
    """
    z = np.asarray(standardized, dtype=float)
    z = z[np.all(np.isfinite(z), axis=1)]
    T, k = z.shape
    corr = _nearest_corr(np.corrcoef(z, rowvar=False))
    inv = np.linalg.pinv(corr)
    sign, logdet = np.linalg.slogdet(corr)
    quad = np.einsum("ti,ij,tj->t", z, inv, z)  # mahalanobis^2

    if dof_grid is None:
        dof_grid = np.concatenate([np.arange(3, 30, 1.0), np.arange(30, 105, 5.0)])

    best_nu, best_ll = float(dof_grid[-1]), -np.inf
    for nu in dof_grid:
        scale = (nu - 2) / nu  # shape = scale*corr, чтобы Cov = corr
        logdet_S = k * np.log(scale) + logdet
        q = quad / scale
        ll = (T * (gammaln((nu + k) / 2) - gammaln(nu / 2)
                   - 0.5 * k * np.log(nu * np.pi) - 0.5 * logdet_S)
              - (nu + k) / 2 * np.sum(np.log1p(q / nu)))
        if ll > best_ll:
            best_ll, best_nu = ll, float(nu)
    return best_nu


def mv_t_standardized(rng: np.random.Generator, corr: np.ndarray, nu: float, size: int) -> np.ndarray:
    """Выборка из многомерного t с корреляцией ``corr`` и единичной дисперсией компонент.

    Стандартное многомерное t имеет Cov = nu/(nu-2)*shape. Чтобы дисперсия каждой
    компоненты была равна 1 (а корреляция = corr), берём shape = corr*(nu-2)/nu.
    При nu=inf вырождается в многомерное нормальное с корреляцией corr.
    """
    k = corr.shape[0]
    corr = _nearest_corr(corr)
    if not np.isfinite(nu) or nu > 200:
        L = np.linalg.cholesky(corr)
        z = rng.standard_normal((size, k))
        return z @ L.T
    shape = corr * (nu - 2) / nu
    L = np.linalg.cholesky(shape)
    z = rng.standard_normal((size, k)) @ L.T
    g = rng.chisquare(nu, size=size) / nu
    return z / np.sqrt(g)[:, None]
