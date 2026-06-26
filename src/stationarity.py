"""Анализ стационарности, тренда и сезонности риск-факторов.

Закрывает требование описательной статистики: «тренд, сезонность, стационарность».

Идея блока:

1. ``stationarity_levels_vs_changes`` — для каждого базового ряда сравниваем
   стационарность УРОВНЯ и его ПРИРАЩЕНИЙ (для цен — лог-доходностей) двумя тестами:
   ADF (H0: единичный корень / нестационарность) и KPSS (H0: стационарность).
   Типичный результат: уровни — I(1) (нестационарны), приращения/доходности — I(0).
   Именно это оправдывает выбор риск-факторов как приращений/лог-доходностей.

2. ``trend_summary`` — линейный тренд уровня по времени (OLS), наклон и его значимость.

3. ``weekday_seasonality`` — проверка календарной сезонности доходностей по дням недели
   (тест Краскела-Уоллиса). Для дневных финансовых рядов сезонность обычно незначима.

4. ``serial_dependence`` — тест Льюнга-Бокса на доходностях и КВАДРАТАХ доходностей.
   Доходности почти некоррелированы, а их квадраты — сильно автокоррелированы
   (кластеризация волатильности). Это прямое обоснование условной модели волатильности
   (EWMA/GARCH) и тяжёлых хвостов из основной модели.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import acf, adfuller, kpss


def _adf_pvalue(x: np.ndarray, regression: str = "c") -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    try:
        res = adfuller(x, autolag="AIC", regression=regression)
        return float(res[0]), float(res[1])
    except Exception:
        return float("nan"), float("nan")


def _kpss_pvalue(x: np.ndarray, regression: str = "c") -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # KPSS клиппит p-value и предупреждает об этом
            res = kpss(x, regression=regression, nlags="auto")
        return float(res[0]), float(res[1])
    except Exception:
        return float("nan"), float("nan")


def stationarity_levels_vs_changes(
    df: pd.DataFrame,
    columns: list[str],
    log_return: bool,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Сравнить стационарность уровня и приращений каждого ряда (ADF + KPSS).

    log_return=True — приращение считается как лог-доходность (для цен/курсов/индексов);
    log_return=False — как простая разность (для процентных ставок).
    """
    rows = []
    for col in columns:
        if col not in df.columns:
            continue
        s = df[col].astype(float)
        level = s.to_numpy()
        change = (np.log(s).diff() if log_return else s.diff()).dropna().to_numpy()

        adf_l, adf_lp = _adf_pvalue(level)
        kpss_l, kpss_lp = _kpss_pvalue(level)
        adf_c, adf_cp = _adf_pvalue(change)
        kpss_c, kpss_cp = _kpss_pvalue(change)

        level_stationary = (adf_lp < alpha) and (kpss_lp > alpha)
        change_stationary = (adf_cp < alpha) and (kpss_cp > alpha)
        if level_stationary:
            order = "I(0)"
        elif change_stationary:
            order = "I(1)"
        else:
            order = "неоднозначно"

        rows.append({
            "factor": col,
            "change_type": "log-return" if log_return else "difference",
            "ADF_level_p": adf_lp,
            "KPSS_level_p": kpss_lp,
            "level_stationary": level_stationary,
            "ADF_change_p": adf_cp,
            "KPSS_change_p": kpss_cp,
            "change_stationary": change_stationary,
            "integration_order": order,
        })
    return pd.DataFrame(rows).set_index("factor")


def trend_summary(df: pd.DataFrame, columns: list[str], periods_per_year: int = 252) -> pd.DataFrame:
    """Линейный тренд уровня по времени (OLS): наклон, его значимость, R^2."""
    t = np.arange(len(df), dtype=float)
    rows = []
    for col in columns:
        if col not in df.columns:
            continue
        y = df[col].astype(float).to_numpy()
        mask = np.isfinite(y)
        if mask.sum() < 10:
            continue
        reg = stats.linregress(t[mask], y[mask])
        rows.append({
            "factor": col,
            "slope_per_day": reg.slope,
            "slope_per_year": reg.slope * periods_per_year,
            "t_stat": reg.slope / reg.stderr if reg.stderr else float("nan"),
            "p_value": reg.pvalue,
            "r_squared": reg.rvalue ** 2,
            "significant_trend_5pct": reg.pvalue < 0.05,
        })
    return pd.DataFrame(rows).set_index("factor")


def weekday_seasonality(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Календарная сезонность лог-доходностей по дням недели (Краскел-Уоллис)."""
    weekday = df["Date"].dt.dayofweek
    names = {0: "mean_mon", 1: "mean_tue", 2: "mean_wed", 3: "mean_thu", 4: "mean_fri"}
    rows = []
    for col in columns:
        if col not in df.columns:
            continue
        r = np.log(df[col].astype(float)).diff()
        row = {"factor": col}
        groups = []
        for d, name in names.items():
            vals = r[weekday == d].dropna()
            row[name] = float(vals.mean()) if len(vals) else float("nan")
            if len(vals) > 1:
                groups.append(vals.to_numpy())
        try:
            stat, p = stats.kruskal(*groups)
        except Exception:
            stat, p = float("nan"), float("nan")
        row["kruskal_stat"] = float(stat)
        row["kruskal_p_value"] = float(p)
        row["seasonality_5pct"] = bool(p < 0.05) if np.isfinite(p) else False
        rows.append(row)
    return pd.DataFrame(rows).set_index("factor")


def serial_dependence(df: pd.DataFrame, columns: list[str], lags: int = 10) -> pd.DataFrame:
    """Тест Льюнга-Бокса на доходностях и их квадратах (ARCH-эффект)."""
    rows = []
    for col in columns:
        if col not in df.columns:
            continue
        r = np.log(df[col].astype(float)).diff().dropna()
        try:
            lb_r = acorr_ljungbox(r, lags=[lags], return_df=True)["lb_pvalue"].iloc[0]
            lb_r2 = acorr_ljungbox(r ** 2, lags=[lags], return_df=True)["lb_pvalue"].iloc[0]
        except Exception:
            lb_r, lb_r2 = float("nan"), float("nan")
        rows.append({
            "factor": col,
            "lags": lags,
            "ljungbox_p_returns": float(lb_r),
            "ljungbox_p_squared_returns": float(lb_r2),
            "returns_autocorrelated_5pct": bool(lb_r < 0.05),
            "vol_clustering_5pct": bool(lb_r2 < 0.05),
        })
    return pd.DataFrame(rows).set_index("factor")


def acf_returns_and_squared(series: pd.Series, nlags: int = 20) -> pd.DataFrame:
    """ACF лог-доходностей и квадратов лог-доходностей (для иллюстрации кластеризации)."""
    r = np.log(series.astype(float)).diff().dropna().to_numpy()
    acf_r = acf(r, nlags=nlags, fft=True)
    acf_r2 = acf(r ** 2, nlags=nlags, fft=True)
    return pd.DataFrame({
        "lag": np.arange(nlags + 1),
        "acf_returns": acf_r,
        "acf_squared_returns": acf_r2,
    }).set_index("lag")
