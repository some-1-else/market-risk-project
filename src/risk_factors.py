from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class PCAResult:
    mean: np.ndarray
    components: np.ndarray
    explained_variance: np.ndarray
    explained_ratio: np.ndarray
    scores: pd.DataFrame
    component_table: pd.DataFrame


def log_returns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = np.log(df[columns]).diff()
    out.insert(0, "Date", df["Date"])
    return out.dropna().reset_index(drop=True)


def rate_changes(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df[columns].diff()
    out.insert(0, "Date", df["Date"])
    return out.dropna().reset_index(drop=True)


def compute_pca(changes: pd.DataFrame, rate_columns: list[str], n_components: int = 3) -> PCAResult:
    x = changes[rate_columns].to_numpy(dtype=float)
    mean = np.nanmean(x, axis=0)
    x = np.nan_to_num(x - mean, nan=0.0)
    cov = np.cov(x, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    components = eigvecs[:, :n_components].T
    explained = eigvals[:n_components]
    ratio = eigvals / eigvals.sum()
    scores = x @ components.T
    score_df = pd.DataFrame(scores, columns=[f"PC{i+1}" for i in range(n_components)])
    score_df.insert(0, "Date", changes["Date"].values)
    component_table = pd.DataFrame(components.T, index=rate_columns, columns=[f"PC{i+1}" for i in range(n_components)])
    return PCAResult(mean=mean, components=components, explained_variance=explained, explained_ratio=ratio[:n_components], scores=score_df, component_table=component_table)


def build_model_factors(
    market: pd.DataFrame,
    asset_columns: list[str],
    rate_columns: list[str],
    n_components: int = 3,
) -> tuple[pd.DataFrame, PCAResult]:
    rets = log_returns(market, asset_columns)
    changes = rate_changes(market, rate_columns)
    pca = compute_pca(changes, rate_columns, n_components=n_components)
    factors = rets.merge(pca.scores, on="Date", how="inner")
    return factors, pca


def descriptive_stats(factors: pd.DataFrame) -> pd.DataFrame:
    numeric = factors.drop(columns=["Date"], errors="ignore")
    rows = []
    for col in numeric.columns:
        x = numeric[col].dropna().to_numpy(dtype=float)
        mean = x.mean()
        std = x.std(ddof=1)
        centered = x - mean
        skew = np.mean(centered**3) / std**3 if std > 0 else np.nan
        kurtosis_excess = np.mean(centered**4) / std**4 - 3 if std > 0 else np.nan
        q = np.quantile(x, [0.01, 0.025, 0.05, 0.5, 0.95, 0.975, 0.99])
        rows.append({
            "factor": col,
            "mean": mean,
            "std": std,
            "skewness": skew,
            "excess_kurtosis": kurtosis_excess,
            "min": x.min(),
            "q01": q[0],
            "q025": q[1],
            "q05": q[2],
            "median": q[3],
            "q95": q[4],
            "q975": q[5],
            "q99": q[6],
            "max": x.max(),
            "heavy_tail_flag": bool(kurtosis_excess > 1.0),
        })
    return pd.DataFrame(rows).set_index("factor")


def adf_summary(factors: pd.DataFrame) -> pd.DataFrame:
    try:
        from statsmodels.tsa.stattools import adfuller
    except Exception:
        return pd.DataFrame({
            "factor": [c for c in factors.columns if c != "Date"],
            "adf_stat": np.nan,
            "p_value": np.nan,
            "note": "statsmodels unavailable in current runtime",
        }).set_index("factor")

    rows = []
    for col in factors.columns:
        if col == "Date":
            continue
        x = factors[col].dropna()
        try:
            result = adfuller(x, autolag="AIC")
            rows.append({"factor": col, "adf_stat": result[0], "p_value": result[1], "note": ""})
        except Exception as exc:
            rows.append({"factor": col, "adf_stat": np.nan, "p_value": np.nan, "note": str(exc)})
    return pd.DataFrame(rows).set_index("factor")


def correlation_matrix(factors: pd.DataFrame) -> pd.DataFrame:
    return factors.drop(columns=["Date"], errors="ignore").corr()

