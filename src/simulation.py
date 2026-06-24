from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .risk_factors import PCAResult, build_model_factors
from .valuation import (
    build_positions,
    component_values_market,
    curve_from_row,
    future_cashflows,
    tenor_grid,
)


@dataclass
class RiskModel:
    factors: pd.DataFrame
    pca: PCAResult
    mean: np.ndarray
    covariance: np.ndarray
    factor_columns: list[str]
    asset_columns: list[str]
    rate_columns: list[str]
    stock_columns: list[str]
    fx_columns: list[str]
    bond_columns: list[str]


def fit_normal_model(
    market: pd.DataFrame,
    as_of: pd.Timestamp,
    stock_columns: list[str],
    fx_columns: list[str],
    rate_columns: list[str],
    bond_columns: list[str],
    n_components: int = 3,
) -> RiskModel:
    train = market[market["Date"] <= as_of].copy()
    asset_columns = stock_columns + fx_columns
    factors, pca = build_model_factors(train, asset_columns, rate_columns, n_components=n_components)
    numeric = factors.drop(columns=["Date"])
    mean = numeric.mean().to_numpy(dtype=float)
    cov = numeric.cov().to_numpy(dtype=float)
    cov = cov + np.eye(cov.shape[0]) * 1e-12
    return RiskModel(
        factors=factors,
        pca=pca,
        mean=mean,
        covariance=cov,
        factor_columns=list(numeric.columns),
        asset_columns=asset_columns,
        rate_columns=rate_columns,
        stock_columns=stock_columns,
        fx_columns=fx_columns,
        bond_columns=bond_columns,
    )


def _scenario_bond_values(
    ofz: pd.DataFrame,
    positions,
    rate_curves: np.ndarray,
    rate_columns: list[str],
) -> np.ndarray:
    values = np.zeros(len(rate_curves))
    tenors = tenor_grid(rate_columns)
    for col, units in positions.bond_units.items():
        code = col.split("_")[0]
        cf = future_cashflows(ofz, code, positions.as_of)
        if cf.empty:
            continue
        times = (cf["payment_date"] - positions.as_of).dt.days.to_numpy(dtype=float) / 365.0
        times = np.maximum(times, 1 / 365)
        amounts = cf["payment_amount"].to_numpy(dtype=float)
        rates = np.empty((len(rate_curves), len(times)))
        for j, time in enumerate(times):
            upper = np.searchsorted(tenors, time, side="left")
            if upper <= 0:
                rates[:, j] = rate_curves[:, 0]
            elif upper >= len(tenors):
                rates[:, j] = rate_curves[:, -1]
            else:
                lower = upper - 1
                weight = (time - tenors[lower]) / (tenors[upper] - tenors[lower])
                rates[:, j] = (1 - weight) * rate_curves[:, lower] + weight * rate_curves[:, upper]
        discount = np.power(1 + rates / 100, times)
        pv = (amounts / discount).sum(axis=1)
        values += units * pv
    return values


def simulate_portfolio_pnl(
    model: RiskModel,
    market: pd.DataFrame,
    ofz: pd.DataFrame,
    as_of: pd.Timestamp,
    horizon_days: int = 1,
    n_scenarios: int = 10_000,
    seed: int = 42,
) -> tuple[pd.DataFrame, dict[str, float]]:
    row = market.loc[market["Date"] == as_of].iloc[0]
    positions = build_positions(row, model.stock_columns, model.fx_columns, model.bond_columns)
    base_market = component_values_market(row, positions, model.stock_columns, model.fx_columns, model.bond_columns)
    base_curve = curve_from_row(row, model.rate_columns)
    base_bond_dcf = _scenario_bond_values(ofz, positions, np.array([base_curve]), model.rate_columns)[0]

    rng = np.random.default_rng(seed + horizon_days)
    mean = model.mean * horizon_days
    covariance = model.covariance * horizon_days
    draws = rng.multivariate_normal(mean, covariance, size=n_scenarios, method="svd")
    draw_df = pd.DataFrame(draws, columns=model.factor_columns)

    stock_values = np.zeros(n_scenarios)
    for col in model.stock_columns:
        stock_values += positions.stock_shares[col] * float(row[col]) * np.exp(draw_df[col].to_numpy())

    fx_values = np.zeros(n_scenarios)
    for col in model.fx_columns:
        fx_values += positions.fx_units[col] * float(row[col]) * np.exp(draw_df[col].to_numpy())

    pc_cols = [c for c in model.factor_columns if c.startswith("PC")]
    pc_draws = draw_df[pc_cols].to_numpy(dtype=float)
    rate_delta = model.pca.mean * horizon_days + pc_draws @ model.pca.components
    rate_curves = np.clip(base_curve + rate_delta, 0.01, 50.0)
    bond_values = _scenario_bond_values(ofz, positions, rate_curves, model.rate_columns)

    pnl = pd.DataFrame({
        "stocks": stock_values - base_market["stocks"],
        "fx": fx_values - base_market["fx"],
        "bonds": bond_values - base_bond_dcf,
    })
    pnl["total"] = pnl["stocks"] + pnl["fx"] + pnl["bonds"]
    base_values = {
        "stocks": base_market["stocks"],
        "fx": base_market["fx"],
        "bonds_market": base_market["bonds"],
        "bonds_dcf": base_bond_dcf,
        "total_market": base_market["total"],
    }
    return pnl, base_values
