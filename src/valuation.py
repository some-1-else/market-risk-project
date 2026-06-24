from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import TENOR_YEARS


STOCK_NOTIONAL_RUB = 1_000_000.0
BOND_NOTIONAL_RUB = 10_000_000.0
FX_NOTIONAL_RUB = 100_000_000.0
BOND_NOMINAL = 1_000.0


@dataclass
class PortfolioPositions:
    as_of: pd.Timestamp
    stock_shares: dict[str, float]
    fx_units: dict[str, float]
    bond_units: dict[str, float]


def curve_from_row(row: pd.Series, rate_columns: list[str]) -> np.ndarray:
    return row[rate_columns].to_numpy(dtype=float)


def tenor_grid(rate_columns: list[str]) -> np.ndarray:
    return np.asarray([TENOR_YEARS[c] for c in rate_columns], dtype=float)


def bond_code(price_column: str) -> str:
    match = re.match(r"(\d{5})_Price", price_column)
    if not match:
        raise ValueError(f"Not a bond price column: {price_column}")
    return match.group(1)


def build_positions(row: pd.Series, stocks: list[str], fx: list[str], bonds: list[str]) -> PortfolioPositions:
    stock_shares = {col: STOCK_NOTIONAL_RUB / float(row[col]) for col in stocks}
    fx_units = {col: FX_NOTIONAL_RUB / float(row[col]) for col in fx}
    bond_units = {col: BOND_NOTIONAL_RUB / (float(row[col]) / 100 * BOND_NOMINAL) for col in bonds}
    return PortfolioPositions(as_of=pd.Timestamp(row["Date"]), stock_shares=stock_shares, fx_units=fx_units, bond_units=bond_units)


def future_cashflows(ofz: pd.DataFrame, code: str, as_of: pd.Timestamp) -> pd.DataFrame:
    cf = ofz[(ofz["code"] == code) & (ofz["payment_date"] > as_of)].copy()
    return cf.sort_values("payment_date")


def price_bond_dcf(
    ofz: pd.DataFrame,
    code: str,
    as_of: pd.Timestamp,
    curve_percent: np.ndarray,
    rate_columns: list[str],
) -> float:
    cf = future_cashflows(ofz, code, as_of)
    if cf.empty:
        return 0.0
    times = (cf["payment_date"] - as_of).dt.days.to_numpy(dtype=float) / 365.0
    times = np.maximum(times, 1 / 365)
    rates = np.interp(times, tenor_grid(rate_columns), curve_percent) / 100
    return float(np.sum(cf["payment_amount"].to_numpy(dtype=float) / np.power(1 + rates, times)))


def value_bonds_dcf(
    positions: PortfolioPositions,
    ofz: pd.DataFrame,
    curve_percent: np.ndarray,
    rate_columns: list[str],
) -> dict[str, float]:
    values = {}
    for col, units in positions.bond_units.items():
        values[col] = units * price_bond_dcf(ofz, bond_code(col), positions.as_of, curve_percent, rate_columns)
    return values


def component_values_market(row: pd.Series, positions: PortfolioPositions, stocks: list[str], fx: list[str], bonds: list[str]) -> dict[str, float]:
    stock_value = sum(positions.stock_shares[c] * float(row[c]) for c in stocks)
    fx_value = sum(positions.fx_units[c] * float(row[c]) for c in fx)
    bond_value = sum(positions.bond_units[c] * (float(row[c]) / 100 * BOND_NOMINAL) for c in bonds)
    return {"stocks": stock_value, "fx": fx_value, "bonds": bond_value, "total": stock_value + fx_value + bond_value}


def actual_pnl(next_row: pd.Series, base_row: pd.Series, positions: PortfolioPositions, stocks: list[str], fx: list[str], bonds: list[str]) -> dict[str, float]:
    base = component_values_market(base_row, positions, stocks, fx, bonds)
    nxt = component_values_market(next_row, positions, stocks, fx, bonds)
    return {k: nxt[k] - base[k] for k in base}


def bond_pricing_errors(
    market: pd.DataFrame,
    ofz: pd.DataFrame,
    as_of: pd.Timestamp,
    rate_columns: list[str],
    bond_columns: list[str],
) -> pd.DataFrame:
    row = market.loc[market["Date"] == as_of].iloc[0]
    curve = curve_from_row(row, rate_columns)
    rows = []
    for col in bond_columns:
        code = bond_code(col)
        model_price = price_bond_dcf(ofz, code, as_of, curve, rate_columns)
        market_price = float(row[col]) / 100 * BOND_NOMINAL
        rows.append({
            "bond": code,
            "market_quote_pct": float(row[col]),
            "market_price_rub": market_price,
            "model_price_rub": model_price,
            "error_rub": model_price - market_price,
            "error_pct_of_market": (model_price / market_price - 1) if market_price else np.nan,
        })
    return pd.DataFrame(rows).set_index("bond")

