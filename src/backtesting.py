from __future__ import annotations

import numpy as np
import pandas as pd

from .risk_metrics import christoffersen_test, kupiec_test, var_es
from .simulation import fit_normal_model, simulate_portfolio_pnl
from .valuation import actual_pnl, build_positions


def run_backtest(
    market: pd.DataFrame,
    ofz: pd.DataFrame,
    stock_columns: list[str],
    fx_columns: list[str],
    rate_columns: list[str],
    bond_columns: list[str],
    year: int = 2025,
    n_scenarios: int = 3_000,
    seed: int = 2025,
    min_history: int = 250,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = market["Date"].tolist()
    rows = []
    for i, as_of in enumerate(dates[:-1]):
        if as_of.year != year or i < min_history:
            continue
        next_row = market.iloc[i + 1]
        base_row = market.iloc[i]
        model = fit_normal_model(market, as_of, stock_columns, fx_columns, rate_columns, bond_columns)
        pnl_sim, _ = simulate_portfolio_pnl(model, market, ofz, as_of, horizon_days=1, n_scenarios=n_scenarios, seed=seed + i)
        positions = build_positions(base_row, stock_columns, fx_columns, bond_columns)
        realized = actual_pnl(next_row, base_row, positions, stock_columns, fx_columns, bond_columns)
        row = {"Date": as_of, "next_date": next_row["Date"]}
        for component in ["stocks", "bonds", "fx", "total"]:
            metric = var_es(pnl_sim[component].to_numpy(), var_level=0.99, es_level=0.975)
            var99 = metric["VaR_99"]
            row[f"{component}_VaR99"] = var99
            row[f"{component}_pnl"] = realized[component]
            row[f"{component}_exception"] = realized[component] < -var99
        rows.append(row)

    details = pd.DataFrame(rows)
    summary_rows = []
    for component in ["stocks", "bonds", "fx", "total"]:
        exceptions = details[f"{component}_exception"].to_numpy(dtype=bool)
        kupiec = kupiec_test(exceptions, alpha=0.01)
        christoffersen = christoffersen_test(exceptions, alpha=0.01)
        summary_rows.append({"component": component, **kupiec, **christoffersen})
    summary = pd.DataFrame(summary_rows).set_index("component")
    return details, summary

