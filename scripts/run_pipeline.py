from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.backtesting import run_backtest
from src.data import (
    classify_columns,
    load_market_data,
    load_ofz_cashflows,
    previous_trading_date,
    repair_price_spikes,
    save_processed,
)
from src.dynamics import fit_garch11, select_distribution
from src.risk_factors import adf_summary, build_model_factors, correlation_matrix, descriptive_stats
from src.risk_metrics import var_es
from src.simulation import (
    fit_dynamics_model,
    fit_normal_model,
    simulate_portfolio_pnl,
    simulate_portfolio_pnl_cond,
)
from src.utils import (
    FIGURES_DIR,
    OUTPUTS_DIR,
    PROCESSED_DIR,
    TABLES_DIR,
    ensure_dirs,
    save_backtest_svg,
    save_bar_svg,
    save_heatmap_svg,
    save_hist_svg,
    save_line_svg,
    save_table,
)
from src.valuation import BOND_NOTIONAL_RUB, FX_NOTIONAL_RUB, STOCK_NOTIONAL_RUB, bond_pricing_errors, build_positions


RISK_DATE = "2025-12-02"
FINAL_SCENARIOS = 10_000
BACKTEST_SCENARIOS = 3_000
SEED = 42


def make_risk_factor_inventory(catalog) -> pd.DataFrame:
    rows = []
    for col in catalog.stocks:
        rows.append({"factor": col, "group": "Акции", "source_columns": col, "transformation": "лог-доходность", "used_in_model": "да", "comment": "цена акции"})
    for col in catalog.fx:
        rows.append({"factor": col, "group": "Валюта", "source_columns": col, "transformation": "лог-доходность курса RUB", "used_in_model": "да", "comment": "USD/RUB или EUR/RUB"})
    rows.append({"factor": "PC1-PC3", "group": "Процентная кривая", "source_columns": ", ".join(catalog.rates), "transformation": "дневные изменения ставок + PCA", "used_in_model": "да", "comment": "первые 3 компоненты кривой"})
    for col in catalog.bond_prices:
        rows.append({"factor": col, "group": "ОФЗ", "source_columns": col, "transformation": "рыночная цена для фактического P&L", "used_in_model": "для backtesting", "comment": "модельный риск идет через кривую"})
    for col in catalog.aux_prices:
        rows.append({"factor": col, "group": "Рыночный индикатор", "source_columns": col, "transformation": "лог-доходность", "used_in_model": "аналитика", "comment": "для статистики и корреляций"})
    return pd.DataFrame(rows).set_index("factor")


def make_portfolio_composition(market: pd.DataFrame, as_of: pd.Timestamp, catalog) -> pd.DataFrame:
    row = market.loc[market["Date"] == as_of].iloc[0]
    positions = build_positions(row, catalog.stocks, catalog.fx, catalog.bond_prices)
    rows = []
    for col in catalog.stocks:
        rows.append({
            "instrument": col,
            "bucket": "Акции",
            "price_or_rate": float(row[col]),
            "quantity": positions.stock_shares[col],
            "market_value_rub": STOCK_NOTIONAL_RUB,
            "comment": "количество рассчитано как 1 млн руб. / цена",
        })
    for col in catalog.bond_prices:
        rows.append({
            "instrument": col.replace("_Price", ""),
            "bucket": "ОФЗ",
            "price_or_rate": float(row[col]),
            "quantity": positions.bond_units[col],
            "market_value_rub": BOND_NOTIONAL_RUB,
            "comment": "цена в % от номинала 1000 руб.",
        })
    for col in catalog.fx:
        rows.append({
            "instrument": col,
            "bucket": "Валюта",
            "price_or_rate": float(row[col]),
            "quantity": positions.fx_units[col],
            "market_value_rub": FX_NOTIONAL_RUB,
            "comment": "валютная сумма рассчитана как 100 млн руб. / курс",
        })
    return pd.DataFrame(rows).set_index("instrument")


def make_limitations_table() -> pd.DataFrame:
    rows = [
        ("Условная волатильность EWMA", "lambda=0.94 фиксирован (RiskMetrics), а не оценивается; реакция на смену режима с лагом", "Оценивать lambda по MLE или перейти к полноценному GARCH(1,1)"),
        ("Многомерное t-Стьюдента", "Один общий параметр степеней свободы на все факторы; хвостовая зависимость симметрична", "Перейти к копулам (t-копула по факторам, разные dof) или EVT для хвостов"),
        ("Процентная кривая", "PCA строится по историческим изменениям ставок и не моделирует отдельные спреды ОФЗ", "Добавить спредовые факторы по выпускам"),
        ("ОФЗ DCF", "Нет полноценного clean/dirty price, НКД, bid/ask и календарей купонов вне данных", "Использовать полные bond analytics и day-count conventions"),
        ("Горизонт 10 дней", "Считается суммой 10 i.i.d. дневных инноваций при замороженной условной sigma; игнорируется возврат волатильности к среднему", "Симулировать траекторию условной волатильности на горизонте"),
        ("Backtesting", "Один год = 260 наблюдений; для 1% VaR ожидается ~2-3 пробоя, мощность тестов ограничена", "Расширить период/уровни confidence и добавить duration-тесты"),
        ("Ликвидность и транзакционные издержки", "Не учитываются", "Добавить liquidity add-on"),
    ]
    return pd.DataFrame(rows, columns=["limitation", "impact", "defense_comment"]).set_index("limitation")


def make_distribution_selection(factors: pd.DataFrame) -> pd.DataFrame:
    """Сравнение Normal vs Student-t (MLE) по каждому риск-фактору."""
    rows = []
    for col in factors.columns:
        if col == "Date":
            continue
        sel = select_distribution(factors[col].to_numpy())
        rows.append({"factor": col, **sel})
    return pd.DataFrame(rows).set_index("factor")


def make_garch_estimates(factors: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """GARCH(1,1)-t оценки методом максимального правдоподобия для ключевых факторов."""
    rows = []
    for col in columns:
        if col not in factors.columns:
            continue
        g = fit_garch11(factors[col].to_numpy(), dist="t")
        rows.append({
            "factor": col,
            "omega": g.omega,
            "alpha": g.alpha,
            "beta": g.beta,
            "persistence_alpha_plus_beta": g.persistence,
            "nu_innovations": g.nu,
            "loglik": g.loglik,
            "aic": g.aic,
            "converged": g.converged,
        })
    return pd.DataFrame(rows).set_index("factor")


def main() -> None:
    ensure_dirs()
    market = load_market_data(ROOT / "df_final.csv")
    ofz = load_ofz_cashflows(ROOT / "ofz_final.csv")
    catalog = classify_columns(market)

    # Чистка очевидных одиночных выбросов в ценовых рядах (например, ошибочный
    # курс USD=5.00 на 2023-03-16). Отчёт о заменах сохраняется для прозрачности.
    repair_cols = catalog.fx + catalog.stocks + catalog.bond_prices + catalog.aux_prices
    market, repair_report = repair_price_spikes(market, repair_cols)
    if repair_report.empty:
        repair_report = pd.DataFrame(columns=["column", "date", "old_value", "prev_value", "next_value", "repaired_value"])
    save_table(repair_report.set_index("column") if not repair_report.empty else repair_report,
               TABLES_DIR / "data_repair_report.csv", TABLES_DIR / "data_repair_report.md")
    save_processed(market, ofz, PROCESSED_DIR)

    data_profile = pd.DataFrame({
        "dataset": ["df_final.csv", "ofz_final.csv"],
        "rows": [len(market), len(ofz)],
        "columns": [market.shape[1], ofz.shape[1]],
        "date_min": [market["Date"].min().date(), ofz["payment_date"].min().date()],
        "date_max": [market["Date"].max().date(), ofz["payment_date"].max().date()],
        "missing_cells": [int(market.isna().sum().sum()), int(ofz.isna().sum().sum())],
    }).set_index("dataset")
    save_table(data_profile, TABLES_DIR / "data_profile.csv", TABLES_DIR / "data_profile.md")

    asset_columns = catalog.stocks + catalog.fx + catalog.aux_prices
    factors_all, pca_all = build_model_factors(market, asset_columns, catalog.rates, n_components=3)
    stats = descriptive_stats(factors_all)
    save_table(stats, TABLES_DIR / "risk_factor_descriptive_stats.csv", TABLES_DIR / "risk_factor_descriptive_stats.md")
    adf = adf_summary(factors_all)
    save_table(adf, TABLES_DIR / "adf_stationarity.csv", TABLES_DIR / "adf_stationarity.md")
    corr = correlation_matrix(factors_all)
    save_table(corr, TABLES_DIR / "risk_factor_correlation.csv")
    save_heatmap_svg(corr, FIGURES_DIR / "risk_factor_correlation.svg", "Risk-factor correlation matrix")

    pca_ev = pd.DataFrame({
        "component": [f"PC{i+1}" for i in range(len(pca_all.explained_ratio))],
        "explained_variance": pca_all.explained_variance,
        "explained_ratio": pca_all.explained_ratio,
        "cumulative_ratio": pca_all.explained_ratio.cumsum(),
    }).set_index("component")
    save_table(pca_ev, TABLES_DIR / "pca_explained_variance.csv", TABLES_DIR / "pca_explained_variance.md")
    save_table(pca_all.component_table, TABLES_DIR / "pca_components.csv", TABLES_DIR / "pca_components.md")
    save_bar_svg(pca_ev.index, pca_ev["explained_ratio"], FIGURES_DIR / "pca_explained_variance.svg", "PCA explained variance")
    save_line_svg(list(pca_all.component_table.index), {c: pca_all.component_table[c] for c in pca_all.component_table.columns}, FIGURES_DIR / "pca_components.svg", "Yield-curve PCA loadings")

    for factor in ["SBER", "Курс_USD", "PC1", "PC2"]:
        if factor in factors_all.columns:
            save_hist_svg(factors_all[factor], FIGURES_DIR / f"distribution_{factor}.svg", f"Distribution: {factor}")

    # Выбор стохастической модели динамики (п.3): сравнение Normal vs Student-t по MLE
    # для каждого фактора и оценка GARCH(1,1)-t по правдоподобию для ключевых факторов.
    distribution_selection = make_distribution_selection(factors_all)
    save_table(distribution_selection, TABLES_DIR / "distribution_selection.csv", TABLES_DIR / "distribution_selection.md")
    garch_estimates = make_garch_estimates(factors_all, ["Курс_USD", "Курс_EUR", "SBER", "PC1", "PC2", "PC3"])
    save_table(garch_estimates, TABLES_DIR / "garch_estimates.csv", TABLES_DIR / "garch_estimates.md")

    risk_anchor = previous_trading_date(market, RISK_DATE)
    portfolio = make_portfolio_composition(market, risk_anchor, catalog)
    save_table(portfolio, TABLES_DIR / "portfolio_composition.csv", TABLES_DIR / "portfolio_composition.md")
    risk_factor_inventory = make_risk_factor_inventory(catalog)
    save_table(risk_factor_inventory, TABLES_DIR / "risk_factor_inventory.csv", TABLES_DIR / "risk_factor_inventory.md")
    limitations = make_limitations_table()
    save_table(limitations, TABLES_DIR / "model_limitations.csv", TABLES_DIR / "model_limitations.md")

    dynamics = market[market["Date"] >= pd.Timestamp("2024-01-01")].copy()
    dynamic_series = {}
    for col in ["SBER", "IMOEX_CLOSE", "Курс_USD", "Курс_EUR", "Нефть_Brent"]:
        if col in dynamics.columns:
            dynamic_series[col] = dynamics[col] / dynamics[col].iloc[0] * 100
    save_line_svg(
        [d.strftime("%Y-%m-%d") for d in dynamics["Date"]],
        dynamic_series,
        FIGURES_DIR / "risk_factor_dynamics_prices_fx.svg",
        "Main risk-factor dynamics, indexed to 100",
    )
    save_line_svg(
        [d.strftime("%Y-%m-%d") for d in dynamics["Date"]],
        {col: dynamics[col] for col in ["0.25 years", "1 year", "5 years", "10 years", "30 years"] if col in dynamics.columns},
        FIGURES_DIR / "risk_factor_dynamics_rates.svg",
        "Yield-curve dynamics",
    )

    # Основная модель оценки риска: условная волатильность EWMA + многомерное t-Стьюдента.
    # Baseline (безусловная нормаль) считаем рядом — для сравнительной таблицы.
    model = fit_dynamics_model(market, risk_anchor, catalog.stocks, catalog.fx, catalog.rates, catalog.bond_prices,
                               vol_method="ewma", innovation="t")
    baseline = fit_normal_model(market, risk_anchor, catalog.stocks, catalog.fx, catalog.rates, catalog.bond_prices)

    dyn_diag = model.diagnostics.copy()
    dyn_diag["mvt_dof"] = model.nu
    save_table(dyn_diag, TABLES_DIR / "dynamics_diagnostics.csv", TABLES_DIR / "dynamics_diagnostics.md")

    pricing_errors = bond_pricing_errors(market, ofz, risk_anchor, catalog.rates, catalog.bond_prices)
    save_table(pricing_errors, TABLES_DIR / "bond_pricing_errors.csv", TABLES_DIR / "bond_pricing_errors.md")
    save_bar_svg(
        pricing_errors.index,
        pricing_errors["error_pct_of_market"] * 100,
        FIGURES_DIR / "bond_pricing_errors.svg",
        "OFZ DCF pricing error, % of market price",
    )

    metric_rows = []
    comparison_rows = []
    for horizon in [1, 10]:
        pnl, base_values = simulate_portfolio_pnl_cond(model, market, ofz, risk_anchor, horizon_days=horizon, n_scenarios=FINAL_SCENARIOS, seed=SEED)
        pnl_base, _ = simulate_portfolio_pnl(baseline, market, ofz, risk_anchor, horizon_days=horizon, n_scenarios=FINAL_SCENARIOS, seed=SEED)
        pnl.to_csv(OUTPUTS_DIR / f"mc_pnl_h{horizon}.csv", index=False)
        for component in ["stocks", "bonds", "fx", "total"]:
            metrics = var_es(pnl[component].to_numpy(), var_level=0.99, es_level=0.975)
            metrics_base = var_es(pnl_base[component].to_numpy(), var_level=0.99, es_level=0.975)
            metric_rows.append({
                "risk_date": RISK_DATE,
                "anchor_date": risk_anchor.date(),
                "horizon_days": horizon,
                "component": component,
                "VaR_99_RUB": metrics["VaR_99"],
                "ES_97_5_RUB": metrics["ES_97.5%"],
                "mean_pnl_RUB": pnl[component].mean(),
                "std_pnl_RUB": pnl[component].std(ddof=1),
                "scenarios": FINAL_SCENARIOS,
                "model": f"EWMA cond. vol + multivariate Student-t (dof={model.nu:.0f})",
            })
            comparison_rows.append({
                "horizon_days": horizon,
                "component": component,
                "VaR_baseline_normal": metrics_base["VaR_99"],
                "VaR_ewma_student_t": metrics["VaR_99"],
                "ratio_new_over_baseline": metrics["VaR_99"] / metrics_base["VaR_99"] if metrics_base["VaR_99"] else float("nan"),
            })
            save_hist_svg(pnl[component], FIGURES_DIR / f"pnl_distribution_{component}_h{horizon}.svg", f"P&L distribution: {component}, {horizon}d")
        pd.Series(base_values).to_csv(OUTPUTS_DIR / f"portfolio_base_values_h{horizon}.csv")
    risk_metrics = pd.DataFrame(metric_rows).set_index(["horizon_days", "component"])
    save_table(risk_metrics, OUTPUTS_DIR / "risk_metrics_var_es.csv", OUTPUTS_DIR / "risk_metrics_var_es.md")
    save_table(risk_metrics, TABLES_DIR / "var_es_summary.csv", TABLES_DIR / "var_es_summary.md")
    model_comparison = pd.DataFrame(comparison_rows).set_index(["horizon_days", "component"])
    save_table(model_comparison, TABLES_DIR / "model_comparison_var.csv", TABLES_DIR / "model_comparison_var.md")

    backtest_details, backtest_summary = run_backtest(
        market,
        ofz,
        catalog.stocks,
        catalog.fx,
        catalog.rates,
        catalog.bond_prices,
        year=2025,
        n_scenarios=BACKTEST_SCENARIOS,
        seed=SEED,
    )
    backtest_details.to_csv(OUTPUTS_DIR / "backtest_2025_details.csv", index=False)
    save_table(backtest_summary, OUTPUTS_DIR / "backtest_2025_summary.csv", OUTPUTS_DIR / "backtest_2025_summary.md")
    save_table(backtest_summary, TABLES_DIR / "backtesting_summary.csv", TABLES_DIR / "backtesting_summary.md")
    kupiec = backtest_summary[["observations", "exceptions", "exception_rate", "LR_uc", "p_value"]].rename(columns={"p_value": "kupiec_p_value"})
    christoffersen = backtest_summary[["n00", "n01", "n10", "n11", "LR_ind", "p_ind", "LR_cc", "p_cc"]]
    save_table(kupiec, TABLES_DIR / "kupiec_test.csv", TABLES_DIR / "kupiec_test.md")
    save_table(christoffersen, TABLES_DIR / "christoffersen_test.csv", TABLES_DIR / "christoffersen_test.md")
    for component in ["stocks", "bonds", "fx", "total"]:
        labels = [d.strftime("%Y-%m-%d") for d in backtest_details["Date"]]
        save_line_svg(
            labels,
            {
                "actual P&L": backtest_details[f"{component}_pnl"],
                "-VaR 99%": -backtest_details[f"{component}_VaR99"],
            },
            FIGURES_DIR / f"backtest_{component}_pnl_vs_var.svg",
            f"Backtest 2025: {component} P&L vs VaR",
        )
        save_backtest_svg(
            labels,
            backtest_details[f"{component}_pnl"],
            backtest_details[f"{component}_VaR99"],
            backtest_details[f"{component}_exception"],
            FIGURES_DIR / f"backtest_{component}_breaches.svg",
            f"Backtest 2025: {component} VaR breaches",
        )

    print("Pipeline complete")
    print(f"Risk anchor: {risk_anchor.date()}")
    print(risk_metrics[["VaR_99_RUB", "ES_97_5_RUB"]].round(0).to_string())
    print(backtest_summary[["observations", "exceptions", "exception_rate", "p_value", "p_cc"]].to_string())


if __name__ == "__main__":
    main()
