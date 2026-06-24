from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


RATE_COLUMNS = [
    "0 years", "0.25 years", "0.5 years", "0.75 years", "1 year", "2 years",
    "3 years", "5 years", "7 years", "10 years", "15 years", "20 years", "30 years",
]
TENOR_YEARS = {
    "0 years": 0.01,
    "0.25 years": 0.25,
    "0.5 years": 0.5,
    "0.75 years": 0.75,
    "1 year": 1.0,
    "2 years": 2.0,
    "3 years": 3.0,
    "5 years": 5.0,
    "7 years": 7.0,
    "10 years": 10.0,
    "15 years": 15.0,
    "20 years": 20.0,
    "30 years": 30.0,
}
STOCK_COLUMNS = ["SBER", "GAZP", "LKOH", "ROSN", "TATN", "GMKN", "CHMF", "NLMK", "NVTK", "MTSS"]
FX_COLUMNS = ["Курс_USD", "Курс_EUR"]
AUX_PRICE_COLUMNS = ["IMOEX_CLOSE", "RTSI_CLOSE", "Нефть_Brent"]


@dataclass(frozen=True)
class DataCatalog:
    rates: list[str]
    stocks: list[str]
    fx: list[str]
    bond_prices: list[str]
    aux_prices: list[str]


def load_market_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")], errors="ignore")
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").drop_duplicates("Date").reset_index(drop=True)
    for col in df.columns:
        if col != "Date":
            df[col] = pd.to_numeric(df[col].astype(str).str.replace("%", "", regex=False).str.replace(",", ".", regex=False), errors="coerce")
    return df


def load_ofz_cashflows(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")], errors="ignore")
    df["Дата"] = pd.to_datetime(df["Дата"], dayfirst=True)
    df["code"] = df["Полное наименование ЦБ"].str.extract(r"(\d{5})", expand=False)
    df = df.rename(columns={"Дата": "payment_date", "Размер выплаты": "payment_amount"})
    return df.sort_values(["code", "payment_date"]).reset_index(drop=True)


def classify_columns(df: pd.DataFrame) -> DataCatalog:
    rates = [c for c in RATE_COLUMNS if c in df.columns]
    stocks = [c for c in STOCK_COLUMNS if c in df.columns]
    fx = [c for c in FX_COLUMNS if c in df.columns]
    bond_prices = [c for c in df.columns if re.fullmatch(r"\d{5}_Price", c)]
    aux_prices = [c for c in AUX_PRICE_COLUMNS if c in df.columns]
    return DataCatalog(rates=rates, stocks=stocks, fx=fx, bond_prices=bond_prices, aux_prices=aux_prices)


def previous_trading_date(df: pd.DataFrame, date: str | pd.Timestamp) -> pd.Timestamp:
    target = pd.Timestamp(date)
    dates = df.loc[df["Date"] < target, "Date"]
    if dates.empty:
        raise ValueError(f"No trading date before {target.date()}")
    return dates.max()


def row_on_or_before(df: pd.DataFrame, date: str | pd.Timestamp) -> pd.Series:
    target = pd.Timestamp(date)
    rows = df[df["Date"] <= target]
    if rows.empty:
        raise ValueError(f"No data on or before {target.date()}")
    return rows.iloc[-1]


def save_processed(market: pd.DataFrame, ofz: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    market.to_csv(out_dir / "market_data_clean.csv", index=False)
    ofz.to_csv(out_dir / "ofz_cashflows_clean.csv", index=False)

