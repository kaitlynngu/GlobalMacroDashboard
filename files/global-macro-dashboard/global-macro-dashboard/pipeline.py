"""
pipeline.py
-----------
End-to-end data pipeline for the Global Macro Dashboard.

Steps:
  1. Fetch macroeconomic indicators from the World Bank API
  2. Clean and validate the raw data (QC checks)
  3. Load into a structured SQLite database
  4. Log pipeline run metadata

Indicators collected:
  - NY.GDP.MKTP.KD.ZG  : GDP growth (annual %)
  - FP.CPI.TOTL.ZG     : Inflation, CPI (annual %)
  - SL.UEM.TOTL.ZS     : Unemployment, total (% of labor force)
  - NE.EXP.GNFS.ZS     : Exports of goods and services (% of GDP)
  - GC.DOD.TOTL.GD.ZS  : Central government debt (% of GDP)
"""

import requests
import pandas as pd
import sqlite3
import json
import logging
import time
from datetime import datetime
from pathlib import Path

# ── Configuration ────────────────────────────────────────────────────────────

BASE_DIR   = Path(__file__).parent
DATA_DIR   = BASE_DIR / "data"
RAW_DIR    = DATA_DIR / "raw"
PROC_DIR   = DATA_DIR / "processed"
DB_PATH    = DATA_DIR / "macro.db"

WORLD_BANK_BASE = "https://api.worldbank.org/v2"

INDICATORS = {
    "NY.GDP.MKTP.KD.ZG" : "gdp_growth",
    "FP.CPI.TOTL.ZG"    : "inflation",
    "SL.UEM.TOTL.ZS"    : "unemployment",
    "NE.EXP.GNFS.ZS"    : "exports_pct_gdp",
    "GC.DOD.TOTL.GD.ZS" : "govt_debt_pct_gdp",
}

# 40 countries spanning all major regions
COUNTRIES = [
    # North America
    "US", "CA", "MX",
    # Europe
    "GB", "DE", "FR", "IT", "ES", "NL", "SE", "PL", "TR",
    # Asia-Pacific
    "CN", "JP", "KR", "IN", "AU", "ID", "TH", "VN", "PH", "MY",
    # Latin America
    "BR", "AR", "CL", "CO", "PE",
    # Middle East & Africa
    "SA", "AE", "ZA", "NG", "EG", "KE", "MA",
    # Eastern Europe / Central Asia
    "RU", "UA", "KZ",
    # Other
    "SG", "NZ",
]

START_YEAR = 2000
END_YEAR   = 2023

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── World Bank API ────────────────────────────────────────────────────────────

def fetch_indicator(indicator_code: str, countries: list[str],
                    start: int, end: int) -> pd.DataFrame:
    """Fetch a single indicator for all countries from the World Bank API."""
    country_str = ";".join(countries)
    url = (
        f"{WORLD_BANK_BASE}/country/{country_str}/indicator/{indicator_code}"
        f"?format=json&per_page=10000&mrv=&date={start}:{end}"
    )
    log.info(f"Fetching {indicator_code} ...")
    response = requests.get(url, timeout=30)
    response.raise_for_status()

    payload = response.json()
    if len(payload) < 2 or not payload[1]:
        log.warning(f"No data returned for {indicator_code}")
        return pd.DataFrame()

    records = []
    for entry in payload[1]:
        if entry.get("value") is not None:
            records.append({
                "country_code" : entry["countryiso3code"],
                "country_name" : entry["country"]["value"],
                "year"         : int(entry["date"]),
                "value"        : float(entry["value"]),
                "indicator"    : indicator_code,
            })

    df = pd.DataFrame(records)
    log.info(f"  → {len(df):,} rows for {indicator_code}")
    return df


def fetch_all_indicators() -> pd.DataFrame:
    """Fetch all configured indicators and combine into one long-format DataFrame."""
    frames = []
    for code, name in INDICATORS.items():
        df = fetch_indicator(code, COUNTRIES, START_YEAR, END_YEAR)
        if not df.empty:
            df["indicator_name"] = name
            frames.append(df)
        time.sleep(0.3)   # polite rate limiting

    combined = pd.concat(frames, ignore_index=True)
    log.info(f"Total raw rows fetched: {len(combined):,}")
    return combined


# ── Data Cleaning & QC ────────────────────────────────────────────────────────

def run_qc_checks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run quality-control checks and flag anomalies.

    QC flags added:
      - outlier_flag  : value > 4 IQR from median within indicator
      - missing_flag  : placeholder for missing years (not in this df, but logged)
    """
    log.info("Running QC checks ...")
    original_len = len(df)

    # Remove rows with null country codes (World Bank aggregates)
    df = df[df["country_code"].str.len() == 3].copy()

    # Flag statistical outliers per indicator using IQR method
    df["outlier_flag"] = False
    for ind in df["indicator_name"].unique():
        mask = df["indicator_name"] == ind
        q1   = df.loc[mask, "value"].quantile(0.25)
        q3   = df.loc[mask, "value"].quantile(0.75)
        iqr  = q3 - q1
        lo   = q1 - 4 * iqr
        hi   = q3 + 4 * iqr
        outlier_mask = mask & ((df["value"] < lo) | (df["value"] > hi))
        df.loc[outlier_mask, "outlier_flag"] = True

    n_outliers = df["outlier_flag"].sum()
    n_removed  = original_len - len(df)

    log.info(f"  Rows removed (bad country codes): {n_removed}")
    log.info(f"  Outliers flagged (kept, not dropped): {n_outliers}")

    # Check coverage: warn if a country is missing more than 30% of years
    expected_years = END_YEAR - START_YEAR + 1
    coverage = (
        df.groupby(["country_code", "indicator_name"])["year"]
        .count()
        .reset_index(name="n_years")
    )
    low_cov = coverage[coverage["n_years"] < expected_years * 0.7]
    if not low_cov.empty:
        log.warning(f"  {len(low_cov)} country-indicator pairs have <70% year coverage")

    return df


def pivot_to_wide(df: pd.DataFrame) -> pd.DataFrame:
    """Convert long-format data to wide format (one row per country-year)."""
    wide = df.pivot_table(
        index   = ["country_code", "country_name", "year"],
        columns = "indicator_name",
        values  = "value",
        aggfunc = "mean",
    ).reset_index()
    wide.columns.name = None
    log.info(f"Wide-format shape: {wide.shape}")
    return wide


# ── SQLite Loading ────────────────────────────────────────────────────────────

def load_to_sqlite(long_df: pd.DataFrame, wide_df: pd.DataFrame) -> None:
    """
    Load cleaned data into SQLite.

    Tables created:
      - macro_long    : tidy long-format (country, year, indicator, value)
      - macro_wide    : one row per country-year, one column per indicator
      - country_meta  : distinct countries and their codes
      - pipeline_runs : metadata log of each pipeline execution
    """
    log.info(f"Loading to SQLite: {DB_PATH}")
    con = sqlite3.connect(DB_PATH)

    # Long table
    long_df.to_sql("macro_long", con, if_exists="replace", index=False)
    log.info("  → macro_long loaded")

    # Wide table
    wide_df.to_sql("macro_wide", con, if_exists="replace", index=False)
    log.info("  → macro_wide loaded")

    # Country metadata
    meta = long_df[["country_code", "country_name"]].drop_duplicates()
    meta["region"] = meta["country_code"].map(REGION_MAP)
    meta.to_sql("country_meta", con, if_exists="replace", index=False)
    log.info("  → country_meta loaded")

    # Pipeline run log
    run_record = pd.DataFrame([{
        "run_at"      : datetime.utcnow().isoformat(),
        "rows_long"   : len(long_df),
        "rows_wide"   : len(wide_df),
        "countries"   : len(long_df["country_code"].unique()),
        "indicators"  : len(long_df["indicator_name"].unique()),
        "year_range"  : f"{START_YEAR}-{END_YEAR}",
    }])
    run_record.to_sql("pipeline_runs", con, if_exists="append", index=False)
    log.info("  → pipeline_runs updated")

    # Add indexes for query performance
    con.execute("CREATE INDEX IF NOT EXISTS idx_long_country ON macro_long(country_code)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_long_year    ON macro_long(year)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_long_ind     ON macro_long(indicator_name)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_wide_country ON macro_wide(country_code)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_wide_year    ON macro_wide(year)")
    con.commit()
    con.close()
    log.info("  → Indexes created. DB ready.")


# ── Region mapping ────────────────────────────────────────────────────────────

REGION_MAP = {
    "USA": "North America",  "CAN": "North America",  "MEX": "Latin America",
    "GBR": "Europe",         "DEU": "Europe",          "FRA": "Europe",
    "ITA": "Europe",         "ESP": "Europe",          "NLD": "Europe",
    "SWE": "Europe",         "POL": "Europe",          "TUR": "Europe",
    "CHN": "Asia-Pacific",   "JPN": "Asia-Pacific",    "KOR": "Asia-Pacific",
    "IND": "Asia-Pacific",   "AUS": "Asia-Pacific",    "IDN": "Asia-Pacific",
    "THA": "Asia-Pacific",   "VNM": "Asia-Pacific",    "PHL": "Asia-Pacific",
    "MYS": "Asia-Pacific",   "BRA": "Latin America",   "ARG": "Latin America",
    "CHL": "Latin America",  "COL": "Latin America",   "PER": "Latin America",
    "SAU": "Middle East & Africa", "ARE": "Middle East & Africa",
    "ZAF": "Middle East & Africa", "NGA": "Middle East & Africa",
    "EGY": "Middle East & Africa", "KEN": "Middle East & Africa",
    "MAR": "Middle East & Africa", "RUS": "Eastern Europe",
    "UKR": "Eastern Europe", "KAZ": "Eastern Europe",
    "SGP": "Asia-Pacific",   "NZL": "Asia-Pacific",
}


# ── Main ──────────────────────────────────────────────────────────────────────

def run_pipeline():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROC_DIR.mkdir(parents=True, exist_ok=True)

    log.info("=" * 55)
    log.info("  GLOBAL MACRO DASHBOARD — DATA PIPELINE")
    log.info("=" * 55)

    # 1. Fetch
    raw = fetch_all_indicators()
    raw.to_csv(RAW_DIR / "macro_raw.csv", index=False)
    log.info(f"Raw data saved → {RAW_DIR / 'macro_raw.csv'}")

    # 2. Clean & QC
    clean = run_qc_checks(raw)
    clean.to_csv(PROC_DIR / "macro_clean_long.csv", index=False)

    # 3. Pivot wide
    wide = pivot_to_wide(clean)
    wide.to_csv(PROC_DIR / "macro_wide.csv", index=False)

    # 4. Load to DB
    load_to_sqlite(clean, wide)

    log.info("=" * 55)
    log.info("  Pipeline complete.")
    log.info("=" * 55)


if __name__ == "__main__":
    run_pipeline()
