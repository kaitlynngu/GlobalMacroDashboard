"""
generate_sample_data.py
-----------------------
Generates realistic synthetic macroeconomic data that mirrors
the World Bank API output. Used for local development and testing
when API access is unavailable.

In production, replace by running: python pipeline.py
"""

import numpy as np
import pandas as pd
import sqlite3
from pathlib import Path

np.random.seed(42)

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR  = DATA_DIR / "raw"
PROC_DIR = DATA_DIR / "processed"
DB_PATH  = DATA_DIR / "macro.db"

COUNTRIES = {
    "USA": ("United States",    "North America"),
    "CAN": ("Canada",           "North America"),
    "MEX": ("Mexico",           "Latin America"),
    "GBR": ("United Kingdom",   "Europe"),
    "DEU": ("Germany",          "Europe"),
    "FRA": ("France",           "Europe"),
    "ITA": ("Italy",            "Europe"),
    "ESP": ("Spain",            "Europe"),
    "NLD": ("Netherlands",      "Europe"),
    "SWE": ("Sweden",           "Europe"),
    "POL": ("Poland",           "Europe"),
    "TUR": ("Turkey",           "Europe"),
    "CHN": ("China",            "Asia-Pacific"),
    "JPN": ("Japan",            "Asia-Pacific"),
    "KOR": ("South Korea",      "Asia-Pacific"),
    "IND": ("India",            "Asia-Pacific"),
    "AUS": ("Australia",        "Asia-Pacific"),
    "IDN": ("Indonesia",        "Asia-Pacific"),
    "THA": ("Thailand",         "Asia-Pacific"),
    "VNM": ("Vietnam",          "Asia-Pacific"),
    "PHL": ("Philippines",      "Asia-Pacific"),
    "MYS": ("Malaysia",         "Asia-Pacific"),
    "BRA": ("Brazil",           "Latin America"),
    "ARG": ("Argentina",        "Latin America"),
    "CHL": ("Chile",            "Latin America"),
    "COL": ("Colombia",         "Latin America"),
    "PER": ("Peru",             "Latin America"),
    "SAU": ("Saudi Arabia",     "Middle East & Africa"),
    "ARE": ("UAE",              "Middle East & Africa"),
    "ZAF": ("South Africa",     "Middle East & Africa"),
    "NGA": ("Nigeria",          "Middle East & Africa"),
    "EGY": ("Egypt",            "Middle East & Africa"),
    "KEN": ("Kenya",            "Middle East & Africa"),
    "MAR": ("Morocco",          "Middle East & Africa"),
    "RUS": ("Russia",           "Eastern Europe"),
    "UKR": ("Ukraine",          "Eastern Europe"),
    "KAZ": ("Kazakhstan",       "Eastern Europe"),
    "SGP": ("Singapore",        "Asia-Pacific"),
    "NZL": ("New Zealand",      "Asia-Pacific"),
    "BEL": ("Belgium",          "Europe"),
}

YEARS = list(range(2000, 2024))

# Realistic baseline profiles per country: (gdp_mean, gdp_vol, inflation_mean, inflation_vol, unemp_mean)
PROFILES = {
    "USA": (2.3, 1.8, 2.3, 1.2, 5.8),
    "CAN": (2.1, 1.6, 2.0, 1.0, 6.5),
    "MEX": (2.0, 2.5, 5.5, 2.5, 3.8),
    "GBR": (1.8, 1.7, 2.5, 1.5, 5.5),
    "DEU": (1.5, 2.0, 1.8, 0.9, 6.2),
    "FRA": (1.3, 1.8, 1.6, 0.8, 9.1),
    "ITA": (0.5, 2.0, 2.0, 1.2, 9.5),
    "ESP": (1.5, 2.5, 2.2, 1.5, 14.0),
    "NLD": (1.8, 1.8, 2.0, 1.0, 5.0),
    "SWE": (2.2, 1.8, 1.5, 0.9, 6.8),
    "POL": (3.8, 2.0, 3.0, 2.0, 8.5),
    "TUR": (4.5, 3.5, 15.0, 8.0, 9.5),
    "CHN": (8.0, 2.5, 2.5, 1.5, 4.2),
    "JPN": (0.8, 1.8, 0.3, 0.8, 3.8),
    "KOR": (3.5, 2.0, 2.5, 1.2, 3.8),
    "IND": (6.5, 2.5, 6.5, 2.5, 5.5),
    "AUS": (2.8, 1.5, 2.5, 1.0, 5.5),
    "IDN": (5.0, 2.0, 6.0, 2.5, 5.8),
    "THA": (3.5, 3.0, 2.5, 1.5, 1.2),
    "VNM": (6.5, 2.0, 5.5, 3.0, 2.0),
    "PHL": (5.5, 2.5, 4.5, 2.0, 6.5),
    "MYS": (4.8, 2.5, 2.5, 1.2, 3.2),
    "BRA": (2.0, 3.0, 7.0, 3.5, 8.5),
    "ARG": (1.5, 5.5, 35.0, 20.0, 10.5),
    "CHL": (3.5, 2.5, 3.5, 1.5, 7.5),
    "COL": (3.8, 2.5, 4.5, 2.0, 10.5),
    "PER": (4.5, 3.0, 3.0, 1.5, 5.5),
    "SAU": (3.0, 4.0, 2.5, 1.5, 5.5),
    "ARE": (3.5, 3.5, 2.5, 1.5, 2.2),
    "ZAF": (1.5, 2.5, 5.5, 2.0, 27.0),
    "NGA": (4.5, 4.0, 12.0, 5.0, 5.0),
    "EGY": (4.5, 2.5, 12.0, 5.0, 10.0),
    "KEN": (5.5, 2.5, 7.5, 3.0, 5.0),
    "MAR": (4.0, 2.0, 2.5, 1.5, 10.0),
    "RUS": (2.0, 4.5, 9.0, 4.0, 6.5),
    "UKR": (1.5, 6.0, 15.0, 10.0, 9.5),
    "KAZ": (5.0, 3.5, 8.0, 3.5, 5.5),
    "SGP": (4.5, 2.5, 1.5, 1.0, 2.5),
    "NZL": (2.5, 1.8, 2.5, 1.2, 4.5),
    "BEL": (1.5, 1.8, 2.0, 1.0, 7.5),
}

def apply_global_shocks(base_series, years):
    """Apply global shock events: 2001 dot-com, 2008 GFC, 2020 COVID."""
    s = base_series.copy()
    year_arr = np.array(years)
    # 2001 recession
    s[year_arr == 2001] -= 1.5
    s[year_arr == 2002] -= 0.8
    # 2008-09 GFC
    s[year_arr == 2008] -= 3.5
    s[year_arr == 2009] -= 4.5
    s[year_arr == 2010] += 2.0
    # 2020 COVID
    s[year_arr == 2020] -= 5.5
    s[year_arr == 2021] += 4.0
    # 2022 inflation shock
    s[year_arr == 2022] -= 0.8
    return s

def generate_long_df():
    rows = []
    for code, (name, region) in COUNTRIES.items():
        gdp_m, gdp_v, inf_m, inf_v, unemp_m = PROFILES.get(
            code, (2.5, 2.0, 3.5, 1.5, 6.5)
        )
        n = len(YEARS)
        # GDP growth with trend + shocks
        gdp_base  = np.random.normal(gdp_m, gdp_v, n)
        gdp_vals  = apply_global_shocks(gdp_base, YEARS)
        inf_vals  = np.clip(np.random.normal(inf_m, inf_v, n), 0, None)
        unemp_vals = np.clip(np.random.normal(unemp_m, 1.5, n), 0.5, 35)
        export_vals = np.clip(np.random.normal(38, 12, n), 5, 90)
        debt_vals   = np.clip(
            np.cumsum(np.random.normal(0.8, 1.5, n)) + 45 + np.random.normal(0, 5), 5, 250
        )

        for i, yr in enumerate(YEARS):
            for ind_name, val in [
                ("gdp_growth",       round(float(gdp_vals[i]), 2)),
                ("inflation",        round(float(inf_vals[i]), 2)),
                ("unemployment",     round(float(unemp_vals[i]), 2)),
                ("exports_pct_gdp",  round(float(export_vals[i]), 2)),
                ("govt_debt_pct_gdp",round(float(debt_vals[i]), 2)),
            ]:
                rows.append({
                    "country_code":   code,
                    "country_name":   name,
                    "region":         region,
                    "year":           yr,
                    "indicator_name": ind_name,
                    "value":          val,
                    "outlier_flag":   False,
                })
    return pd.DataFrame(rows)

def long_to_wide(df):
    wide = df.pivot_table(
        index=["country_code","country_name","region","year"],
        columns="indicator_name",
        values="value",
        aggfunc="mean"
    ).reset_index()
    wide.columns.name = None
    return wide

if __name__ == "__main__":
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROC_DIR.mkdir(parents=True, exist_ok=True)

    print("Generating synthetic data ...")
    long_df = generate_long_df()
    wide_df = long_to_wide(long_df)

    long_df.to_csv(RAW_DIR / "macro_raw.csv", index=False)
    long_df.to_csv(PROC_DIR / "macro_clean_long.csv", index=False)
    wide_df.to_csv(PROC_DIR / "macro_wide.csv", index=False)

    con = sqlite3.connect(DB_PATH)
    long_df.to_sql("macro_long", con, if_exists="replace", index=False)
    wide_df.to_sql("macro_wide", con, if_exists="replace", index=False)

    meta = long_df[["country_code","country_name","region"]].drop_duplicates()
    meta.to_sql("country_meta", con, if_exists="replace", index=False)

    pd.DataFrame([{
        "run_at": "2024-01-15T12:00:00",
        "rows_long": len(long_df),
        "rows_wide": len(wide_df),
        "countries": long_df["country_code"].nunique(),
        "indicators": long_df["indicator_name"].nunique(),
        "year_range": "2000-2023",
    }]).to_sql("pipeline_runs", con, if_exists="replace", index=False)

    con.execute("CREATE INDEX IF NOT EXISTS idx_long_country ON macro_long(country_code)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_long_year    ON macro_long(year)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_long_ind     ON macro_long(indicator_name)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_wide_country ON macro_wide(country_code)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_wide_year    ON macro_wide(year)")
    con.commit()
    con.close()

    print(f"Done. {len(long_df):,} rows long | {len(wide_df):,} rows wide")
    print(f"DB: {DB_PATH}")
