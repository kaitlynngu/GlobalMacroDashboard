"""
analysis.py
-----------
Core analytical layer for the Global Macro Dashboard.

Modules:
  - SQLQueryRunner : executes SQL views and returns DataFrames
  - TrendForecaster: fits linear + CAGR trend models per country/indicator
  - RiskScorer     : builds composite macro stability scores
  - CorrelationAnalyzer: cross-country and cross-indicator correlations
"""

import sqlite3
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

warnings.filterwarnings("ignore")

DB_PATH  = Path(__file__).parent / "data" / "macro.db"
SQL_PATH = Path(__file__).parent / "sql" / "queries.sql"


# ── SQL Runner ────────────────────────────────────────────────────────────────

class SQLQueryRunner:
    """
    Executes SQL views and ad-hoc queries against the SQLite database.
    Automatically creates views defined in queries.sql on first use.
    """

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self._init_views()

    def _init_views(self):
        """Run queries.sql to register all CREATE VIEW statements."""
        sql_text = SQL_PATH.read_text()
        con = sqlite3.connect(self.db_path)
        # Strip comment lines, split on semicolons
        stmts = []
        current = []
        for line in sql_text.splitlines():
            stripped = line.strip()
            if stripped.startswith("--"):
                continue
            current.append(line)
            if stripped.endswith(";"):
                joined = "\n".join(current).strip().rstrip(";")
                if joined:
                    stmts.append(joined)
                current = []
        for stmt in stmts:
            try:
                con.execute(stmt)
            except sqlite3.Error:
                pass  # View may already exist
        con.commit()
        con.close()

    def query(self, sql: str) -> pd.DataFrame:
        con = sqlite3.connect(self.db_path)
        df  = pd.read_sql_query(sql, con)
        con.close()
        return df

    def get_risk_scores(self) -> pd.DataFrame:
        return self.query("SELECT * FROM v_risk_scores ORDER BY macro_stability_score DESC")

    def get_regional_gdp(self) -> pd.DataFrame:
        return self.query("SELECT * FROM v_regional_gdp ORDER BY region, year")

    def get_shock_years(self) -> pd.DataFrame:
        return self.query("SELECT * FROM v_shock_years ORDER BY year")

    def get_crisis_recovery(self) -> pd.DataFrame:
        return self.query("SELECT * FROM v_crisis_recovery ORDER BY crisis, years_to_recover")

    def get_top_bottom(self) -> pd.DataFrame:
        return self.query("SELECT * FROM v_top_bottom_performers ORDER BY avg_gdp_5yr DESC")

    def get_wide(self) -> pd.DataFrame:
        return self.query("SELECT * FROM macro_wide")

    def get_long(self) -> pd.DataFrame:
        return self.query("SELECT * FROM macro_long")


# ── Trend Forecaster ──────────────────────────────────────────────────────────

class TrendForecaster:
    """
    Fits linear trend models per country-indicator and generates 5-year forecasts.

    Methods
    -------
    fit(df_wide)          : trains models on historical data
    forecast(n_years)     : returns forecast DataFrame for next n years
    get_model_summary()   : R² scores and trend slopes for all models
    """

    def __init__(self):
        self.models   : dict = {}
        self.scalers  : dict = {}
        self.history  : pd.DataFrame | None = None
        self.last_year: int = 0

    def fit(self, df_wide: pd.DataFrame) -> "TrendForecaster":
        self.history   = df_wide.copy()
        self.last_year = int(df_wide["year"].max())
        indicators     = ["gdp_growth", "inflation", "unemployment",
                          "exports_pct_gdp", "govt_debt_pct_gdp"]

        for country in df_wide["country_code"].unique():
            cdf = df_wide[df_wide["country_code"] == country].sort_values("year")
            for ind in indicators:
                col_data = cdf[["year", ind]].dropna()
                if len(col_data) < 5:
                    continue
                X = col_data["year"].values.reshape(-1, 1)
                y = col_data[ind].values

                scaler = StandardScaler()
                X_sc   = scaler.fit_transform(X)
                model  = LinearRegression().fit(X_sc, y)

                key = (country, ind)
                self.models[key]  = model
                self.scalers[key] = scaler

        print(f"Trained {len(self.models)} trend models across "
              f"{df_wide['country_code'].nunique()} countries.")
        return self

    def forecast(self, n_years: int = 5) -> pd.DataFrame:
        """Return a DataFrame of point forecasts for the next n_years."""
        future_years = list(range(self.last_year + 1, self.last_year + n_years + 1))
        rows = []

        # Get country metadata from history
        meta = (self.history[["country_code", "country_name", "region"]]
                .drop_duplicates()
                .set_index("country_code"))

        for (country, ind), model in self.models.items():
            scaler = self.scalers[(country, ind)]
            X_fut  = np.array(future_years).reshape(-1, 1)
            X_sc   = scaler.transform(X_fut)
            preds  = model.predict(X_sc)

            country_name = meta.loc[country, "country_name"] if country in meta.index else country
            region       = meta.loc[country, "region"]       if country in meta.index else "Unknown"

            for yr, pred in zip(future_years, preds):
                rows.append({
                    "country_code"  : country,
                    "country_name"  : country_name,
                    "region"        : region,
                    "year"          : yr,
                    "indicator"     : ind,
                    "forecast_value": round(float(pred), 2),
                    "is_forecast"   : True,
                })

        return pd.DataFrame(rows)

    def get_model_summary(self) -> pd.DataFrame:
        """Return R² and slope for each country-indicator model."""
        rows = []
        indicators = ["gdp_growth", "inflation", "unemployment",
                      "exports_pct_gdp", "govt_debt_pct_gdp"]

        meta = (self.history[["country_code", "country_name"]]
                .drop_duplicates()
                .set_index("country_code"))

        for (country, ind), model in self.models.items():
            scaler  = self.scalers[(country, ind)]
            cdf     = self.history[self.history["country_code"] == country]
            col_data= cdf[["year", ind]].dropna()
            X       = scaler.transform(col_data["year"].values.reshape(-1, 1))
            y       = col_data[ind].values
            y_pred  = model.predict(X)
            r2      = r2_score(y, y_pred)
            slope   = float(model.coef_[0])

            rows.append({
                "country_code"  : country,
                "country_name"  : meta.loc[country, "country_name"] if country in meta.index else country,
                "indicator"     : ind,
                "r2_score"      : round(r2, 3),
                "trend_slope"   : round(slope, 4),
                "trend_direction": "↑ Improving" if slope > 0.05 else
                                   "↓ Declining" if slope < -0.05 else "→ Stable",
            })

        df = pd.DataFrame(rows).sort_values(["indicator", "r2_score"], ascending=[True, False])
        return df


# ── Risk Scorer ───────────────────────────────────────────────────────────────

class RiskScorer:
    """
    Builds a composite Macro Stability Score (0–100) for each country.

    Score components (configurable):
        GDP Growth    40%  (higher → lower risk)
        Inflation     30%  (lower  → lower risk)
        Unemployment  30%  (lower  → lower risk)
    """

    WEIGHTS = {"gdp_growth": 0.40, "inflation": 0.30, "unemployment": 0.30}

    def __init__(self, window_years: int = 5):
        self.window = window_years

    def score(self, df_wide: pd.DataFrame) -> pd.DataFrame:
        last_year  = df_wide["year"].max()
        start_year = last_year - self.window + 1

        recent = (
            df_wide[df_wide["year"] >= start_year]
            .groupby(["country_code", "country_name", "region"])
            [["gdp_growth", "inflation", "unemployment"]]
            .mean()
            .reset_index()
        )

        # Min-max normalise each component
        def norm_asc(s):   # higher raw = higher score
            return (s - s.min()) / (s.max() - s.min()).clip(lower=1e-9)

        def norm_desc(s):  # lower raw = higher score
            return 1 - norm_asc(s)

        recent["gdp_norm"]   = norm_asc( recent["gdp_growth"])
        recent["inf_norm"]   = norm_desc(recent["inflation"])
        recent["unemp_norm"] = norm_desc(recent["unemployment"])

        recent["macro_stability_score"] = (
            self.WEIGHTS["gdp_growth"]  * recent["gdp_norm"]   +
            self.WEIGHTS["inflation"]   * recent["inf_norm"]    +
            self.WEIGHTS["unemployment"]* recent["unemp_norm"]
        ) * 100

        recent["risk_tier"] = pd.cut(
            recent["macro_stability_score"],
            bins  = [0, 40, 70, 100],
            labels= ["High Risk", "Medium Risk", "Low Risk"],
        )

        return recent.sort_values("macro_stability_score", ascending=False).round(2)


# ── Correlation Analyzer ──────────────────────────────────────────────────────

class CorrelationAnalyzer:
    """
    Cross-indicator and cross-country correlation analysis.
    """

    def indicator_correlation(self, df_wide: pd.DataFrame) -> pd.DataFrame:
        """Pearson correlation matrix across all macroeconomic indicators."""
        cols = ["gdp_growth", "inflation", "unemployment",
                "exports_pct_gdp", "govt_debt_pct_gdp"]
        return df_wide[cols].corr().round(3)

    def gdp_vs_inflation_by_region(self, df_wide: pd.DataFrame) -> pd.DataFrame:
        """Correlation between GDP growth and inflation, computed per region."""
        rows = []
        for region, grp in df_wide.groupby("region"):
            valid = grp[["gdp_growth", "inflation"]].dropna()
            if len(valid) >= 10:
                corr = valid["gdp_growth"].corr(valid["inflation"])
                rows.append({"region": region, "gdp_inflation_corr": round(corr, 3),
                             "n_obs": len(valid)})
        return pd.DataFrame(rows).sort_values("gdp_inflation_corr")

    def country_gdp_correlation_matrix(self, df_wide: pd.DataFrame,
                                        region: str | None = None) -> pd.DataFrame:
        """
        Pivot GDP growth by country-year and return the country×country
        correlation matrix. Optionally filter to one region.
        """
        sub = df_wide if region is None else df_wide[df_wide["region"] == region]
        pivot = sub.pivot_table(index="year", columns="country_name",
                                values="gdp_growth")
        return pivot.corr().round(3)


# ── Convenience loader ────────────────────────────────────────────────────────

def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load wide and long DataFrames directly from CSV (no DB needed)."""
    base = Path(__file__).parent / "data" / "processed"
    wide = pd.read_csv(base / "macro_wide.csv")
    long = pd.read_csv(base / "macro_clean_long.csv")
    return wide, long


if __name__ == "__main__":
    # Quick sanity check
    runner = SQLQueryRunner()
    print("Risk scores (top 5):")
    print(runner.get_risk_scores().head())
    print("\nShock years:")
    print(runner.get_shock_years()[["year", "global_avg_gdp_growth", "year_classification"]])
