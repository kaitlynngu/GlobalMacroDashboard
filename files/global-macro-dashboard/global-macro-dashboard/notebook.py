"""
notebook.py  ·  Global Macro Dashboard — Full Analysis
=======================================================

This script is the "notebook" layer of the project: it runs the full
analytical pipeline end-to-end and prints a structured report to stdout.
It is designed to be readable both as a script and as a reference for the
accompanying Jupyter notebook (notebooks/analysis.ipynb).

Sections
--------
  0. Setup & data loading
  1. Database overview
  2. Global shock detection (SQL)
  3. Regional GDP performance (SQL)
  4. Country macro stability scores (Python + SQL)
  5. Trend forecasting (sklearn linear regression)
  6. Correlation analysis
  7. Crisis recovery analysis (SQL)
  8. Top / bottom performers (SQL)
  9. Forecast model diagnostics
"""

import warnings
import pandas as pd
import numpy as np
warnings.filterwarnings("ignore")

from analysis import SQLQueryRunner, TrendForecaster, RiskScorer, CorrelationAnalyzer, load_data
from visualizations import generate_all_charts

SEP = "─" * 65


def section(n, title):
    print(f"\n{'═'*65}")
    print(f"  SECTION {n}: {title}")
    print(f"{'═'*65}\n")


# ─────────────────────────────────────────────────────────────────────────────
# 0. SETUP
# ─────────────────────────────────────────────────────────────────────────────
section(0, "Setup & Data Loading")

df_wide, df_long = load_data()
runner           = SQLQueryRunner()

print(f"Wide table  : {df_wide.shape[0]:,} rows × {df_wide.shape[1]} cols")
print(f"Long table  : {df_long.shape[0]:,} rows × {df_long.shape[1]} cols")
print(f"Countries   : {df_wide['country_code'].nunique()}")
print(f"Year range  : {df_wide['year'].min()} – {df_wide['year'].max()}")
print(f"Indicators  : {df_long['indicator_name'].unique().tolist()}")
print(f"\nWide table columns:\n{list(df_wide.columns)}")
print(f"\nSample (wide):\n{df_wide.head(3).to_string(index=False)}")


# ─────────────────────────────────────────────────────────────────────────────
# 1. DATABASE OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────
section(1, "Database Overview")

print("Pipeline run log:")
runs = runner.query("SELECT * FROM pipeline_runs")
print(runs.to_string(index=False))

print("\nCountry coverage by region:")
region_counts = runner.query(
    "SELECT region, COUNT(*) as n_countries FROM country_meta GROUP BY region ORDER BY n_countries DESC"
)
print(region_counts.to_string(index=False))

print("\nData completeness (% of country-year-indicator cells filled):")
total_possible = df_wide["country_code"].nunique() * df_wide["year"].nunique()
indicators = ["gdp_growth", "inflation", "unemployment", "exports_pct_gdp", "govt_debt_pct_gdp"]
for ind in indicators:
    filled = df_wide[ind].notna().sum()
    pct    = filled / total_possible * 100
    print(f"  {ind:<25} {filled:>4}/{total_possible}  ({pct:.0f}%)")


# ─────────────────────────────────────────────────────────────────────────────
# 2. GLOBAL SHOCK DETECTION
# ─────────────────────────────────────────────────────────────────────────────
section(2, "Global Shock Detection (SQL View: v_shock_years)")

shocks = runner.get_shock_years()
print(shocks[["year","global_avg_gdp_growth","z_score","year_classification"]].to_string(index=False))

crisis_years = shocks[shocks["year_classification"] == "CRISIS"]
print(f"\n⚠  Crisis years detected: {crisis_years['year'].tolist()}")
print(f"   Methodology: z-score < -1.5 below long-run mean\n"
      f"   Long-run mean GDP growth: {shocks['long_run_mean'].iloc[0]:.2f}%\n"
      f"   Std deviation: {shocks['std_dev'].iloc[0]:.2f}%")


# ─────────────────────────────────────────────────────────────────────────────
# 3. REGIONAL GDP PERFORMANCE
# ─────────────────────────────────────────────────────────────────────────────
section(3, "Regional GDP Performance (SQL View: v_regional_gdp)")

regional = runner.get_regional_gdp()

decade_summary = (
    regional
    .assign(decade=lambda d: d["year"].apply(
        lambda y: "2000s" if y < 2010 else ("2010s" if y < 2020 else "2020s")
    ))
    .groupby(["region","decade"])[["avg_gdp_growth","avg_inflation","avg_unemployment"]]
    .mean()
    .round(2)
    .reset_index()
)
print("Average macroeconomic indicators by region and decade:")
print(decade_summary.to_string(index=False))

# Best and worst decades per region
print("\nFastest-growing region each decade:")
for decade in ["2000s","2010s","2020s"]:
    row = decade_summary[decade_summary["decade"] == decade].nlargest(1, "avg_gdp_growth")
    print(f"  {decade}: {row.iloc[0]['region']} ({row.iloc[0]['avg_gdp_growth']:.2f}%)")


# ─────────────────────────────────────────────────────────────────────────────
# 4. COUNTRY MACRO STABILITY SCORES
# ─────────────────────────────────────────────────────────────────────────────
section(4, "Country Macro Stability Scores")

# Via SQL view
sql_scores = runner.get_risk_scores()
print("SQL-computed scores (top 10):")
print(sql_scores[["country_name","region","avg_gdp_5yr","avg_inf_5yr",
                   "macro_stability_score","risk_tier"]].head(10).to_string(index=False))

# Via Python RiskScorer (cross-validation)
scorer    = RiskScorer(window_years=5)
py_scores = scorer.score(df_wide)
print("\nPython-computed scores (top 10):")
print(py_scores[["country_name","region","gdp_growth","inflation",
                  "unemployment","macro_stability_score","risk_tier"]].head(10).to_string(index=False))

print("\nRisk tier distribution:")
print(py_scores["risk_tier"].value_counts().to_string())


# ─────────────────────────────────────────────────────────────────────────────
# 5. TREND FORECASTING
# ─────────────────────────────────────────────────────────────────────────────
section(5, "Trend Forecasting (sklearn LinearRegression)")

forecaster  = TrendForecaster()
forecaster.fit(df_wide)
forecast_df = forecaster.forecast(n_years=5)

print(f"Forecast rows generated: {len(forecast_df):,}")
print(f"Forecast years: {sorted(forecast_df['year'].unique())}")

# Sample: GDP forecasts for G7 countries
g7 = ["USA","GBR","DEU","FRA","ITA","JPN","CAN"]
g7_fc = (
    forecast_df[
        (forecast_df["country_code"].isin(g7)) &
        (forecast_df["indicator"] == "gdp_growth")
    ]
    .pivot(index="country_code", columns="year", values="forecast_value")
    .round(2)
)
print("\nGDP Growth Forecasts — G7 Countries:")
print(g7_fc.to_string())

# Inflation forecasts for high-inflation economies
hi_inf = ["TUR","ARG","NGA","EGY"]
inf_fc = (
    forecast_df[
        (forecast_df["country_code"].isin(hi_inf)) &
        (forecast_df["indicator"] == "inflation")
    ]
    .pivot(index="country_code", columns="year", values="forecast_value")
    .round(1)
)
print("\nInflation Forecasts — High-Inflation Economies:")
print(inf_fc.to_string())


# ─────────────────────────────────────────────────────────────────────────────
# 6. CORRELATION ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
section(6, "Correlation Analysis")

analyzer = CorrelationAnalyzer()

print("Macro indicator correlation matrix (all countries, 2000–2023):")
corr = analyzer.indicator_correlation(df_wide)
print(corr.to_string())

print("\nGDP vs Inflation correlation by region:")
print(analyzer.gdp_vs_inflation_by_region(df_wide).to_string(index=False))

print("\nKey findings:")
gdp_inf_corr = df_wide[["gdp_growth","inflation"]].corr().iloc[0,1]
gdp_unemp    = df_wide[["gdp_growth","unemployment"]].corr().iloc[0,1]
print(f"  GDP ↔ Inflation    : r = {gdp_inf_corr:.3f} "
      f"({'negative — weak stagflation signal' if gdp_inf_corr < 0 else 'positive'})")
print(f"  GDP ↔ Unemployment : r = {gdp_unemp:.3f} "
      f"({'negative — Okun\'s Law confirmed' if gdp_unemp < 0 else 'positive'})")


# ─────────────────────────────────────────────────────────────────────────────
# 7. CRISIS RECOVERY ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
section(7, "Crisis Recovery Analysis (SQL View: v_crisis_recovery)")

recovery = runner.get_crisis_recovery()
print("Recovery data (first 20 rows):")
print(recovery.head(20).to_string(index=False))

summary = (
    recovery.dropna(subset=["years_to_recover"])
    .groupby(["crisis","region"])["years_to_recover"]
    .agg(["mean","min","max","count"])
    .round(1)
    .reset_index()
)
print("\nRecovery summary by crisis and region:")
print(summary.to_string(index=False))

for crisis in ["GFC","COVID"]:
    avg = recovery[recovery["crisis"] == crisis]["years_to_recover"].mean()
    print(f"\n  {crisis} avg recovery time: {avg:.1f} years")


# ─────────────────────────────────────────────────────────────────────────────
# 8. TOP / BOTTOM PERFORMERS
# ─────────────────────────────────────────────────────────────────────────────
section(8, "Top & Bottom Performers (2019–2023)")

performers = runner.get_top_bottom()
top    = performers[performers["performance_tier"] == "Top 10"]
bottom = performers[performers["performance_tier"] == "Bottom 10"]

print("Top 10 by avg GDP growth (2019–2023):")
print(top[["country_name","region","avg_gdp_5yr","avg_inf_5yr","avg_unemp_5yr"]].to_string(index=False))

print("\nBottom 10 by avg GDP growth (2019–2023):")
print(bottom[["country_name","region","avg_gdp_5yr","avg_inf_5yr","avg_unemp_5yr"]].to_string(index=False))


# ─────────────────────────────────────────────────────────────────────────────
# 9. FORECAST MODEL DIAGNOSTICS
# ─────────────────────────────────────────────────────────────────────────────
section(9, "Forecast Model Diagnostics")

model_summary = forecaster.get_model_summary()

print("R² distribution across all country-indicator models:")
print(model_summary["r2_score"].describe().round(3).to_string())

print("\nTop 10 best-fit models (highest R²):")
print(model_summary.nlargest(10, "r2_score")[
    ["country_name","indicator","r2_score","trend_slope","trend_direction"]
].to_string(index=False))

print("\nTrend direction summary (GDP growth):")
gdp_trends = model_summary[model_summary["indicator"] == "gdp_growth"]
print(gdp_trends["trend_direction"].value_counts().to_string())


# ─────────────────────────────────────────────────────────────────────────────
# 10. GENERATE ALL VISUALIZATIONS
# ─────────────────────────────────────────────────────────────────────────────
section(10, "Generating Visualizations")

generate_all_charts(
    df_wide     = df_wide,
    risk_df     = py_scores,
    forecast_df = forecast_df,
    recovery_df = recovery,
)

print(f"\n{'═'*65}")
print("  Analysis complete. All outputs saved.")
print(f"{'═'*65}")
