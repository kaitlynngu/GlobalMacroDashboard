# 🌍 Global Macro Dashboard

**End-to-end macroeconomic data pipeline, analysis, and forecasting across 40 countries (2000–2023)**

---

## Overview

This project mirrors the core workflow of a global economic research analyst: collecting macroeconomic data, loading it into a structured database, running analytical SQL queries, building forecasting models, and producing publication-ready visualizations.

It covers five indicators across 40 countries and six world regions, with a focus on:
- Data pipeline design and QC methodology
- Structured SQL analytics (views, aggregations, anomaly detection)
- Linear trend forecasting using scikit-learn
- Composite risk scoring with normalized, weighted components
- Crisis event detection and recovery speed analysis

**Stack:** Python · SQLite · SQL · pandas · scikit-learn · matplotlib · seaborn

---

## Project Structure

```
global-macro-dashboard/
│
├── pipeline.py               # Data pipeline: fetch → clean → QC → SQLite
├── generate_sample_data.py   # Synthetic data generator (for offline use)
├── analysis.py               # SQLQueryRunner, TrendForecaster, RiskScorer, CorrelationAnalyzer
├── visualizations.py         # All 8 charts (matplotlib + seaborn)
├── notebook.py               # Full analysis script (readable as a notebook)
│
├── sql/
│   └── queries.sql           # SQL views: regional aggregates, risk scores, crisis detection
│
├── data/
│   ├── raw/                  # Raw API output (macro_raw.csv)
│   ├── processed/            # Cleaned long + wide format CSVs
│   └── macro.db              # SQLite database (5 tables, 6 indexed views)
│
└── visualizations/           # Output charts (PNG)
    ├── 01_global_gdp_timeseries.png
    ├── 02_regional_gdp_heatmap.png
    ├── 03_macro_stability_scores.png
    ├── 04_gdp_vs_inflation_scatter.png
    ├── 05_gdp_forecasts.png
    ├── 06_correlation_heatmap.png
    ├── 07_crisis_recovery.png
    └── 08_unemployment_trends.png
```

---

## Data

**Source:** World Bank Open Data API (`api.worldbank.org/v2`)  
**Fallback:** `generate_sample_data.py` produces realistic synthetic data with calibrated country profiles and global shock events (2001, 2008–09, 2020) when API access is unavailable.

**Indicators collected:**

| Code | Name | Unit |
|------|------|------|
| NY.GDP.MKTP.KD.ZG | GDP growth | Annual % |
| FP.CPI.TOTL.ZG | Inflation (CPI) | Annual % |
| SL.UEM.TOTL.ZS | Unemployment | % of labor force |
| NE.EXP.GNFS.ZS | Exports of goods & services | % of GDP |
| GC.DOD.TOTL.GD.ZS | Central government debt | % of GDP |

**Coverage:** 40 countries · 6 regions · 24 years (2000–2023)

---

## Database Schema

```sql
macro_long      -- Tidy long-format (country, year, indicator, value, outlier_flag)
macro_wide      -- One row per country-year; one column per indicator
country_meta    -- Country codes, names, regions
pipeline_runs   -- Pipeline execution log (rows, coverage, timestamp)

-- SQL Views
v_shock_years           -- Z-score crisis detection
v_regional_gdp          -- Regional aggregates by year
v_risk_scores           -- Composite stability scores (2019–2023)
v_crisis_recovery       -- Years to recover from GFC and COVID
v_inflation_gdp_by_decade -- Decade × region summaries
v_top_bottom_performers -- Top/bottom 10 by recent GDP growth
```

---

## Analytical Modules

### `SQLQueryRunner`
Executes all SQL views and ad-hoc queries. Views are initialized automatically from `queries.sql` on first use.

### `TrendForecaster`
Fits a `LinearRegression` model (scikit-learn) per country × indicator combination. Generates 5-year point forecasts and reports R² and trend direction for all 200 models.

### `RiskScorer`
Builds a composite **Macro Stability Score (0–100)** using a configurable weighted average:
- GDP Growth: 40% (higher → lower risk)
- Inflation: 30% (lower → lower risk)
- Unemployment: 30% (lower → lower risk)

All components are min-max normalized before weighting.

### `CorrelationAnalyzer`
Computes Pearson correlation matrices across indicators and countries, including GDP–inflation correlation by region (stagflation risk signal).

---

## QC Methodology

Data quality checks run automatically in the pipeline:

1. **Country code validation** — removes World Bank regional aggregate rows (non-ISO-3 codes)
2. **Outlier flagging** — IQR method (4× IQR fence) flags but retains extreme values with `outlier_flag = True`
3. **Coverage check** — logs a warning for any country-indicator pair with <70% year coverage
4. **Pipeline log** — every run records row counts, country/indicator coverage, and timestamp to `pipeline_runs`

---

## Key Findings

| Finding | Detail |
|---------|--------|
| Crisis years detected | 2008, 2009, 2020 (z-score < −1.5 vs. long-run mean) |
| Fastest-growing region (all decades) | Asia-Pacific (avg 3.6–4.7% GDP growth) |
| Highest macro stability (2019–2023) | Vietnam, Philippines, India, Singapore |
| Highest risk economies | Argentina, Ukraine, Turkey (high inflation + volatility) |
| COVID recovery faster than GFC | Most regions returned to baseline growth within 1–2 years vs. 2–4 for GFC |
| Okun's Law confirmed | GDP growth and unemployment: r ≈ −0.25 across full panel |

---

## Visualizations

| Chart | Description |
|-------|-------------|
| `01_global_gdp_timeseries` | Global avg GDP growth with ±1 std dev band and crisis annotations |
| `02_regional_gdp_heatmap` | Heatmap: avg GDP per region × year, crisis years outlined in red |
| `03_macro_stability_scores` | Horizontal bar chart: all 40 countries ranked by stability score |
| `04_gdp_vs_inflation_scatter` | Scatter: GDP vs inflation with quadrant labels and unemployment sizing |
| `05_gdp_forecasts` | Historical + 5-year linear forecast for top 5 and bottom 5 countries |
| `06_correlation_heatmap` | Pearson correlation matrix of all five indicators |
| `07_crisis_recovery` | Grouped bar: avg years to recover from GFC vs COVID by region |
| `08_unemployment_trends` | Multi-line: regional unemployment over time with crisis markers |

---

## Setup & Usage

```bash
# 1. Clone and install dependencies
git clone https://github.com/YOUR_USERNAME/global-macro-dashboard
cd global-macro-dashboard
pip install -r requirements.txt

# 2a. Run the live data pipeline (requires internet access to api.worldbank.org)
python pipeline.py

# 2b. Or generate synthetic data offline
python generate_sample_data.py

# 3. Run the full analysis
python notebook.py

# 4. Generate charts only
python visualizations.py
```

---

## Requirements

```
pandas>=2.0
numpy>=1.24
matplotlib>=3.7
seaborn>=0.12
scikit-learn>=1.3
requests>=2.31
```

---

## Skills Demonstrated

- **Data pipeline engineering**: API ingestion, cleaning, QC flagging, SQLite loading with indexes
- **SQL analytics**: multi-table views, window-function-style aggregations, composite scoring in SQL
- **Statistical modeling**: linear regression forecasting, R² diagnostics, correlation analysis
- **Data visualization**: time series, heatmaps, scatter plots, bar charts (matplotlib + seaborn)
- **Macroeconomic analysis**: crisis detection, regional benchmarking, risk scoring methodology

---

*Data sourced from the World Bank Open Data API under the CC BY 4.0 license.*
