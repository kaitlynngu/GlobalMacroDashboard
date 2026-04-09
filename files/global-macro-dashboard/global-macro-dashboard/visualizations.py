"""
visualizations.py
-----------------
All charts for the Global Macro Dashboard.

Charts produced:
  1. Global GDP growth time series with crisis annotations
  2. Regional GDP heatmap by year
  3. Country macro stability score bar chart (risk ranking)
  4. GDP vs Inflation scatter with risk quadrants
  5. 5-year GDP forecast: top 10 vs bottom 10 countries
  6. Correlation heatmap of macroeconomic indicators
  7. Crisis recovery speed comparison (GFC vs COVID)
  8. Unemployment trends by region
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path

warnings.filterwarnings("ignore")

OUT_DIR = Path(__file__).parent / "visualizations"
OUT_DIR.mkdir(exist_ok=True)

# ── Style ─────────────────────────────────────────────────────────────────────
PALETTE = {
    "North America"      : "#2196F3",
    "Europe"             : "#4CAF50",
    "Asia-Pacific"       : "#FF9800",
    "Latin America"      : "#E91E63",
    "Middle East & Africa": "#9C27B0",
    "Eastern Europe"     : "#F44336",
}

CRISIS_YEARS = {2001: "Dot-com\nbust", 2009: "GFC\ntrough", 2020: "COVID-19"}

plt.rcParams.update({
    "font.family"     : "DejaVu Sans",
    "axes.spines.top" : False,
    "axes.spines.right": False,
    "axes.grid"       : True,
    "grid.alpha"      : 0.3,
    "figure.dpi"      : 120,
})


def save(fig, name: str):
    path = OUT_DIR / f"{name}.png"
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved → {path.name}")


# ── Chart 1: Global GDP Time Series ──────────────────────────────────────────

def plot_global_gdp_timeseries(df_wide: pd.DataFrame):
    """Global average GDP growth 2000-2023 with crisis bands."""
    annual = df_wide.groupby("year")["gdp_growth"].agg(["mean", "std"]).reset_index()
    annual.columns = ["year", "mean", "std"]

    fig, ax = plt.subplots(figsize=(12, 5))

    # Crisis shading
    for yr, label in CRISIS_YEARS.items():
        ax.axvspan(yr - 0.5, yr + 0.5, alpha=0.15, color="#F44336", zorder=1)
        ax.text(yr, annual["mean"].min() - 1.2, label,
                ha="center", fontsize=7.5, color="#B71C1C", style="italic")

    # Confidence band
    ax.fill_between(annual["year"],
                    annual["mean"] - annual["std"],
                    annual["mean"] + annual["std"],
                    alpha=0.15, color="#2196F3", label="±1 Std Dev (cross-country)")

    ax.plot(annual["year"], annual["mean"], color="#1565C0",
            linewidth=2.5, marker="o", markersize=4, label="Global Average GDP Growth")
    ax.axhline(0, color="#666", linewidth=0.8, linestyle="--")

    ax.set_title("Global GDP Growth (2000–2023): Mean Across 40 Countries",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Year")
    ax.set_ylabel("GDP Growth (%)")
    ax.legend(fontsize=9)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(2))
    fig.tight_layout()
    save(fig, "01_global_gdp_timeseries")


# ── Chart 2: Regional GDP Heatmap ────────────────────────────────────────────

def plot_regional_gdp_heatmap(df_wide: pd.DataFrame):
    """Heatmap: avg GDP growth per region × year (2000–2023)."""
    pivot = (
        df_wide.groupby(["region", "year"])["gdp_growth"]
        .mean()
        .unstack(level="year")
        .round(2)
    )

    fig, ax = plt.subplots(figsize=(16, 5))
    sns.heatmap(
        pivot, ax=ax,
        cmap="RdYlGn", center=0, vmin=-6, vmax=10,
        annot=True, fmt=".1f", annot_kws={"size": 7},
        linewidths=0.4, linecolor="#ddd",
        cbar_kws={"label": "GDP Growth (%)", "shrink": 0.6},
    )

    # Mark crisis years
    years = list(pivot.columns)
    for yr in CRISIS_YEARS:
        if yr in years:
            col_idx = years.index(yr)
            ax.add_patch(plt.Rectangle((col_idx, 0), 1, len(pivot),
                         fill=False, edgecolor="#F44336", lw=2.5))

    ax.set_title("Regional GDP Growth Heatmap (2000–2023)\n"
                 "Red borders = global crisis years", fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Year")
    ax.set_ylabel("")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    fig.tight_layout()
    save(fig, "02_regional_gdp_heatmap")


# ── Chart 3: Macro Stability Scores ──────────────────────────────────────────

def plot_risk_scores(risk_df: pd.DataFrame):
    """Horizontal bar chart of macro stability scores, colour-coded by risk tier."""
    df = risk_df.copy().sort_values("macro_stability_score")

    tier_colors = {"Low Risk": "#388E3C", "Medium Risk": "#F57C00", "High Risk": "#C62828"}
    colors = df["risk_tier"].map(tier_colors).fillna("#999")

    fig, ax = plt.subplots(figsize=(10, 11))
    bars = ax.barh(df["country_name"], df["macro_stability_score"],
                   color=colors, edgecolor="white", linewidth=0.4)

    # Score labels
    for bar, val in zip(bars, df["macro_stability_score"]):
        ax.text(val + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{val:.0f}", va="center", fontsize=7.5)

    # Legend
    patches = [mpatches.Patch(color=c, label=t) for t, c in tier_colors.items()]
    ax.legend(handles=patches, loc="lower right", fontsize=9)

    ax.set_xlim(0, 108)
    ax.set_title("Macro Stability Score by Country (2019–2023)\n"
                 "Composite: GDP Growth 40% · Inflation 30% · Unemployment 30%",
                 fontsize=11, fontweight="bold", pad=10)
    ax.set_xlabel("Macro Stability Score (0–100)")
    ax.tick_params(axis="y", labelsize=8)
    fig.tight_layout()
    save(fig, "03_macro_stability_scores")


# ── Chart 4: GDP vs Inflation Scatter ────────────────────────────────────────

def plot_gdp_vs_inflation(df_wide: pd.DataFrame):
    """Scatter of 5-yr avg GDP growth vs inflation, sized by unemployment."""
    last = df_wide["year"].max()
    recent = (
        df_wide[df_wide["year"] >= last - 4]
        .groupby(["country_code", "country_name", "region"])
        [["gdp_growth", "inflation", "unemployment"]]
        .mean()
        .reset_index()
    )
    recent = recent[recent["inflation"] < 50]  # exclude hyperinflation outliers for viz

    fig, ax = plt.subplots(figsize=(11, 7))

    for region, grp in recent.groupby("region"):
        sc = ax.scatter(
            grp["inflation"], grp["gdp_growth"],
            s=grp["unemployment"] * 8 + 20,
            alpha=0.75,
            color=PALETTE.get(region, "#999"),
            label=region, edgecolors="white", linewidth=0.5,
        )

    # Quadrant lines
    ax.axhline(recent["gdp_growth"].median(), color="#666", lw=0.8, ls="--")
    ax.axvline(recent["inflation"].median(),  color="#666", lw=0.8, ls="--")

    # Quadrant labels
    xlim = ax.get_xlim(); ylim = ax.get_ylim()
    kw = dict(fontsize=8, color="#555", alpha=0.7, style="italic")
    ax.text(xlim[0] + 0.3, ylim[1] - 0.4, "Low inflation\nHigh growth\n✅ Ideal", **kw)
    ax.text(xlim[1] - 3,   ylim[0] + 0.2, "High inflation\nLow growth\n⚠ Stagflation", **kw)

    # Country labels for notable outliers
    for _, row in recent.iterrows():
        if row["inflation"] > 20 or row["gdp_growth"] > 8 or row["gdp_growth"] < -3:
            ax.annotate(row["country_code"], (row["inflation"], row["gdp_growth"]),
                        fontsize=7, xytext=(3, 3), textcoords="offset points")

    ax.set_title("GDP Growth vs Inflation (5-Year Avg, 2019–2023)\n"
                 "Bubble size = unemployment rate", fontsize=11, fontweight="bold", pad=10)
    ax.set_xlabel("Avg Inflation Rate (%)")
    ax.set_ylabel("Avg GDP Growth (%)")
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    save(fig, "04_gdp_vs_inflation_scatter")


# ── Chart 5: GDP Forecasts ────────────────────────────────────────────────────

def plot_gdp_forecasts(df_wide: pd.DataFrame, forecast_df: pd.DataFrame):
    """
    Line chart: historical GDP growth + 5-year forecast for
    top 5 and bottom 5 countries by recent average growth.
    """
    last = df_wide["year"].max()
    avg  = (
        df_wide[df_wide["year"] >= last - 4]
        .groupby("country_name")["gdp_growth"].mean()
        .sort_values()
    )
    bottom5 = avg.head(5).index.tolist()
    top5    = avg.tail(5).index.tolist()
    selected = bottom5 + top5

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    for ax, group, title_suffix in zip(
        axes,
        [bottom5, top5],
        ["Bottom 5 Countries by Avg GDP Growth", "Top 5 Countries by Avg GDP Growth"]
    ):
        for i, country in enumerate(group):
            hist = df_wide[df_wide["country_name"] == country].sort_values("year")
            fc   = forecast_df[
                (forecast_df["country_name"] == country) &
                (forecast_df["indicator"] == "gdp_growth")
            ].sort_values("year")

            color = plt.cm.tab10(i)
            ax.plot(hist["year"], hist["gdp_growth"],
                    color=color, linewidth=1.8, label=country)
            if not fc.empty:
                # Connect historical endpoint to forecast
                last_hist = hist.iloc[-1]
                fc_extended = pd.concat([
                    pd.DataFrame([{"year": last_hist["year"],
                                   "forecast_value": last_hist["gdp_growth"]}]),
                    fc[["year", "forecast_value"]]
                ])
                ax.plot(fc_extended["year"], fc_extended["forecast_value"],
                        color=color, linewidth=1.8, linestyle="--", alpha=0.7)

        ax.axvspan(last + 0.5, last + 5.5, alpha=0.07, color="#FF9800")
        ax.text(last + 2.8, ax.get_ylim()[1] * 0.9, "Forecast",
                color="#E65100", fontsize=8, style="italic")
        ax.axhline(0, color="#999", lw=0.8, ls="--")
        ax.set_title(title_suffix, fontsize=10, fontweight="bold")
        ax.set_xlabel("Year")
        ax.legend(fontsize=8)

    axes[0].set_ylabel("GDP Growth (%)")
    fig.suptitle("GDP Growth: Historical Trend + 5-Year Linear Forecast",
                 fontsize=12, fontweight="bold", y=1.01)
    fig.tight_layout()
    save(fig, "05_gdp_forecasts")


# ── Chart 6: Correlation Heatmap ─────────────────────────────────────────────

def plot_correlation_heatmap(df_wide: pd.DataFrame):
    """Pearson correlation matrix across all macro indicators."""
    cols    = ["gdp_growth", "inflation", "unemployment",
               "exports_pct_gdp", "govt_debt_pct_gdp"]
    labels  = ["GDP Growth", "Inflation", "Unemployment",
               "Exports\n(% GDP)", "Govt Debt\n(% GDP)"]
    corr_mx = df_wide[cols].corr()

    mask = np.triu(np.ones_like(corr_mx, dtype=bool), k=1)

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        corr_mx, ax=ax, mask=mask,
        cmap="coolwarm", center=0, vmin=-1, vmax=1,
        annot=True, fmt=".2f", annot_kws={"size": 10},
        linewidths=0.5, square=True,
        xticklabels=labels, yticklabels=labels,
        cbar_kws={"shrink": 0.7, "label": "Pearson r"},
    )
    ax.set_title("Macro Indicator Correlation Matrix\n(All Countries, 2000–2023)",
                 fontsize=11, fontweight="bold", pad=10)
    plt.xticks(fontsize=9); plt.yticks(fontsize=9, rotation=0)
    fig.tight_layout()
    save(fig, "06_correlation_heatmap")


# ── Chart 7: Crisis Recovery ──────────────────────────────────────────────────

def plot_crisis_recovery(recovery_df: pd.DataFrame):
    """Grouped bar chart: years to recover from GFC vs COVID by region."""
    df = recovery_df.dropna(subset=["years_to_recover"])
    df = df[df["years_to_recover"] >= 0]

    summary = (
        df.groupby(["region", "crisis"])["years_to_recover"]
        .mean()
        .reset_index()
        .round(1)
    )

    pivot = summary.pivot(index="region", columns="crisis", values="years_to_recover").fillna(0)

    fig, ax = plt.subplots(figsize=(10, 5))
    x     = np.arange(len(pivot))
    width = 0.35
    ax.bar(x - width/2, pivot.get("GFC",   0), width, label="GFC (2009)",   color="#1565C0", alpha=0.85)
    ax.bar(x + width/2, pivot.get("COVID", 0), width, label="COVID (2020)", color="#E65100", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, fontsize=9)
    ax.set_ylabel("Avg Years to Recover")
    ax.set_title("Average Years to GDP Recovery After Crisis: GFC vs COVID\n"
                 "(recovery = return to 80% of pre-crisis baseline growth)",
                 fontsize=11, fontweight="bold", pad=10)
    ax.legend()
    ax.set_ylim(0, pivot.values.max() * 1.25 + 0.5)

    for bar in ax.patches:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.05,
                    f"{h:.1f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    save(fig, "07_crisis_recovery")


# ── Chart 8: Unemployment by Region ──────────────────────────────────────────

def plot_unemployment_trends(df_wide: pd.DataFrame):
    """Multi-line chart: regional average unemployment over time."""
    regional = (
        df_wide.groupby(["region", "year"])["unemployment"]
        .mean()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(12, 5))

    for region, grp in regional.groupby("region"):
        ax.plot(grp["year"], grp["unemployment"],
                label=region, color=PALETTE.get(region, "#999"),
                linewidth=2, marker="o", markersize=3)

    for yr, label in CRISIS_YEARS.items():
        ax.axvline(yr, color="#F44336", lw=0.8, ls="--", alpha=0.5)

    ax.set_title("Regional Unemployment Trends (2000–2023)",
                 fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Year")
    ax.set_ylabel("Unemployment Rate (%)")
    ax.legend(fontsize=8, loc="upper right")
    ax.xaxis.set_major_locator(mticker.MultipleLocator(2))
    fig.tight_layout()
    save(fig, "08_unemployment_trends")


# ── Main ──────────────────────────────────────────────────────────────────────

def generate_all_charts(df_wide, risk_df, forecast_df, recovery_df):
    print("Generating visualizations ...")
    plot_global_gdp_timeseries(df_wide)
    plot_regional_gdp_heatmap(df_wide)
    plot_risk_scores(risk_df)
    plot_gdp_vs_inflation(df_wide)
    plot_gdp_forecasts(df_wide, forecast_df)
    plot_correlation_heatmap(df_wide)
    plot_crisis_recovery(recovery_df)
    plot_unemployment_trends(df_wide)
    print(f"\nAll charts saved to: {OUT_DIR}")


if __name__ == "__main__":
    from analysis import SQLQueryRunner, TrendForecaster, RiskScorer, load_data

    df_wide, _ = load_data()
    runner      = SQLQueryRunner()
    risk_df     = runner.get_risk_scores()
    recovery_df = runner.get_crisis_recovery()

    forecaster  = TrendForecaster().fit(df_wide)
    forecast_df = forecaster.forecast(n_years=5)

    generate_all_charts(df_wide, risk_df, forecast_df, recovery_df)
