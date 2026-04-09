-- =============================================================================
-- queries.sql
-- Global Macro Dashboard — Analytical SQL Layer
-- =============================================================================
-- These queries run against macro.db and power the analysis notebook.
-- Each section represents a distinct analytical task.
-- =============================================================================


-- -----------------------------------------------------------------------------
-- 1. REGIONAL GDP PERFORMANCE OVERVIEW
--    Average GDP growth by region and decade
-- -----------------------------------------------------------------------------

CREATE VIEW IF NOT EXISTS v_regional_gdp AS
SELECT
    cm.region,
    mw.year,
    CASE
        WHEN mw.year BETWEEN 2000 AND 2009 THEN '2000s'
        WHEN mw.year BETWEEN 2010 AND 2019 THEN '2010s'
        ELSE '2020s'
    END AS decade,
    ROUND(AVG(mw.gdp_growth), 2)        AS avg_gdp_growth,
    ROUND(AVG(mw.inflation), 2)         AS avg_inflation,
    ROUND(AVG(mw.unemployment), 2)      AS avg_unemployment,
    COUNT(DISTINCT mw.country_code)     AS n_countries
FROM macro_wide mw
JOIN country_meta cm ON mw.country_code = cm.country_code
GROUP BY cm.region, mw.year;


-- -----------------------------------------------------------------------------
-- 2. GLOBAL SHOCK DETECTION
--    Years where global avg GDP growth was > 1.5 std devs below mean
--    (identifies crisis years: 2001, 2009, 2020)
-- -----------------------------------------------------------------------------

CREATE VIEW IF NOT EXISTS v_shock_years AS
WITH global_stats AS (
    SELECT
        year,
        AVG(gdp_growth) AS avg_gdp
    FROM macro_wide
    GROUP BY year
),
stats AS (
    SELECT
        AVG(avg_gdp)                        AS mean_gdp,
        -- SQLite lacks STDDEV; compute manually
        SQRT(AVG((avg_gdp - (SELECT AVG(avg_gdp) FROM global_stats)) *
                 (avg_gdp - (SELECT AVG(avg_gdp) FROM global_stats))))  AS std_gdp
    FROM global_stats
)
SELECT
    gs.year,
    ROUND(gs.avg_gdp, 2)                            AS global_avg_gdp_growth,
    ROUND(st.mean_gdp, 2)                           AS long_run_mean,
    ROUND(st.std_gdp, 2)                            AS std_dev,
    ROUND((gs.avg_gdp - st.mean_gdp) / st.std_gdp, 2) AS z_score,
    CASE WHEN (gs.avg_gdp - st.mean_gdp) / st.std_gdp < -1.5
         THEN 'CRISIS' ELSE 'Normal' END             AS year_classification
FROM global_stats gs, stats st
ORDER BY gs.year;


-- -----------------------------------------------------------------------------
-- 3. COUNTRY RISK SCORING
--    Composite score (0-100) based on:
--      - Average GDP growth    (higher = lower risk, weight 40%)
--      - Average inflation     (lower = lower risk, weight 30%)
--      - Average unemployment  (lower = lower risk, weight 30%)
--    Scored over the most recent 5-year window (2019-2023)
-- -----------------------------------------------------------------------------

CREATE VIEW IF NOT EXISTS v_risk_scores AS
WITH recent AS (
    SELECT
        mw.country_code,
        cm.country_name,
        cm.region,
        AVG(mw.gdp_growth)   AS avg_gdp,
        AVG(mw.inflation)    AS avg_inf,
        AVG(mw.unemployment) AS avg_unemp
    FROM macro_wide mw
    JOIN country_meta cm ON mw.country_code = cm.country_code
    WHERE mw.year BETWEEN 2019 AND 2023
    GROUP BY mw.country_code
),
minmax AS (
    SELECT
        MIN(avg_gdp)  AS min_gdp,  MAX(avg_gdp)  AS max_gdp,
        MIN(avg_inf)  AS min_inf,  MAX(avg_inf)  AS max_inf,
        MIN(avg_unemp)AS min_unemp,MAX(avg_unemp)AS max_unemp
    FROM recent
)
SELECT
    r.country_code,
    r.country_name,
    r.region,
    ROUND(r.avg_gdp,  2) AS avg_gdp_5yr,
    ROUND(r.avg_inf,  2) AS avg_inf_5yr,
    ROUND(r.avg_unemp,2) AS avg_unemp_5yr,
    -- Normalize each dimension 0-1, then compute weighted score
    ROUND(
        (
          -- GDP: higher is better → normalize ascending
          0.40 * ((r.avg_gdp   - m.min_gdp)   / NULLIF(m.max_gdp   - m.min_gdp,  0)) +
          -- Inflation: lower is better → normalize descending
          0.30 * (1 - (r.avg_inf   - m.min_inf)   / NULLIF(m.max_inf   - m.min_inf,  0)) +
          -- Unemployment: lower is better → normalize descending
          0.30 * (1 - (r.avg_unemp - m.min_unemp) / NULLIF(m.max_unemp - m.min_unemp,0))
        ) * 100
    , 1) AS macro_stability_score,
    CASE
        WHEN (
          0.40 * ((r.avg_gdp   - m.min_gdp)   / NULLIF(m.max_gdp   - m.min_gdp,  0)) +
          0.30 * (1 - (r.avg_inf   - m.min_inf)   / NULLIF(m.max_inf   - m.min_inf,  0)) +
          0.30 * (1 - (r.avg_unemp - m.min_unemp) / NULLIF(m.max_unemp - m.min_unemp,0))
        ) >= 0.70 THEN 'Low Risk'
        WHEN (
          0.40 * ((r.avg_gdp   - m.min_gdp)   / NULLIF(m.max_gdp   - m.min_gdp,  0)) +
          0.30 * (1 - (r.avg_inf   - m.min_inf)   / NULLIF(m.max_inf   - m.min_inf,  0)) +
          0.30 * (1 - (r.avg_unemp - m.min_unemp) / NULLIF(m.max_unemp - m.min_unemp,0))
        ) >= 0.40 THEN 'Medium Risk'
        ELSE 'High Risk'
    END AS risk_tier
FROM recent r, minmax m
ORDER BY macro_stability_score DESC;


-- -----------------------------------------------------------------------------
-- 4. GDP RECOVERY SPEED AFTER CRISES
--    Measures years to return to pre-crisis GDP growth level
--    post-2009 GFC and post-2020 COVID
-- -----------------------------------------------------------------------------

CREATE VIEW IF NOT EXISTS v_crisis_recovery AS
WITH pre_gfc AS (
    SELECT country_code, AVG(gdp_growth) AS baseline_gdp
    FROM macro_wide WHERE year BETWEEN 2005 AND 2007
    GROUP BY country_code
),
post_gfc AS (
    SELECT mw.country_code, mw.year, mw.gdp_growth,
           pg.baseline_gdp,
           mw.gdp_growth >= pg.baseline_gdp * 0.80 AS recovered
    FROM macro_wide mw
    JOIN pre_gfc pg ON mw.country_code = pg.country_code
    WHERE mw.year BETWEEN 2010 AND 2015
),
first_recovery_gfc AS (
    SELECT country_code, MIN(year) AS recovery_year, 'GFC' AS crisis
    FROM post_gfc WHERE recovered = 1
    GROUP BY country_code
),
pre_covid AS (
    SELECT country_code, AVG(gdp_growth) AS baseline_gdp
    FROM macro_wide WHERE year BETWEEN 2017 AND 2019
    GROUP BY country_code
),
post_covid AS (
    SELECT mw.country_code, mw.year, mw.gdp_growth,
           pc.baseline_gdp,
           mw.gdp_growth >= pc.baseline_gdp * 0.80 AS recovered
    FROM macro_wide mw
    JOIN pre_covid pc ON mw.country_code = pc.country_code
    WHERE mw.year BETWEEN 2021 AND 2023
),
first_recovery_covid AS (
    SELECT country_code, MIN(year) AS recovery_year, 'COVID' AS crisis
    FROM post_covid WHERE recovered = 1
    GROUP BY country_code
)
SELECT
    cm.country_name,
    cm.region,
    r.crisis,
    r.recovery_year,
    CASE r.crisis
        WHEN 'GFC'   THEN r.recovery_year - 2009
        WHEN 'COVID' THEN r.recovery_year - 2020
    END AS years_to_recover
FROM (
    SELECT * FROM first_recovery_gfc
    UNION ALL
    SELECT * FROM first_recovery_covid
) r
JOIN country_meta cm ON r.country_code = cm.country_code
ORDER BY r.crisis, years_to_recover;


-- -----------------------------------------------------------------------------
-- 5. INFLATION VS GDP GROWTH CORRELATION BY DECADE
--    Used to assess stagflation risk across time periods
-- -----------------------------------------------------------------------------

CREATE VIEW IF NOT EXISTS v_inflation_gdp_by_decade AS
SELECT
    cm.region,
    CASE
        WHEN mw.year BETWEEN 2000 AND 2009 THEN '2000s'
        WHEN mw.year BETWEEN 2010 AND 2019 THEN '2010s'
        ELSE '2020s'
    END AS decade,
    ROUND(AVG(mw.gdp_growth), 2)  AS avg_gdp_growth,
    ROUND(AVG(mw.inflation), 2)   AS avg_inflation,
    ROUND(AVG(mw.unemployment), 2)AS avg_unemployment,
    COUNT(*)                       AS observations
FROM macro_wide mw
JOIN country_meta cm ON mw.country_code = cm.country_code
GROUP BY cm.region, decade
ORDER BY cm.region, decade;


-- -----------------------------------------------------------------------------
-- 6. TOP & BOTTOM PERFORMERS — Most recent 5 years
-- -----------------------------------------------------------------------------

CREATE VIEW IF NOT EXISTS v_top_bottom_performers AS
WITH scored AS (
    SELECT
        mw.country_code,
        cm.country_name,
        cm.region,
        ROUND(AVG(mw.gdp_growth),   2) AS avg_gdp_5yr,
        ROUND(AVG(mw.inflation),    2) AS avg_inf_5yr,
        ROUND(AVG(mw.unemployment), 2) AS avg_unemp_5yr,
        ROW_NUMBER() OVER (ORDER BY AVG(mw.gdp_growth) DESC) AS gdp_rank_desc,
        ROW_NUMBER() OVER (ORDER BY AVG(mw.gdp_growth) ASC)  AS gdp_rank_asc
    FROM macro_wide mw
    JOIN country_meta cm ON mw.country_code = cm.country_code
    WHERE mw.year BETWEEN 2019 AND 2023
    GROUP BY mw.country_code
)
SELECT
    *,
    CASE WHEN gdp_rank_desc <= 10 THEN 'Top 10'
         WHEN gdp_rank_asc  <= 10 THEN 'Bottom 10'
         ELSE 'Middle' END AS performance_tier
FROM scored
WHERE gdp_rank_desc <= 10 OR gdp_rank_asc <= 10
ORDER BY avg_gdp_5yr DESC;
