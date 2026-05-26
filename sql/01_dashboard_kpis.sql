-- KPI queries for Lakeview Dashboard (SQL Warehouse)
-- Replace ${catalog}.${schema} with your Unity Catalog binding

USE CATALOG ${catalog};
USE SCHEMA ${schema};

-- Mail volume & keep rate
SELECT
  campaign_id,
  COUNT(*) AS sd_rows,
  SUM(CASE WHEN keep_flag = 'Y' THEN 1 ELSE 0 END) AS mail_count,
  ROUND(100.0 * SUM(CASE WHEN keep_flag = 'Y' THEN 1 ELSE 0 END) / COUNT(*), 2) AS mail_pct
FROM ${catalog}.${schema}.gold_so_output
GROUP BY campaign_id;

-- Validation: top model combinations (AUC + PPPM)
SELECT
  combo_id,
  model_a,
  model_b,
  ROUND(auc, 4) AS auc,
  ROUND(pppm_score, 4) AS pppm_score,
  ROUND(pppm_corr, 4) AS pppm_corr,
  segment_id
FROM ${catalog}.${schema}.gold_validation_results
WHERE segment_id = 'ALL'
ORDER BY auc DESC
LIMIT 25;

-- LOL segment × decile counts
SELECT
  segment_id,
  decile,
  name_count,
  ROUND(mean_rank_mix, 2) AS mean_rank_mix
FROM ${catalog}.${schema}.gold_list_of_lists
ORDER BY segment_id, decile;
