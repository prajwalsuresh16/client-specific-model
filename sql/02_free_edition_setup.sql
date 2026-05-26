-- Free Edition: minimal UC setup (run in notebook %sql or SQL Editor)
-- Replace catalog/schema if your workspace uses different defaults (e.g. main.default)

CREATE SCHEMA IF NOT EXISTS ${catalog}.${schema};

-- After pipeline runs, query KPIs:
-- SELECT * FROM ${catalog}.${schema}.gold_so_output LIMIT 10;
-- SELECT * FROM ${catalog}.${schema}.gold_validation_results WHERE segment_id = 'ALL';
