# FMG Client-Specific Modeling Pipeline (Regional Bank)

End-to-end replication of the FMG AS-IS architecture (diagram steps 1–9 → notebooks `01`–`08` + `00` synthetic data).

## Git / data policy

- **Commit:** Python code, configs, notebooks only.
- **Do not commit:** `data/regional_bank/bronze/**`, `bronze_prior/**`, silver/gold Parquet (see `.gitignore`).
- **After clone on Databricks:** run `python scripts/generate_data.py --rows 5000000` (creates **current + prior** bronze), then `01`–`08`.

## Products (one campaign / product at a time)

| Code | Product |
|------|---------|
| ADD | Accidental Death & Dismemberment |
| HAP | Hospital Accident Protection |
| RC | Recuperative Care |
| TERM_LIFE | Term Life Insurance |
| APP | Accidental Protection Plan |

Change `campaign.product_code` in `config/settings.yaml` per run.

## Quick start

```bash
pip install -r requirements.txt
export GROQ_API_KEY=your_key   # optional; rule-based FE fallback

# On Databricks after clone:
python scripts/generate_data.py --rows 5000000
python scripts/run_pipeline.py --sample-n 500000
```

## GROQ feature engineering

Uses **`llama-3.3-70b-versatile`** (best Groq option for structured JSON feature shortlisting). Set Databricks secret `GROQ_API_KEY`.

## Step 8 selection (legacy-aligned)

Configured in `config/selection_rules.yaml`:

- Per **composite segment_id** (age × news × tenure), top **40%** by rank-mix within eligible deciles
- **News P / News D** names get extra decile depth (transcript manual intervention pattern)
- Optional **manual CSV** overrides: `config/manual_selection_overrides.csv`
- Optional **PPPM trim** for auto jobs (`pppm_trim.enabled`)

## Legacy parity (what each step does now)

| Step | Legacy behavior implemented |
|------|----------------------------|
| 01 | Wide MRGAL join (SD+STAT), dedupe, marketable flags; **current + prior** silver tables |
| 02 | **Prior campaign** labeled train/val; **current** scoring only; stratified split |
| 03 | GROQ shortlist + **WOE/index reports**; **generic / product / client_product** tiers |
| 04 | **Rank-based** combo validation, AUC + **PPPM score**, orthogonality filter, segment grids |
| 05 | Weighted AUC/PPPM recommendation + stability archive |
| 06 | Probability → rank → **average rank** (rank-mix) |
| 07 | **Composite segment** × decile LOL + `list_key` |
| 08 | LOL-guided selection, SO **100% SD parity**, optional PPPM trim |

## Layout

```
client-specific-model/
  config/          settings, segments, selection_rules
  notebooks/       00–08
  scripts/         generate_data.py, run_pipeline.py
  src/
    pipeline/      steps 01–08
    mrgal_builder.py, indexing.py, segments.py, metrics.py
  data/            gitignored (bronze, bronze_prior, silver, gold)
```

## Env flags

| Variable | Purpose |
|----------|---------|
| `GROQ_API_KEY` | GROQ feature shortlist (optional) |
| `FMG_ROW_COUNT` | Override synthetic row count |
| `FMG_ALLOW_SYNTHETIC_RESPONDERS=true` | POC only: inflate responders if prior join is thin |
| `FMG_UC_CATALOG` / `FMG_UC_SCHEMA` | Unity Catalog binding on Databricks |
| `FMG_SQL_WAREHOUSE_ID` | SQL Warehouse for SQL tasks & dashboards |
| `FMG_FORCE_LOCAL_IO=true` | Force Parquet even on Databricks cluster |

---

## Databricks platform (end-to-end)

The repo is a **Databricks Asset Bundle** that wires every major platform capability into the modeling pipeline.

| Platform feature | Where in repo |
|------------------|---------------|
| **Unity Catalog** | `config/databricks.yaml`, `sql/00_setup_catalog.sql`, `src/databricks/uc_io.py` |
| **Delta tables** | All gold/silver/bronze via `src/dataset_io.py` → `saveAsTable` on cluster |
| **Volumes** | `regional_bank_bronze_vol` — raw file landing |
| **Lakeflow / DLT** | `pipelines/dlt_medallion.py`, `resources/pipeline_dlt.yml` |
| **Workflows / multi-task Jobs** | `resources/job_modeling_e2e.yml` — tasks 00–09 + SQL + DLT |
| **SQL Warehouse** | SQL tasks: catalog setup + dashboard KPIs (`sql/*.sql`) |
| **Lakeview Dashboard** | `dashboards/fmg_modeling_dashboard.lvdash.json`, `resources/dashboard.yml` |
| **MLflow + UC Model Registry** | `src/databricks/mlflow_setup.py`, experiment in `resources/experiments.yml` |
| **Feature Store** | `src/databricks/feature_store.py` (step 03) |
| **Model Serving** | `notebooks/09_model_serving_and_monitoring.py` |
| **Secrets** | Scope `fmg` / key `groq_api_key` (notebook bootstrap) |
| **Task values** | Job orchestration between tasks (`src/databricks/runtime.py`) |
| **Notebooks** | `notebooks/00`–`09` with `bootstrap_notebook()` |
| **Cluster** | Job cluster spec in `resources/job_modeling_e2e.yml` |

### Deploy & run on Databricks

1. Install [Databricks CLI](https://docs.databricks.com/dev-tools/cli/databricks-cli.html) and configure auth.
2. Edit `databricks.yml` → set `var.databricks_host` and `var.warehouse_id` for your workspace.
3. Create secret scope:

   ```text
   databricks secrets create-scope fmg
   databricks secrets put-secret fmg groq_api_key --string-value '<GROQ_KEY>'
   ```

4. Deploy bundle:

   ```bash
   databricks bundle validate
   databricks bundle deploy -t dev
   ```

5. Run full E2E job (synthetic → DLT → steps 01–08 → SQL dashboard refresh → model serving):

   ```bash
   databricks bundle run fmg_modeling_e2e -t dev
   ```

6. Open **Workflows** → `FMG Regional Bank Modeling E2E`, **Lakeflow** pipeline `FMG Medallion Bronze-Silver DLT`, **SQL** dashboard `FMG Regional Bank Modeling Ops`, and **MLflow** experiment `/Shared/fmg_regional_bank_modeling`.

### Local dev (unchanged)

`python scripts/run_pipeline.py` still uses Parquet under `data/`. On a Databricks cluster, I/O automatically switches to Unity Catalog unless `FMG_FORCE_LOCAL_IO=true`.

---

## Databricks Free Edition (personal account) — recommended path

Free Edition is **serverless-only** with quotas. This repo ships a profile that uses the **best-supported** features without paid-only dependencies.

| Feature | Free Edition | How this repo uses it |
|---------|--------------|------------------------|
| **Unity Catalog + Delta** | Yes | Primary storage (`main.fmg_regional_bank.*` or auto-detected catalog) |
| **DBFS Delta fallback** | Yes | If UC write fails → `dbfs:/FileStore/fmg/...` + external table |
| **Notebooks + Widgets** | Yes | `bootstrap_notebook()` + edition/row_count/groq widgets |
| **MLflow tracking** | Yes | Workspace experiments; UC registry optional |
| **Jobs (1 task)** | Yes | `fmg_free_e2e` → `00_orchestrator_e2e.py` (saves task quota) |
| **SQL / %sql KPIs** | Yes | End of orchestrator + `sql/02_free_edition_setup.sql` |
| **Lakeflow / DLT** | Quota | Inline in orchestrator (no separate DLT job by default) |
| **GROQ API** | May be blocked on some workspaces | **On by default** when key is set; rule-based fallback only if API fails |
| **Custom clusters** | No | Job YAML has no `job_cluster_key` (serverless) |
| **GPU model serving** | No | Step 09 skipped on free tier |
| **Feature Store** | Quota | Off — use `gold_feature_matrix` Delta table |

### Quick start (Free Edition)

1. Clone repo to workspace (Repos: `/Repos/<user>/client-specific-model`).
2. Open **`notebooks/00_orchestrator_e2e.py`** → Run all (or attach to job below).
3. Optional — deploy one serverless job:

   ```bash
   export FMG_DATABRICKS_TIER=free
   databricks bundle deploy -t free
   databricks bundle run fmg_free_e2e -t free
   ```

4. Set widgets: `edition=free`, `row_count=100000`, `use_groq=false`, catalog/schema = your workspace defaults.

5. View results: **Data** → schema `fmg_regional_bank` → tables `gold_so_output`, `gold_validation_results`, etc.  
   **MLflow** → experiment `/Users/fmg_regional_bank_modeling`.

Config file: `config/databricks_free.yaml` (merged when `FMG_DATABRICKS_TIER=free`).

### Cluster libraries

```text
%pip install -r /Workspace/.../client-specific-model/requirements-databricks.txt
```
