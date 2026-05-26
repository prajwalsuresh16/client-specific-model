-- Databricks SQL Warehouse: Unity Catalog bootstrap for FMG Regional Bank
-- Run as Job task 00_setup_catalog (SQL warehouse) or from notebook %sql

CREATE CATALOG IF NOT EXISTS ${catalog};
CREATE SCHEMA IF NOT EXISTS ${catalog}.${schema};
CREATE VOLUME IF NOT EXISTS ${catalog}.${schema}.regional_bank_bronze_vol
  COMMENT 'Raw bronze parquet landing / file exports';

USE CATALOG ${catalog};
USE SCHEMA ${schema};
