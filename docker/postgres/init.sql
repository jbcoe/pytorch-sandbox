-- Check if database exists before creating
SELECT 'CREATE DATABASE mlflow'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'mlflow')\gexec

-- Connect to the mlflow database
\c mlflow

-- Create necessary extensions
CREATE EXTENSION IF NOT EXISTS pgcrypto;
