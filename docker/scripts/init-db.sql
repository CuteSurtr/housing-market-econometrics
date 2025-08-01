-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- Create database indexes for better performance
CREATE INDEX IF NOT EXISTS idx_housing_data_date ON housing_data(observation_date);
CREATE INDEX IF NOT EXISTS idx_housing_data_region ON housing_data(region_id);
CREATE INDEX IF NOT EXISTS idx_model_runs_created ON model_runs(created_at);
CREATE INDEX IF NOT EXISTS idx_forecasts_created ON forecasts(created_at); 