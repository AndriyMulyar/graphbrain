CREATE USER graphbrain_api WITH PASSWORD 'BXh&R76Z7ZJvxg:+L#WxVY#ykK[f3C';
CREATE DATABASE graphbrain;

GRANT CONNECT ON DATABASE graphbrain TO graphbrain_api;
GRANT USAGE ON SCHEMA public TO graphbrain_api;

-- Tables
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO graphbrain_api;

ALTER DEFAULT PRIVILEGES IN SCHEMA public
  GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO graphbrain_api;

-- Sequence
GRANT USAGE, SELECT, UPDATE ON ALL SEQUENCES IN SCHEMA public TO graphbrain_api;
ALTER DEFAULT PRIVILEGES IN SCHEMA public
  GRANT USAGE, SELECT, UPDATE ON SEQUENCES TO graphbrain_api;