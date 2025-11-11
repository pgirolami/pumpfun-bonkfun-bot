-- Database migration script to add pool and trader_public_key columns to pumpportal_messages table.
-- This migration:
--   1. Adds pool column to store the pool name directly from PumpPortal messages
--   2. Adds trader_public_key column to store the trader's public key
--
-- Usage: sqlite3 <database_path> < migrate_add_pool_and_trader_public_key.sql

-- Add pool column
ALTER TABLE pumpportal_messages ADD COLUMN pool TEXT;

-- Add trader_public_key column
ALTER TABLE pumpportal_messages ADD COLUMN trader_public_key TEXT;

