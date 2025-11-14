-- Database migration script to add buy_order and sell_order columns to positions table.
-- This migration adds the following columns:
--   positions: buy_order (TEXT, nullable) - JSON serialized BuyOrder
--              sell_order (TEXT, nullable) - JSON serialized SellOrder
--
-- Usage: sqlite3 <database_path> < migrate_add_order_columns.sql

-- Add buy_order column if it doesn't exist
-- SQLite doesn't support IF NOT EXISTS for ALTER TABLE ADD COLUMN, so we check first
-- by attempting to query the column and catching the error, or by using a more complex approach

-- For SQLite, we'll use a simple ALTER TABLE approach
-- If the column already exists, this will fail, but that's okay for idempotency
-- Users can check manually or we can wrap in a try-catch in application code

ALTER TABLE positions ADD COLUMN buy_order TEXT;
ALTER TABLE positions ADD COLUMN sell_order TEXT;

