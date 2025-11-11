-- Database migration script to replace price_history table with pumpportal_messages table.
-- This migration:
--   1. Drops the price_history table
--   2. Creates the new pumpportal_messages table with diagnostic information
--   3. Creates an index on (mint, platform)
--
-- Usage: sqlite3 <database_path> < migrate_replace_price_history_with_pumpportal_messages.sql

-- Drop the old price_history table
DROP TABLE IF EXISTS price_history;

-- Drop the old index if it exists
DROP INDEX IF EXISTS idx_price_history_mint;

-- Create the new pumpportal_messages table
CREATE TABLE IF NOT EXISTS pumpportal_messages (
    mint TEXT NOT NULL,
    platform TEXT NOT NULL,
    timestamp INTEGER NOT NULL,  -- Unix epoch milliseconds
    message_type TEXT NOT NULL,  -- "buy", "sell", or "create"
    virtual_sol_reserves REAL NOT NULL,  -- decimal SOL
    virtual_token_reserves REAL NOT NULL,  -- decimal tokens
    sol_amount_swapped REAL,  -- decimal SOL from trade (nullable for create messages)
    token_amount_swapped REAL,  -- decimal tokens from trade (nullable for create messages)
    price_reserves_decimal REAL NOT NULL,  -- price from reserves: virtual_sol_reserves / (virtual_token_reserves / 10^TOKEN_DECIMALS)
    price_swap_decimal REAL  -- price from swap: sol_amount_swapped / (token_amount_swapped / 10^TOKEN_DECIMALS) (nullable)
);

-- Create index for efficient queries
CREATE INDEX IF NOT EXISTS idx_pumpportal_messages_mint_platform ON pumpportal_messages(mint, platform);

