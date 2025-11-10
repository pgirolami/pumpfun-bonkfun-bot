-- Database migration script to add accounting columns to positions and trades tables.
-- This migration adds the following columns:
--   positions: rent_exemption_amount_raw, unattributed_sol_amount_raw,
--              total_sol_swapout_amount_raw, total_sol_swapin_amount_raw
--   trades: rent_exemption_amount_raw, unattributed_sol_amount_raw, sol_swap_amount_raw
--
-- The migration maintains the exact column order as specified in the new schema.
--
-- Usage: sqlite3 <database_path> < migrate_add_accounting_columns.sql

-- ============================================================================
-- MIGRATE POSITIONS TABLE
-- ============================================================================

-- Check if migration is needed (positions table)
-- If rent_exemption_amount_raw doesn't exist, we need to migrate
-- Note: This check is done by attempting to create the new table structure

-- Create new positions table with correct column order
CREATE TABLE IF NOT EXISTS positions_new (
    id TEXT PRIMARY KEY,
    mint TEXT NOT NULL,
    platform TEXT NOT NULL,
    entry_net_price_decimal REAL,
    token_decimals INTEGER,
    total_token_swapin_amount_raw INTEGER,
    total_token_swapout_amount_raw INTEGER,
    entry_ts INTEGER,
    exit_strategy TEXT,
    highest_price REAL,
    max_no_price_change_time INTEGER,
    last_price_change_ts REAL,
    is_active INTEGER,
    exit_reason TEXT,
    exit_net_price_decimal REAL,
    exit_ts INTEGER,
    transaction_fee_raw INTEGER,
    platform_fee_raw INTEGER,
    tip_fee_raw INTEGER,
    rent_exemption_amount_raw INTEGER,
    unattributed_sol_amount_raw INTEGER,
    realized_pnl_sol_decimal REAL,
    realized_net_pnl_sol_decimal REAL,
    buy_amount REAL,
    total_net_sol_swapout_amount_raw INTEGER,
    total_net_sol_swapin_amount_raw INTEGER,
    total_sol_swapout_amount_raw INTEGER,
    total_sol_swapin_amount_raw INTEGER,
    created_ts INTEGER NOT NULL,
    updated_ts INTEGER NOT NULL
);

-- Copy data from old table, inserting NULL for new columns
INSERT INTO positions_new (
    id, mint, platform, entry_net_price_decimal, token_decimals,
    total_token_swapin_amount_raw, total_token_swapout_amount_raw,
    entry_ts, exit_strategy, highest_price, max_no_price_change_time,
    last_price_change_ts, is_active, exit_reason, exit_net_price_decimal,
    exit_ts, transaction_fee_raw, platform_fee_raw, tip_fee_raw,
    rent_exemption_amount_raw, unattributed_sol_amount_raw,
    realized_pnl_sol_decimal, realized_net_pnl_sol_decimal, buy_amount,
    total_net_sol_swapout_amount_raw, total_net_sol_swapin_amount_raw,
    total_sol_swapout_amount_raw, total_sol_swapin_amount_raw,
    created_ts, updated_ts
)
SELECT
    id, mint, platform, entry_net_price_decimal, token_decimals,
    total_token_swapin_amount_raw, total_token_swapout_amount_raw,
    entry_ts, exit_strategy, highest_price, max_no_price_change_time,
    last_price_change_ts, is_active, exit_reason, exit_net_price_decimal,
    exit_ts, transaction_fee_raw, platform_fee_raw, tip_fee_raw,
    NULL AS rent_exemption_amount_raw,
    NULL AS unattributed_sol_amount_raw,
    realized_pnl_sol_decimal, realized_net_pnl_sol_decimal, buy_amount,
    total_net_sol_swapout_amount_raw, total_net_sol_swapin_amount_raw,
    NULL AS total_sol_swapout_amount_raw,
    NULL AS total_sol_swapin_amount_raw,
    created_ts, updated_ts
FROM positions;

-- Drop old table and rename new one
DROP TABLE positions;
ALTER TABLE positions_new RENAME TO positions;

-- Recreate indices
CREATE INDEX IF NOT EXISTS idx_positions_mint ON positions(mint);
CREATE INDEX IF NOT EXISTS idx_positions_platform ON positions(platform);
CREATE INDEX IF NOT EXISTS idx_positions_is_active ON positions(is_active);
CREATE INDEX IF NOT EXISTS idx_positions_entry_ts ON positions(entry_ts);
CREATE INDEX IF NOT EXISTS idx_positions_exit_ts ON positions(exit_ts);

-- ============================================================================
-- MIGRATE TRADES TABLE
-- ============================================================================

-- Create new trades table with correct column order
CREATE TABLE IF NOT EXISTS trades_new (
    mint TEXT NOT NULL,
    timestamp INTEGER NOT NULL,
    position_id TEXT,
    success INTEGER NOT NULL,
    platform TEXT NOT NULL,
    trade_type TEXT NOT NULL,
    tx_signature TEXT,
    error_message TEXT,
    token_swap_amount_raw INTEGER,
    net_sol_swap_amount_raw INTEGER,
    transaction_fee_raw INTEGER,
    platform_fee_raw INTEGER,
    tip_fee_raw INTEGER,
    rent_exemption_amount_raw INTEGER,
    unattributed_sol_amount_raw INTEGER,
    sol_swap_amount_raw INTEGER,
    price_decimal REAL,
    net_price_decimal REAL,
    trade_duration_ms INTEGER,
    time_to_block_ms INTEGER,
    run_id TEXT NOT NULL,
    block_time INTEGER,
    PRIMARY KEY (mint, timestamp)
);

-- Copy data from old table, inserting NULL for new columns
INSERT INTO trades_new (
    mint, timestamp, position_id, success, platform, trade_type,
    tx_signature, error_message, token_swap_amount_raw,
    net_sol_swap_amount_raw, transaction_fee_raw, platform_fee_raw,
    tip_fee_raw, rent_exemption_amount_raw, unattributed_sol_amount_raw,
    sol_swap_amount_raw, price_decimal, net_price_decimal,
    trade_duration_ms, time_to_block_ms, run_id, block_time
)
SELECT
    mint, timestamp, position_id, success, platform, trade_type,
    tx_signature, error_message, token_swap_amount_raw,
    net_sol_swap_amount_raw, transaction_fee_raw, platform_fee_raw,
    tip_fee_raw,
    NULL AS rent_exemption_amount_raw,
    NULL AS unattributed_sol_amount_raw,
    NULL AS sol_swap_amount_raw,
    price_decimal, net_price_decimal, trade_duration_ms, time_to_block_ms,
    run_id, block_time
FROM trades;

-- Drop old table and rename new one
DROP TABLE trades;
ALTER TABLE trades_new RENAME TO trades;

-- Recreate indices
CREATE INDEX IF NOT EXISTS idx_trades_position_id ON trades(position_id);
CREATE INDEX IF NOT EXISTS idx_trades_platform ON trades(platform);
CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp);
CREATE INDEX IF NOT EXISTS idx_trades_success ON trades(success);
CREATE INDEX IF NOT EXISTS idx_trades_run_id ON trades(run_id);

