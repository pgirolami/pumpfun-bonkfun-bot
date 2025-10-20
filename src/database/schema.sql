-- SQLite schema for trading bot database
-- Database file naming: data/{bot_name}_{wallet_pubkey_short}_{mode}.db

-- Token information table (written once per token)
CREATE TABLE IF NOT EXISTS token_info (
    mint TEXT NOT NULL,
    platform TEXT NOT NULL,
    name TEXT NOT NULL,
    symbol TEXT NOT NULL,
    uri TEXT NOT NULL,
    bonding_curve TEXT,
    associated_bonding_curve TEXT,
    pool_state TEXT,
    base_vault TEXT,
    quote_vault TEXT,
    user TEXT,
    creator TEXT,
    creator_vault TEXT,
    additional_data TEXT,  -- JSON serialized dict
    PRIMARY KEY (mint, platform)
);

-- Positions table (updated when position changes)
CREATE TABLE IF NOT EXISTS positions (
    id TEXT PRIMARY KEY,  -- Hash of mint + platform + entry_ts
    mint TEXT NOT NULL,
    platform TEXT NOT NULL,
    entry_net_price_decimal REAL,  -- Net price from TradeResult.net_price_decimal()
    token_decimals INTEGER,  -- Value of 10 ** TOKEN_DECIMALS
    total_token_swapin_amount_raw INTEGER,  -- Total tokens bought
    total_token_swapout_amount_raw INTEGER,  -- Total tokens sold (accumulation)
    entry_ts INTEGER,  -- Unix epoch milliseconds
    exit_strategy TEXT,  -- Exit strategy from config
    highest_price REAL,  -- Already net decimal from on-chain calculation
    is_active INTEGER,  -- 0/1 boolean
    exit_reason TEXT,
    exit_net_price_decimal REAL,  -- Net price from TradeResult.net_price_decimal()
    exit_ts INTEGER,  -- Unix epoch milliseconds
    transaction_fee_raw INTEGER,  -- Accumulated fees
    platform_fee_raw INTEGER,  -- Accumulated fees
    realized_pnl_sol_decimal REAL,  -- From get_net_pnl()["realized_pnl_sol_decimal"]
    realized_net_pnl_sol_decimal REAL,  -- From get_net_pnl()["realized_net_pnl_sol_decimal"]
    buy_amount REAL,  -- Intended SOL amount to invest
    total_net_sol_swapout_amount_raw INTEGER,  -- Total SOL spent on buys
    total_net_sol_swapin_amount_raw INTEGER,  -- Total SOL received from sells (starts at 0)
    created_ts INTEGER NOT NULL,  -- Unix epoch milliseconds
    updated_ts INTEGER NOT NULL  -- Unix epoch milliseconds
);

-- Trades table (all trade attempts, success and failure)
CREATE TABLE IF NOT EXISTS trades (
    mint TEXT NOT NULL,
    timestamp INTEGER NOT NULL,  -- Unix epoch milliseconds from block time
    position_id TEXT,  -- Foreign key to positions.id
    success INTEGER NOT NULL,  -- 0/1 boolean
    platform TEXT NOT NULL,
    trade_type TEXT NOT NULL,  -- "buy" or "sell"
    tx_signature TEXT,
    error_message TEXT,
    token_swap_amount_raw INTEGER,
    net_sol_swap_amount_raw INTEGER,
    transaction_fee_raw INTEGER,
    platform_fee_raw INTEGER,
    price_decimal REAL,  -- calculated from trade result
    net_price_decimal REAL,  -- calculated net price
    run_id TEXT NOT NULL,  -- Bot run identifier (timestamp + git hash)
    PRIMARY KEY (mint, timestamp)
);

-- Create indices for frequently queried fields
CREATE INDEX IF NOT EXISTS idx_positions_mint ON positions(mint);
CREATE INDEX IF NOT EXISTS idx_positions_platform ON positions(platform);
CREATE INDEX IF NOT EXISTS idx_positions_is_active ON positions(is_active);
CREATE INDEX IF NOT EXISTS idx_positions_entry_ts ON positions(entry_ts);

CREATE INDEX IF NOT EXISTS idx_trades_position_id ON trades(position_id);
CREATE INDEX IF NOT EXISTS idx_trades_platform ON trades(platform);
CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp);
CREATE INDEX IF NOT EXISTS idx_trades_success ON trades(success);

-- Price history table (prices calculated during monitoring)
CREATE TABLE IF NOT EXISTS price_history (
    mint TEXT NOT NULL,
    platform TEXT NOT NULL,
    timestamp INTEGER NOT NULL,  -- Unix epoch milliseconds
    price_decimal REAL NOT NULL  -- Price in SOL (decimal)
);

-- Create index for efficient queries
CREATE INDEX IF NOT EXISTS idx_price_history_mint ON price_history(mint);
