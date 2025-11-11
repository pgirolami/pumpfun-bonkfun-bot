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
    initial_buy_token_amount_decimal REAL,  -- PumpPortal initial buy token amount (decimal)
    initial_buy_sol_amount_decimal REAL,  -- PumpPortal initial buy SOL amount (decimal)
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
    max_no_price_change_time INTEGER,  -- Maximum seconds without price change
    last_price_change_ts REAL,  -- Timestamp of last price change
    is_active INTEGER,  -- 0/1 boolean
    exit_reason TEXT,
    exit_net_price_decimal REAL,  -- Net price from TradeResult.net_price_decimal()
    exit_ts INTEGER,  -- Unix epoch milliseconds
    transaction_fee_raw INTEGER,  -- Accumulated fees
    platform_fee_raw INTEGER,  -- Accumulated fees
    tip_fee_raw INTEGER,  -- Accumulated Helius tip fees
    rent_exemption_amount_raw INTEGER,
    unattributed_sol_amount_raw INTEGER,
    realized_pnl_sol_decimal REAL,  -- From get_net_pnl()["realized_pnl_sol_decimal"]
    realized_net_pnl_sol_decimal REAL,  -- From get_net_pnl()["realized_net_pnl_sol_decimal"]
    buy_amount REAL,  -- Intended SOL amount to invest
    total_net_sol_swapout_amount_raw INTEGER,  -- Total SOL spent on buys
    total_net_sol_swapin_amount_raw INTEGER,  -- Total SOL received from sells (starts at 0)
    total_sol_swapout_amount_raw INTEGER,  -- Total SOL spent on buys
    total_sol_swapin_amount_raw INTEGER,  -- Total SOL received from sells (starts at 0)
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
    tip_fee_raw INTEGER,
    rent_exemption_amount_raw INTEGER,
    unattributed_sol_amount_raw INTEGER,
    sol_swap_amount_raw INTEGER,
    price_decimal REAL,  -- calculated from trade result
    net_price_decimal REAL,  -- calculated net price
    trade_duration_ms INTEGER,  -- Trade execution duration in milliseconds
    time_to_block_ms INTEGER,  -- Time to blocktime in milliseconds
    run_id TEXT NOT NULL,  -- Bot run identifier (timestamp + git hash)
    block_time INTEGER,  -- Original on-chain block time in Unix epoch (nullable)
    PRIMARY KEY (mint, timestamp)
);

-- Create indices for frequently queried fields
CREATE INDEX IF NOT EXISTS idx_positions_mint ON positions(mint);
CREATE INDEX IF NOT EXISTS idx_positions_platform ON positions(platform);
CREATE INDEX IF NOT EXISTS idx_positions_is_active ON positions(is_active);
CREATE INDEX IF NOT EXISTS idx_positions_entry_ts ON positions(entry_ts);
CREATE INDEX IF NOT EXISTS idx_positions_exit_ts ON positions(exit_ts);

CREATE INDEX IF NOT EXISTS idx_trades_position_id ON trades(position_id);
CREATE INDEX IF NOT EXISTS idx_trades_platform ON trades(platform);
CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp);
CREATE INDEX IF NOT EXISTS idx_trades_success ON trades(success);
CREATE INDEX IF NOT EXISTS idx_trades_run_id ON trades(run_id);

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
    price_swap_decimal REAL,  -- price from swap: sol_amount_swapped / (token_amount_swapped / 10^TOKEN_DECIMALS) (nullable)
    pool TEXT,  -- pool name from PumpPortal message (nullable)
    trader_public_key TEXT  -- trader public key from PumpPortal message (nullable)
);

-- Create index for efficient queries
CREATE INDEX IF NOT EXISTS idx_pumpportal_messages_mint_platform ON pumpportal_messages(mint, platform);


-- Wallet balance history table (balance updates every minute)
CREATE TABLE IF NOT EXISTS wallet_balances (
    wallet_pubkey TEXT NOT NULL,
    timestamp INTEGER NOT NULL,  -- Unix epoch milliseconds
    balance_sol REAL NOT NULL,   -- Balance in SOL (decimal)
    balance_lamports INTEGER NOT NULL,  -- Balance in lamports
    run_id TEXT NOT NULL,  -- Bot run identifier (timestamp + git hash)
    PRIMARY KEY (wallet_pubkey, timestamp)
);

-- Create index for efficient queries
CREATE INDEX IF NOT EXISTS idx_wallet_balances_wallet ON wallet_balances(wallet_pubkey);
CREATE INDEX IF NOT EXISTS idx_wallet_balances_timestamp ON wallet_balances(timestamp);
CREATE INDEX IF NOT EXISTS idx_wallet_balances_run_id ON wallet_balances(run_id);
