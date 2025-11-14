"""
Database manager for trading bot persistence.

This module provides the main DatabaseManager class for SQLite operations.
"""

import sqlite3
from contextlib import asynccontextmanager
from pathlib import Path
from time import time
from typing import Any

from interfaces.core import TokenInfo
from trading.base import TradeResult
from trading.position import Position
from utils.logger import get_logger

from .models import (
    PositionConverter,
    TokenInfoConverter,
    TradeConverter,
)

logger = get_logger(__name__)


class DatabaseManager:
    """Manages SQLite database operations for trading bot persistence."""

    def __init__(self, db_path: str):
        """Initialize database manager.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._ensure_data_directory()
        self._initialize_schema()

    def _ensure_data_directory(self) -> None:
        """Ensure data directory exists."""
        data_dir = Path(self.db_path).parent
        data_dir.mkdir(parents=True, exist_ok=True)

    def _initialize_schema(self) -> None:
        """Initialize database schema if not exists."""
        schema_path = Path(__file__).parent / "schema.sql"
        with open(schema_path) as f:
            schema_sql = f.read()

        with sqlite3.connect(self.db_path) as conn:
            conn.executescript(schema_sql)
            # Lightweight migration: ensure 'block_time' column exists on trades table
            try:
                cursor = conn.execute("PRAGMA table_info(trades)")
                columns = {row[1] for row in cursor.fetchall()}
                if "block_time" not in columns:
                    conn.execute("ALTER TABLE trades ADD COLUMN block_time INTEGER")
            except Exception:
                # Do not fail initialization if pragma/alter fails; log and continue
                logger.exception("Failed to migrate 'trades' table to add block_time column")
            
            # Migration: ensure 'buy_order' and 'sell_order' columns exist on positions table
            try:
                cursor = conn.execute("PRAGMA table_info(positions)")
                columns = {row[1] for row in cursor.fetchall()}
                if "buy_order" not in columns:
                    conn.execute("ALTER TABLE positions ADD COLUMN buy_order TEXT")
                if "sell_order" not in columns:
                    conn.execute("ALTER TABLE positions ADD COLUMN sell_order TEXT")
            except Exception:
                # Do not fail initialization if migration fails; log and continue
                logger.exception("Failed to migrate 'positions' table to add buy_order and sell_order columns")
            
            conn.commit()

        logger.info(f"Database initialized at {self.db_path}")

    @asynccontextmanager
    async def get_connection(self):
        """Get database connection with proper cleanup.

        Yields:
            sqlite3.Connection: Database connection
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Enable dict-like access
        try:
            yield conn
        finally:
            conn.close()

    # Token Info Operations
    async def insert_token_info(self, token_info: TokenInfo) -> None:
        """Insert or ignore token info.

        Args:
            token_info: TokenInfo to insert
        """
        row = TokenInfoConverter.to_row(token_info)

        async with self.get_connection() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO token_info 
                (mint, platform, name, symbol, uri, bonding_curve, associated_bonding_curve,
                 pool_state, base_vault, quote_vault, user, creator, creator_vault, additional_data,
                 initial_buy_token_amount_decimal, initial_buy_sol_amount_decimal)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                row,
            )
            conn.commit()

        logger.debug(
            f"Token info inserted/ignored: {token_info.symbol} ({token_info.mint})"
        )

    async def get_token_info(self, mint: str, platform: str) -> TokenInfo | None:
        """Get token info by mint and platform.

        Args:
            mint: Token mint address
            platform: Platform name

        Returns:
            TokenInfo if found, None otherwise
        """
        async with self.get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM token_info WHERE mint = ? AND platform = ?",
                (mint, platform),
            )
            row = cursor.fetchone()

            if row:
                return TokenInfoConverter.from_row(tuple(row))
            return None

    # Position Operations
    async def insert_position(self, position: Position) -> None:
        """Insert position into database.

        Args:
            position: Position to insert
        """
        row = PositionConverter.to_row(position)

        async with self.get_connection() as conn:
            conn.execute(
                """
                INSERT INTO positions 
                (id, mint, platform, entry_net_price_decimal, token_decimals, total_token_swapin_amount_raw,
                 total_token_swapout_amount_raw, entry_ts, highest_price, max_no_price_change_time, last_price_change_ts, is_active,
                 exit_reason, exit_net_price_decimal, exit_ts, transaction_fee_raw, platform_fee_raw, tip_fee_raw,
                 rent_exemption_amount_raw, unattributed_sol_amount_raw,
                 realized_pnl_sol_decimal, realized_net_pnl_sol_decimal, buy_amount, total_net_sol_swapout_amount_raw, total_net_sol_swapin_amount_raw,
                 total_sol_swapout_amount_raw, total_sol_swapin_amount_raw, buy_order, sell_order, created_ts, updated_ts)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                row,
            )
            conn.commit()

        logger.debug(f"Position inserted: {position.position_id}")

    async def update_position(self, position: Position) -> None:
        """Update existing position.

        Args:
            position: Position with updated data
        """
        position_id = position.position_id

        async with self.get_connection() as conn:
            # Get serialized orders from PositionConverter
            from .models import PositionConverter
            
            conn.execute(
                """
                UPDATE positions SET
                    total_token_swapin_amount_raw = ?,
                    total_token_swapout_amount_raw = ?,
                    highest_price = ?,
                    max_no_price_change_time = ?,
                    last_price_change_ts = ?,
                    is_active = ?,
                    exit_reason = ?,
                    entry_net_price_decimal = ?,
                    exit_net_price_decimal = ?,
                    entry_ts = ?,
                    exit_ts = ?,
                    transaction_fee_raw = ?,
                    platform_fee_raw = ?,
                    tip_fee_raw = ?,
                    rent_exemption_amount_raw = ?,
                    unattributed_sol_amount_raw = ?,
                    realized_pnl_sol_decimal = ?,
                    realized_net_pnl_sol_decimal = ?,
                    total_net_sol_swapin_amount_raw = ?,
                    total_net_sol_swapout_amount_raw = ?,
                    total_sol_swapin_amount_raw = ?,
                    total_sol_swapout_amount_raw = ?,
                    buy_order = ?,
                    sell_order = ?,
                    updated_ts = ?
                WHERE id = ?
            """,
                (
                    position.total_token_swapin_amount_raw,
                    position.total_token_swapout_amount_raw,
                    position.highest_price,
                    position.max_no_price_change_time,
                    position.last_price_change_ts,
                    1 if position.is_active else 0,
                    position.exit_reason.value if position.exit_reason else None,
                    position.entry_net_price_decimal,
                    position.exit_net_price_decimal,
                    position.entry_ts,
                    position.exit_ts,
                    position.transaction_fee_raw or 0,
                    position.platform_fee_raw or 0,
                    position.tip_fee_raw or 0,
                    position.rent_exemption_amount_raw or 0,
                    position.unattributed_sol_amount_raw or 0,
                    position.realized_pnl_sol_decimal,
                    position.realized_net_pnl_sol_decimal,
                    position.total_net_sol_swapin_amount_raw or 0,  # Direct value, not accumulation
                    position.total_net_sol_swapout_amount_raw or 0,  # Direct value, not accumulation
                    position.total_sol_swapin_amount_raw or 0,  # Direct value, not accumulation
                    position.total_sol_swapout_amount_raw or 0,  # Direct value, not accumulation
                    PositionConverter._serialize_order(position.buy_order),
                    PositionConverter._serialize_order(position.sell_order),
                    int(time()*1000),
                    position_id,
                ),
            )
            conn.commit()

        logger.debug(f"Position updated: {position_id}")

    async def get_position(self, position_id: str, min_gain_percentage: float | None = None, min_gain_time_window: int = 2) -> Position | None:
        """Get position by ID.

        Args:
            position_id: Position ID
            min_gain_percentage: Current min_gain_percentage from bot configuration
            min_gain_time_window: Current min_gain_time_window from bot configuration

        Returns:
            Position if found, None otherwise
        """
        async with self.get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM positions WHERE id = ?", (position_id,)
            )
            row = cursor.fetchone()

            if row:
                return PositionConverter.from_row(tuple(row), min_gain_percentage, min_gain_time_window)
            return None

    async def get_active_positions(self, min_gain_percentage: float | None = None, min_gain_time_window: int = 2) -> list[Position]:
        """Get all active positions.

        Args:
            min_gain_percentage: Current min_gain_percentage from bot configuration
            min_gain_time_window: Current min_gain_time_window from bot configuration

        Returns:
            List of active positions
        """
        async with self.get_connection() as conn:
            cursor = conn.execute("SELECT * FROM positions WHERE is_active = 1")
            rows = cursor.fetchall()

            return [PositionConverter.from_row(tuple(row), min_gain_percentage, min_gain_time_window) for row in rows]

    async def get_recent_closed_positions(
        self, lookback_minutes: int, run_id: str
    ) -> list[Position]:
        """Get closed positions within time window for specific run_id.

        Args:
            lookback_minutes: Time window in minutes to look back
            run_id: Bot run identifier (mandatory)

        Returns:
            List of closed positions with entry/exit times and PnL data
        """
        from time import time as current_time

        # Calculate cutoff timestamp (milliseconds)
        current_time_ms = int(current_time() * 1000)
        cutoff_time_ms = current_time_ms - (lookback_minutes * 60 * 1000)

        async with self.get_connection() as conn:
            # Join positions with trades to filter by run_id
            # Get distinct positions that have trades with the specified run_id
            cursor = conn.execute(
                """
                SELECT DISTINCT p.*
                FROM positions p
                INNER JOIN trades t ON p.id = t.position_id
                WHERE p.is_active = 0
                  AND t.run_id = ?
                  AND p.exit_ts >= ?
                ORDER BY p.exit_ts DESC
                """,
                (run_id, cutoff_time_ms),
            )
            rows = cursor.fetchall()

            return [
                PositionConverter.from_row(tuple(row)) for row in rows
            ]

    async def get_recent_closed_positions_pnl_data(
        self, lookback_minutes: int, run_id: str
    ) -> list[Any]:
        """Get closed positions PnL data within time window for specific run_id.
        
        Returns only the fields needed for market quality calculations, avoiding
        deserialization of buy_order/sell_order which may be NULL.

        Args:
            lookback_minutes: Time window in minutes to look back
            run_id: Bot run identifier (mandatory)

        Returns:
            List of PositionPnLData objects with PnL data
        """
        from time import time as current_time

        from .models import PositionPnLData

        # Calculate cutoff timestamp (milliseconds)
        current_time_ms = int(current_time() * 1000)
        cutoff_time_ms = current_time_ms - (lookback_minutes * 60 * 1000)

        async with self.get_connection() as conn:
            # Join positions with trades to filter by run_id
            # Query only the fields needed for PnL calculations
            cursor = conn.execute(
                """
                SELECT DISTINCT 
                    p.id,
                    p.mint,
                    p.realized_pnl_sol_decimal,
                    p.total_net_sol_swapout_amount_raw,
                    p.transaction_fee_raw,
                    p.platform_fee_raw,
                    p.tip_fee_raw
                FROM positions p
                INNER JOIN trades t ON p.id = t.position_id
                WHERE p.is_active = 0
                  AND t.run_id = ?
                  AND p.exit_ts >= ?
                ORDER BY p.exit_ts DESC
                """,
                (run_id, cutoff_time_ms),
            )
            rows = cursor.fetchall()

            return [
                PositionPnLData(
                    position_id=row[0],
                    mint=row[1],
                    realized_pnl_sol_decimal=row[2],
                    total_net_sol_swapout_amount_raw=row[3],
                    transaction_fee_raw=row[4],
                    platform_fee_raw=row[5],
                    tip_fee_raw=row[6],
                )
                for row in rows
            ]

    # Trade Operations
    async def insert_trade(
        self,
        trade_result: TradeResult,
        mint: str,
        position_id: str | None,
        trade_type: str,
        run_id: str,
    ) -> None:
        """Insert trade record.

        Args:
            trade_result: Trade execution result
            mint: Token mint address
            position_id: Position ID (can be None)
            trade_type: "buy" or "sell"
            run_id: Bot run identifier
        """
        #TODO fix
        timestamp:int = int(time()*1000) 
        
        row = TradeConverter.to_row(
            trade_result, mint, timestamp, position_id, trade_type, run_id
        )

        async with self.get_connection() as conn:
            conn.execute(
                """
                INSERT INTO trades 
                (mint, timestamp, position_id, success, platform, trade_type,
                 tx_signature, error_message, token_swap_amount_raw, net_sol_swap_amount_raw,
                 transaction_fee_raw, platform_fee_raw, tip_fee_raw, rent_exemption_amount_raw, unattributed_sol_amount_raw, sol_swap_amount_raw, price_decimal, net_price_decimal, trade_duration_ms, time_to_block_ms, run_id, block_time)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                row,
            )
            conn.commit()

        logger.debug(f"Trade inserted: {trade_type} for {mint} at {timestamp}")

    async def get_trades_by_position(self, position_id: str) -> list[TradeResult]:
        """Get all trades for a position.

        Args:
            position_id: Position ID

        Returns:
            List of trades for the position
        """
        async with self.get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM trades WHERE position_id = ? ORDER BY timestamp",
                (position_id,),
            )
            rows = cursor.fetchall()

            return [TradeConverter.from_row(tuple(row)) for row in rows]

    async def get_trades_by_mint(self, mint: str) -> list[TradeResult]:
        """Get all trades for a token mint.

        Args:
            mint: Token mint address

        Returns:
            List of trades for the mint
        """
        async with self.get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM trades WHERE mint = ? ORDER BY timestamp", (mint,)
            )
            rows = cursor.fetchall()

            return [TradeConverter.from_row(tuple(row)) for row in rows]

    # Analytics Operations
    async def get_position_pnl_summary(self) -> dict[str, Any]:
        """Get PnL summary for all positions.

        Returns:
            Dictionary with PnL statistics
        """
        async with self.get_connection() as conn:
            cursor = conn.execute("""
                SELECT 
                    COUNT(*) as total_positions,
                    SUM(CASE WHEN is_active = 0 THEN 1 ELSE 0 END) as closed_positions,
                    SUM(CASE WHEN is_active = 1 THEN 1 ELSE 0 END) as active_positions,
                    SUM(CASE WHEN exit_price>entry_price THEN 1 ELSE 0 END) as increase,
                    SUM(CASE WHEN realized_pnl IS NOT NULL THEN realized_pnl ELSE 0 END) as total_realized_pnl,
                    AVG(CASE WHEN realized_pnl IS NOT NULL THEN realized_pnl ELSE NULL END) as avg_realized_pnl,
                    MAX(CASE WHEN realized_pnl IS NOT NULL THEN realized_pnl ELSE NULL END) as max_realized_pnl,
                    SUM(transaction_fee_raw + platform_fee_raw) as total_fees_raw
                FROM positions
            """)
            row = cursor.fetchone()

            return {
                "total_positions": row[0],
                "closed_positions": row[1],
                "active_positions": row[2],
                "total_realized_pnl": row[3] or 0.0,
                "avg_realized_pnl": row[4] or 0.0,
                "total_fees_raw": row[5] or 0,
            }

    async def get_trade_summary(self) -> dict[str, Any]:
        """Get trade summary statistics.

        Returns:
            Dictionary with trade statistics
        """
        async with self.get_connection() as conn:
            cursor = conn.execute("""
                SELECT 
                    COUNT(*) as total_trades,
                    SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful_trades,
                    SUM(CASE WHEN success = 0 THEN 1 ELSE 0 END) as failed_trades,
                    SUM(CASE WHEN trade_type = 'buy' THEN 1 ELSE 0 END) as buy_trades,
                    SUM(CASE WHEN trade_type = 'sell' THEN 1 ELSE 0 END) as sell_trades
                FROM trades
            """)
            row = cursor.fetchone()

            return {
                "total_trades": row[0],
                "successful_trades": row[1],
                "failed_trades": row[2],
                "buy_trades": row[3],
                "sell_trades": row[4],
            }

    # PumpPortal Messages Operations
    async def insert_pumpportal_message(
        self,
        mint: str,
        platform: str,
        timestamp: int,
        message_type: str,
        virtual_sol_reserves: float,
        virtual_token_reserves: float,
        sol_amount_swapped: float | None,
        token_amount_swapped: float | None,
        price_reserves_decimal: float,
        price_swap_decimal: float | None,
        pool: str | None = None,
        trader_public_key: str | None = None,
        signature: str | None = None,
    ) -> None:
        """Insert PumpPortal message record.

        Args:
            mint: Token mint address
            platform: Platform name
            timestamp: Unix epoch milliseconds
            message_type: Message type ("buy", "sell", or "create")
            virtual_sol_reserves: Virtual SOL reserves (decimal)
            virtual_token_reserves: Virtual token reserves (decimal)
            sol_amount_swapped: SOL amount swapped in trade (decimal, nullable)
            token_amount_swapped: Token amount swapped in trade (decimal, nullable)
            price_reserves_decimal: Price calculated from reserves (SOL per token)
            price_swap_decimal: Price calculated from swap amounts (SOL per token, nullable)
            pool: Pool name from PumpPortal message (nullable)
            trader_public_key: Trader's public key from PumpPortal message (nullable)
            signature: Transaction signature from PumpPortal message (nullable)
        """
        from database.models import PumpPortalMessageConverter

        row = PumpPortalMessageConverter.to_row(
            mint,
            platform,
            timestamp,
            message_type,
            virtual_sol_reserves,
            virtual_token_reserves,
            sol_amount_swapped,
            token_amount_swapped,
            price_reserves_decimal,
            price_swap_decimal,
            pool,
            trader_public_key,
            signature,
        )

        async with self.get_connection() as conn:
            conn.execute(
                """
                INSERT INTO pumpportal_messages 
                (mint, platform, timestamp, message_type, virtual_sol_reserves, 
                 virtual_token_reserves, sol_amount_swapped, token_amount_swapped,
                 price_reserves_decimal, price_swap_decimal, pool, trader_public_key, signature)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                row,
            )
            conn.commit()

        logger.debug(
            f"PumpPortal message inserted: {mint} {message_type} at {price_reserves_decimal} SOL"
        )

    async def insert_wallet_balance(
        self,
        wallet_pubkey: str,
        balance_sol: float,
        balance_lamports: int,
        run_id: str,
    ) -> None:
        """Insert wallet balance record.

        Args:
            wallet_pubkey: Wallet public key
            balance_sol: Balance in SOL (decimal)
            balance_lamports: Balance in lamports
            run_id: Bot run identifier
        """
        timestamp = int(time() * 1000)

        async with self.get_connection() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO wallet_balances 
                (wallet_pubkey, timestamp, balance_sol, balance_lamports, run_id)
                VALUES (?, ?, ?, ?, ?)
                """,
                (wallet_pubkey, timestamp, balance_sol, balance_lamports, run_id),
            )
            conn.commit()

        logger.debug(f"Wallet balance inserted: {wallet_pubkey} = {balance_sol:.6f} SOL (run_id: {run_id})")
