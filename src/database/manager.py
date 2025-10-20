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
    PriceHistoryConverter,
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
                 pool_state, base_vault, quote_vault, user, creator, creator_vault, additional_data)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
    async def insert_position(self, position: Position) -> str:
        """Insert position and return position ID.

        Args:
            position: Position to insert

        Returns:
            Position ID
        """
        row = PositionConverter.to_row(position)
        position_id = row[0]  # First element is the ID

        async with self.get_connection() as conn:
            conn.execute(
                """
                INSERT INTO positions 
                (id, mint, platform, entry_net_price_decimal, token_decimals, total_token_swapin_amount_raw,
                 total_token_swapout_amount_raw, entry_ts, exit_strategy, highest_price, is_active,
                 exit_reason, exit_net_price_decimal, exit_ts, transaction_fee_raw, platform_fee_raw,
                 realized_pnl_sol_decimal, realized_net_pnl_sol_decimal, buy_amount, total_net_sol_swapout_amount_raw, total_net_sol_swapin_amount_raw,
                 created_ts, updated_ts)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                row,
            )
            conn.commit()

        logger.debug(f"Position inserted: {position_id}")
        return position_id

    async def update_position(self, position: Position) -> None:
        """Update existing position.

        Args:
            position: Position with updated data
        """
        position_id = PositionConverter.generate_position_id(
            position.mint, position.platform, position.entry_ts
        )

        async with self.get_connection() as conn:
            conn.execute(
                """
                UPDATE positions SET
                    total_token_swapout_amount_raw = ?,
                    highest_price = ?,
                    is_active = ?,
                    exit_reason = ?,
                    exit_net_price_decimal = ?,
                    exit_ts = ?,
                    transaction_fee_raw = ?,
                    platform_fee_raw = ?,
                    realized_pnl_sol_decimal = ?,
                    realized_net_pnl_sol_decimal = ?,
                    total_net_sol_swapin_amount_raw = ?,
                    updated_ts = ?
                WHERE id = ?
            """,
                (
                    position.total_token_swapout_amount_raw,
                    position.highest_price,
                    1 if position.is_active else 0,
                    position.exit_reason.value if position.exit_reason else None,
                    position.exit_net_price_decimal,
                    position.exit_ts,
                    position.transaction_fee_raw or 0,
                    position.platform_fee_raw or 0,
                    position.realized_pnl_sol_decimal,
                    position.realized_net_pnl_sol_decimal,
                    position.total_net_sol_swapin_amount_raw
                    or 0,  # Direct value, not accumulation
                    int(__import__("time").time() * 1000),  # Current timestamp
                    position_id,
                ),
            )
            conn.commit()

        logger.debug(f"Position updated: {position_id}")

    async def get_position(self, position_id: str) -> Position | None:
        """Get position by ID.

        Args:
            position_id: Position ID

        Returns:
            Position if found, None otherwise
        """
        async with self.get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM positions WHERE id = ?", (position_id,)
            )
            row = cursor.fetchone()

            if row:
                return PositionConverter.from_row(tuple(row))
            return None

    async def get_active_positions(self) -> list[Position]:
        """Get all active positions.

        Returns:
            List of active positions
        """
        async with self.get_connection() as conn:
            cursor = conn.execute("SELECT * FROM positions WHERE is_active = 1")
            rows = cursor.fetchall()

            return [PositionConverter.from_row(tuple(row)) for row in rows]

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
        # Use trade_result.block_time or current time as fallback
        timestamp = trade_result.block_time or int(time() * 1000)

        row = TradeConverter.to_row(
            trade_result, mint, timestamp, position_id, trade_type, run_id
        )

        async with self.get_connection() as conn:
            conn.execute(
                """
                INSERT INTO trades 
                (mint, timestamp, position_id, success, platform, trade_type,
                 tx_signature, error_message, token_swap_amount_raw, net_sol_swap_amount_raw,
                 transaction_fee_raw, platform_fee_raw, price_decimal, net_price_decimal, run_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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

    # Price History Operations
    async def insert_price_history(
        self,
        mint: str,
        platform: str,
        price_decimal: float,
    ) -> None:
        """Insert price history record.

        Args:
            mint: Token mint address
            platform: Platform name
            price_decimal: Price in SOL (decimal)
        """
        timestamp = int(time() * 1000)
        row = PriceHistoryConverter.to_row(mint, platform, timestamp, price_decimal)

        async with self.get_connection() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO price_history 
                (mint, platform, timestamp, price_decimal)
                VALUES (?, ?, ?, ?)
            """,
                row,
            )
            conn.commit()

        logger.debug(f"Price history inserted: {mint} at {price_decimal:.8f} SOL")
