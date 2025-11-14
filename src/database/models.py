"""
Database model converters for trading bot data structures.

This module provides conversion between Python dataclasses and database rows.
"""

import json

from solders.pubkey import Pubkey

from core.pubkeys import TOKEN_DECIMALS
from interfaces.core import Platform, TokenInfo
from trading.base import TradeResult
from trading.position import Position,ExitReason


class TokenInfoConverter:
    """Converter for TokenInfo dataclass to/from database rows."""

    @staticmethod
    def to_row(token_info: TokenInfo) -> tuple:
        """Convert TokenInfo to database row tuple.

        Args:
            token_info: TokenInfo instance

        Returns:
            Tuple of values for database insertion
        """
        return (
            str(token_info.mint),
            token_info.platform.value,
            token_info.name,
            token_info.symbol,
            token_info.uri,
            str(token_info.bonding_curve) if token_info.bonding_curve else None,
            str(token_info.associated_bonding_curve)
            if token_info.associated_bonding_curve
            else None,
            str(token_info.pool_state) if token_info.pool_state else None,
            str(token_info.base_vault) if token_info.base_vault else None,
            str(token_info.quote_vault) if token_info.quote_vault else None,
            str(token_info.user) if token_info.user else None,
            str(token_info.creator) if token_info.creator else None,
            str(token_info.creator_vault) if token_info.creator_vault else None,
            json.dumps(token_info.additional_data)
            if token_info.additional_data
            else None,
            token_info.initial_buy_token_amount_decimal,
            token_info.initial_buy_sol_amount_decimal,
        )

    @staticmethod
    def from_row(row: tuple) -> TokenInfo:
        """Convert database row to TokenInfo instance.

        Args:
            row: Database row tuple

        Returns:
            TokenInfo instance
        """
        return TokenInfo(
            mint=Pubkey.from_string(row[0]),
            platform=Platform(row[1]),
            name=row[2],
            symbol=row[3],
            uri=row[4],
            bonding_curve=Pubkey.from_string(row[5]) if row[5] else None,
            associated_bonding_curve=Pubkey.from_string(row[6]) if row[6] else None,
            pool_state=Pubkey.from_string(row[7]) if row[7] else None,
            base_vault=Pubkey.from_string(row[8]) if row[8] else None,
            quote_vault=Pubkey.from_string(row[9]) if row[9] else None,
            user=Pubkey.from_string(row[10]) if row[10] else None,
            creator=Pubkey.from_string(row[11]) if row[11] else None,
            creator_vault=Pubkey.from_string(row[12]) if row[12] else None,
            creation_timestamp=None,
            additional_data=json.loads(row[13]) if row[13] else None,
            initial_buy_token_amount_decimal=row[14] if len(row) > 14 else None,
            initial_buy_sol_amount_decimal=row[15] if len(row) > 15 else None,
        )


class PositionConverter:
    """Converter for Position dataclass to/from database rows."""

    @staticmethod
    def to_row(position: Position) -> tuple:
        """Convert Position to database row tuple.

        Args:
            position: Position instance

        Returns:
            Tuple of values for database insertion
        """
        return (
            position.position_id,
            str(position.mint),
            position.platform.value,
            position.entry_net_price_decimal,
            10**TOKEN_DECIMALS,  # token_decimals constant
            position.total_token_swapin_amount_raw,
            position.total_token_swapout_amount_raw,
            position.entry_ts,
            position.highest_price,
            position.max_no_price_change_time,
            position.last_price_change_ts,
            1 if position.is_active else 0,  # Convert boolean to int
            position.exit_reason.value if position.exit_reason else None,
            position.exit_net_price_decimal,
            position.exit_ts,
            position.transaction_fee_raw or 0,
            position.platform_fee_raw or 0,
            position.tip_fee_raw or 0,
            position.rent_exemption_amount_raw or 0,
            position.unattributed_sol_amount_raw or 0,
            position.realized_pnl_sol_decimal,
            position.realized_net_pnl_sol_decimal,
            position.buy_amount,
            position.total_net_sol_swapout_amount_raw,
            position.total_net_sol_swapin_amount_raw,
            position.total_sol_swapout_amount_raw,
            position.total_sol_swapin_amount_raw,
            position.entry_ts,  # created_ts (same as entry_ts for new positions)
            position.entry_ts,  # updated_ts (will be updated on changes)
        )

    @staticmethod
    def from_row(row: tuple, min_gain_percentage: float | None = None, min_gain_time_window: int = 2) -> Position:
        """Convert database row to Position instance.

        Args:
            row: Database row tuple
            min_gain_percentage: Current min_gain_percentage from bot configuration
            min_gain_time_window: Current min_gain_time_window from bot configuration

        Returns:
            Position instance
        """

        return Position(
            position_id=row[0],
            mint=Pubkey.from_string(row[1]),
            platform=Platform(row[2]),
            entry_net_price_decimal=row[3],
            token_quantity_decimal=None if row[5] is None else row[5] / (10**TOKEN_DECIMALS),  # Calculate from raw amount
            total_token_swapin_amount_raw=row[5],
            total_token_swapout_amount_raw=row[6],
            entry_ts=row[7],
            highest_price=row[9],
            max_no_price_change_time=row[10],
            last_price_change_ts=row[11],
            is_active=bool(row[12]),
            exit_reason=ExitReason(row[13]) if row[13] else None,
            exit_net_price_decimal=row[14],
            exit_ts=row[15],
            transaction_fee_raw=row[16],
            platform_fee_raw=row[17],
            tip_fee_raw=row[18],
            rent_exemption_amount_raw=row[19],
            unattributed_sol_amount_raw=row[20],
            realized_pnl_sol_decimal=row[21],
            realized_net_pnl_sol_decimal=row[22],
            buy_amount=row[23],
            total_net_sol_swapout_amount_raw=row[24],
            total_net_sol_swapin_amount_raw=row[25],
            total_sol_swapout_amount_raw=row[26],
            total_sol_swapin_amount_raw=row[27],
            min_gain_percentage=min_gain_percentage,  # Set from current configuration
            min_gain_time_window=min_gain_time_window,  # Set from current configuration
            monitoring_start_ts=None,  # Will be set when monitoring starts
        )


class TradeConverter:
    """Converter for TradeResult dataclass to/from database rows."""

    @staticmethod
    def to_row(
        trade_result: TradeResult,
        mint: str,
        timestamp: int,
        position_id: str | None,
        trade_type: str,
        run_id: str,
    ) -> tuple:
        """Convert TradeResult to database row tuple.

        Args:
            trade_result: TradeResult instance
            mint: Token mint address
            timestamp: Unix epoch milliseconds
            position_id: Position ID (can be None)
            trade_type: "buy" or "sell"
            run_id: Bot run identifier

        Returns:
            Tuple of values for database insertion
        """
        return (
            mint,
            timestamp,
            position_id,
            1 if trade_result.success else 0,  # Convert boolean to int
            trade_result.platform.value,
            trade_type,
            trade_result.tx_signature,
            trade_result.error_message,
            trade_result.token_swap_amount_raw,
            trade_result.net_sol_swap_amount_raw,
            trade_result.transaction_fee_raw,
            trade_result.platform_fee_raw,
            trade_result.tip_fee_raw,
            trade_result.rent_exemption_amount_raw,
            trade_result.unattributed_sol_amount_raw,
            trade_result.sol_swap_amount_raw,
            trade_result.price_sol_decimal(),
            trade_result.net_price_sol_decimal(),
            trade_result.trade_duration_ms,
            trade_result.time_to_block_ms,
            run_id,
            trade_result.block_time,
        )

    @staticmethod
    def from_row(row: tuple) -> TradeResult:
        """Convert database row to TradeResult instance.

        Args:
            row: Database row tuple

        Returns:
            TradeResult instance
        """
        # Calculate raw SOL amount from stored net value

        return TradeResult(
            success=bool(row[3]),
            platform=Platform(row[4]),
            tx_signature=row[6],
            error_message=row[7],
            block_time=(row[19] or row[1]),
            token_swap_amount_raw=row[8],
            net_sol_swap_amount_raw=row[9],
            transaction_fee_raw=row[10],
            platform_fee_raw=row[11],
            tip_fee_raw=row[12],
            rent_exemption_amount_raw=row[13],
            unattributed_sol_amount_raw=row[14],
            sol_swap_amount_raw=row[15],
            trade_duration_ms=row[18],  # trade_duration_ms
            time_to_block_ms=row[19],
        )


class PumpPortalMessageConverter:
    """Converter for PumpPortal message data to/from database rows."""

    @staticmethod
    def to_row(
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
    ) -> tuple:
        """Convert PumpPortal message data to database row tuple.

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

        Returns:
            Tuple of values for database insertion
        """
        return (
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
