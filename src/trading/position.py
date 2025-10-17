"""
Position management for take profit/stop loss functionality.
"""

from dataclasses import dataclass
from enum import Enum
from time import time

from solders.pubkey import Pubkey
from interfaces.core import Platform
from core.pubkeys import LAMPORTS_PER_SOL, TOKEN_DECIMALS
from utils.logger import get_logger

logger = get_logger(__name__)

class ExitReason(Enum):
    """Reasons for position exit."""

    TAKE_PROFIT = "take_profit"
    STOP_LOSS = "stop_loss"
    TRAILING_STOP = "trailing_stop"
    MAX_HOLD_TIME = "max_hold_time"
    MANUAL = "manual"


@dataclass
class Position:
    """Represents an active trading position."""

    # Token information
    mint: Pubkey
    symbol: str
    platform: Platform

    # Position details
    entry_price: float
    token_quantity_decimal: float  # Renamed from quantity, kept for business logic
    original_token_amount_raw: int  # Renamed from token_amount_raw
    current_token_amount_raw: int  # New field to track current amount
    entry_ts: int  # Unix epoch milliseconds (renamed from entry_time)
    exit_strategy: str  # New field from config

    # Exit conditions (kept for business logic, not persisted to DB)
    take_profit_price: float | None = None
    stop_loss_price: float | None = None
    max_hold_time: int | None = None  # seconds
    # Trailing stop configuration/state
    trailing_stop_percentage: float | None = None  # fraction (e.g., 0.2 for 20%)
    highest_price: float | None = None  # highest observed price since entry

    # Status
    is_active: bool = True
    exit_reason: ExitReason | None = None
    exit_price: float | None = None
    exit_ts: int | None = None  # Unix epoch milliseconds (renamed from exit_time)
    
    # Fees (lamports) - accumulated
    transaction_fee_raw: int | None = None  # Replaces buy_fee_raw and sell_fee_raw
    platform_fee_raw: int | None = None  # New field
    
    # PnL tracking
    realized_pnl: float | None = None  # Calculated when position closes

    @classmethod
    def create_from_buy_result(
        cls,
        mint: Pubkey,
        symbol: str,
        platform: Platform,
        entry_price: float,
        quantity: float,
        token_amount_raw: int,
        entry_ts: int,
        transaction_fee_raw: int | None,
        platform_fee_raw: int | None,
        exit_strategy: str,
        take_profit_percentage: float | None,
        stop_loss_percentage: float | None,
        trailing_stop_percentage: float | None,
        max_hold_time: int | None,
    ) -> "Position":
        """Create a position from a successful buy transaction.

        Args:
            mint: Token mint address
            symbol: Token symbol
            platform: Trading platform
            entry_price: Price at which position was entered
            quantity: Quantity of tokens purchased (decimal)
            token_amount_raw: Raw token amount purchased (integer)
            entry_ts: Unix epoch milliseconds from block time (or current time if unavailable)
            transaction_fee_raw: Transaction fee in lamports
            platform_fee_raw: Platform fee in lamports
            exit_strategy: Exit strategy from config
            take_profit_percentage: Take profit percentage (0.5 = 50% profit)
            stop_loss_percentage: Stop loss percentage (0.2 = 20% loss)
            trailing_stop_percentage: Trailing stop percentage
            max_hold_time: Maximum hold time in seconds

        Returns:
            Position instance
        """
        take_profit_price = None
        if take_profit_percentage is not None:
            take_profit_price = entry_price * (1 + take_profit_percentage)

        stop_loss_price = None
        if stop_loss_percentage is not None:
            stop_loss_price = entry_price * (1 - stop_loss_percentage)

        result = cls(
            mint=mint,
            symbol=symbol,
            platform=platform,
            entry_price=entry_price,
            token_quantity_decimal=quantity,
            original_token_amount_raw=token_amount_raw,
            current_token_amount_raw=token_amount_raw,
            entry_ts=entry_ts,  # Unix epoch milliseconds from block time
            exit_strategy=exit_strategy,
            take_profit_price=take_profit_price,
            stop_loss_price=stop_loss_price,
            max_hold_time=max_hold_time,
            transaction_fee_raw=transaction_fee_raw,
            platform_fee_raw=platform_fee_raw,
            trailing_stop_percentage=trailing_stop_percentage,
            highest_price=entry_price,
        )
        return result

    def should_exit(self, current_price: float) -> tuple[bool, ExitReason | None]:
        """Check if position should be exited based on current conditions.

        Args:
            current_price: Current token price

        Returns:
            Tuple of (should_exit, exit_reason)
        """
        if not self.is_active:
            return False, None

        # Update highest price for trailing stop tracking
        if self.highest_price is None or current_price > self.highest_price:
            self.highest_price = current_price
            logger.info(f"Highest price updated to {self.highest_price}")

        # Check take profit
        if self.take_profit_price and current_price >= self.take_profit_price:
            return True, ExitReason.TAKE_PROFIT
        logger.info(f"current_price {current_price} < take_profit_price {self.take_profit_price}")

        # Check stop loss
        if self.stop_loss_price and current_price <= self.stop_loss_price:
            return True, ExitReason.STOP_LOSS
        logger.info(f"current_price {current_price} > stop_loss_price {self.stop_loss_price}")

        # Check trailing stop (if configured)
        if self.trailing_stop_percentage is not None and self.highest_price is not None:
            trailing_limit = self.highest_price * (1 - self.trailing_stop_percentage)
            if current_price <= trailing_limit:
                return True, ExitReason.TRAILING_STOP
            logger.info(f"current_price {current_price} > trailing_limit {trailing_limit}")

        # Check max hold time
        if self.max_hold_time:
            current_ts = int(time() * 1000)
            elapsed_time = (current_ts - self.entry_ts) / 1000  # Convert to seconds
            if elapsed_time >= self.max_hold_time:
                return True, ExitReason.MAX_HOLD_TIME
            logger.info(f"elapsed_time {elapsed_time} < max_hold_time {self.max_hold_time}")

        return False, None

    def close_position(self, exit_price: float, exit_reason: ExitReason, transaction_fee_raw: int | None = None, platform_fee_raw: int | None = None) -> None:
        """Close the position with exit details.

        Args:
            exit_price: Price at which position was exited
            exit_reason: Reason for exit
            transaction_fee_raw: Additional transaction fee from sell
            platform_fee_raw: Additional platform fee from sell
        """
        self.is_active = False
        self.exit_price = exit_price
        self.exit_reason = exit_reason
        self.exit_ts = int(time() * 1000)  # Unix epoch milliseconds
        
        # Accumulate fees
        if transaction_fee_raw is not None:
            self.transaction_fee_raw = (self.transaction_fee_raw or 0) + transaction_fee_raw
        if platform_fee_raw is not None:
            self.platform_fee_raw = (self.platform_fee_raw or 0) + platform_fee_raw
            
        # Calculate realized PnL
        self.realized_pnl = self._calculate_realized_pnl(exit_price)

    def get_net_pnl(self, current_price: float | None = None) -> dict:
        """Calculate profit/loss for the position.

        Args:
            current_price: Current price (uses exit_price if position is closed)

        Returns:
            Dictionary with PnL information
        """
        if self.is_active and current_price is None:
            raise ValueError("current_price required for active position")

        price_to_use = self.exit_price if not self.is_active else current_price
        if price_to_use is None:
            raise ValueError("No price available for PnL calculation")
        
        if self.entry_price is None:
            raise ValueError("No entry price available for PnL calculation")
        
        if self.token_quantity_decimal is None:
            raise ValueError("No quantity available for PnL calculation")

        price_change = price_to_use - self.entry_price
        price_change_pct = (price_change / self.entry_price) * 100
        gross_pnl_sol = price_change * self.token_quantity_decimal

        transaction_fee_raw = int(self.transaction_fee_raw or 0)
        platform_fee_raw = int(self.platform_fee_raw or 0)
        total_fees_raw = transaction_fee_raw + platform_fee_raw

        # Use the helper method for PnL calculation
        net_pnl_sol = self._calculate_realized_pnl(price_to_use)

        if self.is_active:
            return {
                "entry_price": self.entry_price,
                "current_price": price_to_use,
                "price_change": price_change,
                "price_change_pct": price_change_pct,
                "gross_pnl_sol": gross_pnl_sol,
                "net_unrealized_pnl_sol": net_pnl_sol,
                "quantity": self.token_quantity_decimal,
                "transaction_fee_raw": transaction_fee_raw,
                "platform_fee_raw": platform_fee_raw,
                "total_fees_raw": total_fees_raw,
                "total_fees_sol": total_fees_raw / LAMPORTS_PER_SOL,
            }
        else:
            return {
                "entry_price": self.entry_price,
                "current_price": price_to_use,
                "price_change": price_change,
                "price_change_pct": price_change_pct,
                "gross_pnl_sol": gross_pnl_sol,
                "net_realized_pnl_sol": net_pnl_sol,
                "quantity": self.token_quantity_decimal,
                "transaction_fee_raw": transaction_fee_raw,
                "platform_fee_raw": platform_fee_raw,
                "total_fees_raw": total_fees_raw,
                "total_fees_sol": total_fees_raw / LAMPORTS_PER_SOL,
            }

    def _calculate_realized_pnl(self, exit_price: float) -> float:
        """Calculate realized PnL when position is closed.
        
        Args:
            exit_price: Price at which position was exited
            
        Returns:
            Realized PnL in SOL
        """
        if self.entry_price is None or self.token_quantity_decimal is None:
            return 0.0
            
        price_change = exit_price - self.entry_price
        gross_pnl_sol = price_change * self.token_quantity_decimal
        
        transaction_fee_raw = int(self.transaction_fee_raw or 0)
        platform_fee_raw = int(self.platform_fee_raw or 0)
        total_fees_raw = transaction_fee_raw + platform_fee_raw
        
        return gross_pnl_sol - (float(total_fees_raw) / LAMPORTS_PER_SOL)

    def __str__(self) -> str:
        """String representation of position."""
        if self.is_active:
            status = "ACTIVE"
        elif self.exit_reason:
            status = f"CLOSED ({self.exit_reason.value})"
        else:
            status = "CLOSED (UNKNOWN)"
        quantity_str = f"{self.token_quantity_decimal:.6f}" if self.token_quantity_decimal is not None else "None"
        quantity_raw_str = f"{self.original_token_amount_raw}" if self.original_token_amount_raw is not None else "None"
        price_str = f"{self.entry_price:.8f}" if self.entry_price is not None else "None"
        return f"Position({self.symbol}: {quantity_str} ({quantity_raw_str} raw) @ {price_str} SOL - {status})"
