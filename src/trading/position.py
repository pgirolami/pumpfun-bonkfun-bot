"""
Position management for take profit/stop loss functionality.
"""

from dataclasses import dataclass
from enum import Enum
from time import time
from typing import TYPE_CHECKING
import hashlib

from solders.pubkey import Pubkey
from interfaces.core import Platform
from core.pubkeys import LAMPORTS_PER_SOL, TOKEN_DECIMALS
from utils.logger import get_logger

if TYPE_CHECKING:
    from trading.base import TradeResult

logger = get_logger(__name__)

class ExitReason(Enum):
    """Reasons for position exit."""

    TAKE_PROFIT = "take_profit"
    STOP_LOSS = "stop_loss"
    TRAILING_STOP = "trailing_stop"
    MAX_HOLD_TIME = "max_hold_time"
    NO_PRICE_CHANGE = "no_price_change"
    INSUFFICIENT_GAIN = "insufficient_gain"
    MANUAL = "manual"
    FAILED_BUY = "failed_buy"
    CREATOR_SOLD = "creator_sold"


@dataclass
class Position:
    """Represents an active trading position."""

    # Position identification
    position_id: str  # Unique identifier for this position
    
    # Token information
    mint: Pubkey
    platform: Platform

    # Position details
    entry_net_price_decimal: float  # Net price from TradeResult.net_price_sol_decimal()
    token_quantity_decimal: float  # Renamed from quantity, kept for business logic
    total_token_swapin_amount_raw: int  # Total tokens bought
    total_token_swapout_amount_raw: int  # Total tokens sold (accumulation)
    entry_ts: int  # Unix epoch milliseconds
    exit_strategy: str  # New field from config

    # Exit conditions (kept for business logic, not persisted to DB)
    take_profit_price: float | None = None
    stop_loss_price: float | None = None
    max_hold_time: int | None = None  # seconds
    max_no_price_change_time: int | None = None  # seconds without price change
    # Trailing stop configuration/state
    trailing_stop_percentage: float | None = None  # fraction (e.g., 0.2 for 20%)
    highest_price: float | None = None  # highest observed price since entry (already net decimal from on-chain)
    last_price_change_ts: float | None = None  # timestamp of last price change
    
    # Insufficient gain exit condition
    min_gain_percentage: float | None = None  # minimum gain required within time window
    min_gain_time_window: int = 2  # seconds to check for minimum gain (configurable)
    
    # Monitoring timing
    monitoring_start_ts: int | None = None  # when position monitoring actually started (Unix epoch milliseconds)
    
    # Creator tracking (raw token units)
    creator_token_swap_in: int = 0    # Tokens bought by creator
    creator_token_swap_out: int = 0    # Tokens sold by creator (negative)

    # Status
    is_active: bool = True
    exit_reason: ExitReason | None = None
    exit_net_price_decimal: float | None = None  # Net price from TradeResult.net_price_sol_decimal()
    exit_ts: int | None = None  # Unix epoch milliseconds (renamed from exit_time)
    
    # Fees (lamports) - accumulated
    transaction_fee_raw: int | None = None  # Replaces buy_fee_raw and sell_fee_raw
    platform_fee_raw: int | None = None  # New field
    
    # PnL tracking
    realized_pnl_sol_decimal: float | None = None  # From get_net_pnl()["realized_pnl_sol_decimal"]
    realized_net_pnl_sol_decimal: float | None = None  # From get_net_pnl()["realized_net_pnl_sol_decimal"]
    
    # Investment tracking
    buy_amount: float | None = None  # Intended SOL amount to invest
    total_net_sol_swapout_amount_raw: int | None = None  # Total SOL spent on buys
    total_net_sol_swapin_amount_raw: int | None = None  # Total SOL received from sells (starts at 0)

    @staticmethod
    def generate_position_id(mint: Pubkey, platform: Platform, entry_ts: int) -> str:
        """Generate position ID hash from mint + platform + entry_ts.
        
        Args:
            mint: Token mint address
            platform: Trading platform
            entry_ts: Entry timestamp in milliseconds
            
        Returns:
            SHA256 hash as hex string
        """
        data = f"{mint!s}_{platform.value}_{entry_ts}"
        return hashlib.sha256(data.encode()).hexdigest()

    @classmethod
    def create_from_buy_result(
        cls,
        mint: Pubkey,
        platform: Platform,
        entry_net_price_decimal: float,
        quantity: float,
        token_swapin_amount_raw: int,
        entry_ts: int,
        transaction_fee_raw: int | None,
        platform_fee_raw: int | None,
        exit_strategy: str,
        buy_amount: float,
        total_net_sol_swapout_amount_raw: int | None,
        take_profit_percentage: float | None,
        stop_loss_percentage: float | None,
        trailing_stop_percentage: float | None,
        max_hold_time: int | None,
        max_no_price_change_time: int | None = None,
        min_gain_percentage: float | None = None,
        min_gain_time_window: int = 2,
    ) -> "Position":
        """Create a position from a successful buy transaction.

        Args:
            mint: Token mint address
            platform: Trading platform
        entry_net_price_decimal: Net price at which position was entered (from TradeResult.net_price_sol_decimal())
        quantity: Quantity of tokens purchased (decimal)
        token_swapin_amount_raw: Raw token amount purchased (integer)
        entry_ts: Unix epoch milliseconds from block time (or current time if unavailable)
        transaction_fee_raw: Transaction fee in lamports
        platform_fee_raw: Platform fee in lamports
        exit_strategy: Exit strategy from config
        buy_amount: Intended SOL amount to invest
        total_net_sol_swapout_amount_raw: Total SOL amount spent in lamports for the buy
            take_profit_percentage: Take profit percentage (0.5 = 50% profit)
            stop_loss_percentage: Stop loss percentage (0.2 = 20% loss)
            trailing_stop_percentage: Trailing stop percentage
            max_hold_time: Maximum hold time in seconds
            max_no_price_change_time: Maximum time without price change in seconds
            min_gain_percentage: Minimum gain percentage required within time window (0.1 = 10%)
            min_gain_time_window: Time window in seconds to check for minimum gain (default: 2)

        Returns:
            Position instance
        """
        take_profit_price = None
        if take_profit_percentage is not None:
            take_profit_price = entry_net_price_decimal * (1 + take_profit_percentage)

        stop_loss_price = None
        if stop_loss_percentage is not None:
            stop_loss_price = entry_net_price_decimal * (1 - stop_loss_percentage)

        # Generate position_id
        position_id = cls.generate_position_id(mint, platform, entry_ts)
        
        result = cls(
            position_id=position_id,
            mint=mint,
            platform=platform,
            entry_net_price_decimal=entry_net_price_decimal,
            token_quantity_decimal=quantity,
            total_token_swapin_amount_raw=token_swapin_amount_raw,
            total_token_swapout_amount_raw=0,  # Start at 0, accumulates from sells
            entry_ts=entry_ts,  # Unix epoch milliseconds from block time
            exit_strategy=exit_strategy,
            take_profit_price=take_profit_price,
            stop_loss_price=stop_loss_price,
            max_hold_time=max_hold_time,
            max_no_price_change_time=max_no_price_change_time,
            transaction_fee_raw=transaction_fee_raw,
            platform_fee_raw=platform_fee_raw,
            trailing_stop_percentage=trailing_stop_percentage,
            highest_price=entry_net_price_decimal,
            last_price_change_ts=time(),  # Initialize with current time
            min_gain_percentage=min_gain_percentage,
            min_gain_time_window=min_gain_time_window,
            buy_amount=buy_amount,
            total_net_sol_swapout_amount_raw=total_net_sol_swapout_amount_raw,
            total_net_sol_swapin_amount_raw=0,  # Start at 0, accumulates from sells
        )
        
        # Apply exit strategy configuration to override raw percentages
        result._apply_exit_strategy_config(
            exit_strategy=exit_strategy,
            take_profit_percentage=take_profit_percentage,
            stop_loss_percentage=stop_loss_percentage,
            trailing_stop_percentage=trailing_stop_percentage,
            max_hold_time=max_hold_time,
            max_no_price_change_time=max_no_price_change_time,
        )
        
        return result

    def _apply_exit_strategy_config(
        self,
        exit_strategy: str,
        take_profit_percentage: float | None,
        stop_loss_percentage: float | None,
        trailing_stop_percentage: float | None,
        max_hold_time: int | None,
        max_no_price_change_time: int | None,
    ) -> None:
        """Apply exit strategy configuration to override raw percentages.
        
        This method ensures that the position's exit conditions are set correctly
        based on the exit strategy, overriding any raw percentage calculations.
        """
        self.exit_strategy = exit_strategy
        entry_price = self.entry_net_price_decimal
        
        if entry_price is not None:
            if exit_strategy == "tp_sl":
                # Take profit and stop loss strategy
                self.take_profit_price = (
                    entry_price * (1 + take_profit_percentage)
                    if take_profit_percentage is not None
                    else None
                )
                self.stop_loss_price = (
                    entry_price * (1 - stop_loss_percentage)
                    if stop_loss_percentage is not None
                    else None
                )
                self.trailing_stop_percentage = None
            elif exit_strategy == "trailing":
                # Trailing stop strategy - can have both fixed stop loss and trailing stop
                self.take_profit_price = (
                    entry_price * (1 + take_profit_percentage)
                    if take_profit_percentage is not None
                    else None
                )
                # Allow fixed stop loss for trailing strategy (protects against immediate drops)
                self.stop_loss_price = (
                    entry_price * (1 - stop_loss_percentage)
                    if stop_loss_percentage is not None
                    else None
                )
                self.trailing_stop_percentage = trailing_stop_percentage
            elif exit_strategy == "time_based":
                # Time-based strategy - only take profit, no stop loss
                self.take_profit_price = (
                    entry_price * (1 + take_profit_percentage)
                    if take_profit_percentage is not None
                    else None
                )
                self.stop_loss_price = None
                self.trailing_stop_percentage = None
            elif exit_strategy == "manual":
                # Manual strategy - no automatic exits
                self.take_profit_price = None
                self.stop_loss_price = None
                self.trailing_stop_percentage = None
        
        # Update timing configuration
        self.max_hold_time = max_hold_time
        self.max_no_price_change_time = max_no_price_change_time

    def get_current_token_balance_raw(self) -> int:
        """Get current token balance available to sell.
        
        Returns:
            Current token balance in raw units (total bought - total sold)
        """
        return (self.total_token_swapin_amount_raw or 0) - (self.total_token_swapout_amount_raw or 0)

    def set_monitoring_start_time(self, timestamp: int) -> None:
        """Set the monitoring start time (when first price was received).
        
        Args:
            timestamp: Unix epoch milliseconds when monitoring started
        """
        if self.monitoring_start_ts is None:
            self.monitoring_start_ts = timestamp
            logger.debug(f"Position monitoring started at {timestamp} (entry was at {self.entry_ts})")

    def should_exit(self, current_price: float) -> tuple[bool, ExitReason | None]:
        """Check if position should be exited based on current conditions.

        Args:
            current_price: Current token price

        Returns:
            Tuple of (should_exit, exit_reason)
        """

        if not self.is_active:
            return False, None

        if self.creator_token_swap_out!=0:
            logger.info(f"Creator has sold tokens: {self.creator_token_swap_out}")
            return True, ExitReason.CREATOR_SOLD

        # Check take profit
        if self.take_profit_price and current_price >= self.take_profit_price:
#            logger.info(f"current_price {current_price} >= take_profit_price {self.take_profit_price}")
            return True, ExitReason.TAKE_PROFIT
#        logger.info(f"current_price {current_price} < take_profit_price {self.take_profit_price}")

        # Check stop loss
        if self.stop_loss_price and current_price <= self.stop_loss_price:
#            logger.info(f"current_price {current_price} <= stop_loss_price {self.stop_loss_price}")
            return True, ExitReason.STOP_LOSS
#        logger.info(f"current_price {current_price} > stop_loss_price {self.stop_loss_price}")

        # Check trailing stop (if configured)
        if self.trailing_stop_percentage is not None and self.highest_price is not None:
            trailing_limit = self.highest_price * (1 - self.trailing_stop_percentage)
            if current_price <= trailing_limit:
                # logger.info(f"current_price {current_price} <= trailing_limit {trailing_limit}")
                return True, ExitReason.TRAILING_STOP
            # logger.info(f"current_price {current_price} > trailing_limit {trailing_limit}")

        # Check max hold time
        if self.max_hold_time:
            current_ts = int(time() * 1000)
            # Max hold time is based on entry time (total time position has been open)
            elapsed_time = (current_ts - self.entry_ts) / 1000  # Convert to seconds
            if elapsed_time >= self.max_hold_time:
#                logger.info(f"elapsed_time {elapsed_time} >= max_hold_time {self.max_hold_time}")
                return True, ExitReason.MAX_HOLD_TIME
#            logger.info(f"elapsed_time {elapsed_time} < max_hold_time {self.max_hold_time}")

        # Check max no price change time
        if self.max_no_price_change_time and self.last_price_change_ts is not None:
            current_time = time()
            time_since_price_change = current_time - self.last_price_change_ts
            if time_since_price_change >= self.max_no_price_change_time:
                # logger.info(f"time_since_price_change {time_since_price_change} >= max_no_price_change_time {self.max_no_price_change_time}")
                return True, ExitReason.NO_PRICE_CHANGE
            # logger.info(f"time_since_price_change {time_since_price_change} < max_no_price_change_time {self.max_no_price_change_time}")
        else:
            logger.info(f"max_no_price_change_time is not set : {self.max_no_price_change_time} and {self.last_price_change_ts}")

        # Check insufficient gain within time window
        if self.min_gain_percentage is not None:
            current_ts = int(time() * 1000)
            # Only check if monitoring has started (don't use entry time)
            if self.monitoring_start_ts is not None:
                elapsed_time = (current_ts - self.monitoring_start_ts) / 1000  # Convert to seconds
                
                # Only check if we're within the time window
                if elapsed_time >= self.min_gain_time_window:
                    # Calculate current gain percentage
                    current_gain = (current_price - self.entry_net_price_decimal) / self.entry_net_price_decimal
                    
                    if current_gain < self.min_gain_percentage:
                        logger.info(f"current_gain {current_gain:.4f} < min_gain_percentage {self.min_gain_percentage:.4f} after {elapsed_time:.1f}s")
                        return True, ExitReason.INSUFFICIENT_GAIN

        return False, None

    def close_position(self, sell_result: "TradeResult", exit_reason: ExitReason) -> None:
        """Close the position with exit details.

        Args:
            sell_result: TradeResult from the sell transaction
            exit_reason: Reason for exit
        """
        self.is_active = False
        self.exit_net_price_decimal = sell_result.net_price_sol_decimal()
        self.exit_reason = exit_reason
        self.exit_ts = int(time() * 1000)  # Unix epoch milliseconds
        
        # Update swap amounts from the final sell
        if sell_result.token_swap_amount_raw:
            self.total_token_swapout_amount_raw = (self.total_token_swapout_amount_raw or 0) + sell_result.token_swap_amount_raw
        if sell_result.sol_swap_amount_raw:
            self.total_net_sol_swapin_amount_raw = (self.total_net_sol_swapin_amount_raw or 0) + sell_result.sol_swap_amount_raw
        
        # Update highest price if exit price is higher
        exit_price = sell_result.net_price_sol_decimal()
        if self.highest_price is None or exit_price > self.highest_price:
            self.highest_price = exit_price
        
        # Accumulate fees
        self.transaction_fee_raw = (self.transaction_fee_raw or 0) + (sell_result.transaction_fee_raw or 0)
        self.platform_fee_raw = (self.platform_fee_raw or 0) + (sell_result.platform_fee_raw or 0)
            
        # Calculate realized PnL using get_net_pnl method
        pnl_dict = self._get_pnl()  # No parameter needed since position is now inactive
        self.realized_pnl_sol_decimal = pnl_dict["realized_pnl_sol_decimal"]
        self.realized_net_pnl_sol_decimal = pnl_dict["realized_net_pnl_sol_decimal"]

    def _get_pnl(self, current_price: float | None = None) -> dict:
        """Calculate profit/loss for the position.

        Args:
            current_price: Current price (uses exit_price if position is closed)

        Returns:
            Dictionary with PnL information
        """
        if self.is_active and current_price is None:
            raise ValueError("current_price required for active position")

        # Special case for failed buys - only calculate loss from fees
        if self.exit_reason == ExitReason.FAILED_BUY:
            transaction_fee_raw = int(self.transaction_fee_raw or 0)
            platform_fee_raw = int(self.platform_fee_raw or 0)
            total_fees_raw = transaction_fee_raw + platform_fee_raw
            total_fees_sol = float(total_fees_raw) / LAMPORTS_PER_SOL
            
            return {
                "current_price": 0.0,
                "net_price_change_decimal": 0.0,
                "net_price_change_pct": 0.0,
                "realized_pnl_sol_decimal": -total_fees_sol,  # Loss from fees only
                "realized_net_pnl_sol_decimal": 0.0,  # No net PnL since no tokens were acquired
                "quantity": 0.0,
                "transaction_fee_raw": transaction_fee_raw,
                "platform_fee_raw": platform_fee_raw,
                "total_fees_raw": total_fees_raw,
            }

        price_to_use = self.exit_net_price_decimal if not self.is_active else current_price
        if price_to_use is None:
            raise ValueError("No price available for PnL calculation")
        
        if self.entry_net_price_decimal is None:
            raise ValueError("No entry price available for PnL calculation")
        
        if self.token_quantity_decimal is None:
            raise ValueError("No quantity available for PnL calculation")

        net_price_change_decimal = price_to_use - self.entry_net_price_decimal
        net_price_change_pct = (net_price_change_decimal / self.entry_net_price_decimal) * 100

        transaction_fee_raw = int(self.transaction_fee_raw or 0)
        platform_fee_raw = int(self.platform_fee_raw or 0)
        total_fees_raw = transaction_fee_raw + platform_fee_raw

        # Calculate net PnL (consolidated from _calculate_realized_pnl)
        net_pnl_sol = (net_price_change_decimal * self.token_quantity_decimal)
        gross_pnl_sol = net_pnl_sol - (float(total_fees_raw) / LAMPORTS_PER_SOL)

        if self.is_active:
            return {
                "entry_price": self.entry_net_price_decimal,
                "current_price": price_to_use,
                "net_price_change_decimal": net_price_change_decimal,
                "net_price_change_pct": net_price_change_pct,
                "realized_pnl_sol_decimal": gross_pnl_sol, 
                "realized_net_pnl_sol_decimal": net_pnl_sol,  
                "quantity": self.token_quantity_decimal,
                "transaction_fee_raw": transaction_fee_raw,
                "platform_fee_raw": platform_fee_raw,
                "total_fees_raw": total_fees_raw,
                "total_fees_sol": total_fees_raw / LAMPORTS_PER_SOL,
            }
        else:
            return {
                "entry_price": self.entry_net_price_decimal,
                "current_price": price_to_use,
                "net_price_change_decimal": net_price_change_decimal,
                "net_price_change_pct": net_price_change_pct,
                "realized_pnl_sol_decimal": gross_pnl_sol,
                "realized_net_pnl_sol_decimal": net_pnl_sol,
                "quantity": self.token_quantity_decimal,
                "transaction_fee_raw": transaction_fee_raw,
                "platform_fee_raw": platform_fee_raw,
                "total_fees_raw": total_fees_raw,
                "total_fees_sol": total_fees_raw / LAMPORTS_PER_SOL,
            }

    def update_creator_tracking(self, creator_swap_in: int, creator_swap_out: int) -> None:
        """Update creator token swap tracking from trade tracker.
        
        Args:
            creator_swap_in: Total tokens bought by creator (raw units)
            creator_swap_out: Total tokens sold by creator (raw units, negative)
        """
        self.creator_token_swap_in = creator_swap_in
        self.creator_token_swap_out = creator_swap_out
        logger.debug(f"[{str(self.mint)[:8]}] Updated creator tracking: swap_in={creator_swap_in}, swap_out={creator_swap_out}")

    def get_creator_net_position(self) -> int:
        """Get creator's net token position (bought - sold).
        
        Returns:
            Net position in raw token units (positive = net buyer, negative = net seller)
        """
        return self.creator_token_swap_in + self.creator_token_swap_out

    def __str__(self) -> str:
        """String representation of position."""
        if self.is_active:
            status = "ACTIVE"
        elif self.exit_reason:
            status = f"CLOSED ({self.exit_reason.value})"
        else:
            status = "CLOSED (UNKNOWN)"
        quantity_str = f"{self.token_quantity_decimal:.6f}" if self.token_quantity_decimal is not None else "None"
        quantity_raw_str = f"{self.get_current_token_balance_raw()}" if self.total_token_swapin_amount_raw is not None else "None"
        price_str = f"{self.entry_net_price_decimal:.10f}" if self.entry_net_price_decimal is not None else "None"
        return f"Position({str(self.mint)}: {quantity_str} ({quantity_raw_str} raw) @ {price_str} SOL - {status})"
