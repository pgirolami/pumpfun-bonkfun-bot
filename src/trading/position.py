"""
Position management for take profit/stop loss functionality.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from solders.pubkey import Pubkey
from core.pubkeys import LAMPORTS_PER_SOL


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

    # Position details
    entry_price: float
    quantity: float
    entry_time: datetime

    # Exit conditions
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
    exit_time: datetime | None = None
    # Fees (lamports)
    buy_fee_lamports: int | None = None
    sell_fee_lamports: int | None = None

    @classmethod
    def create_from_buy_result(
        cls,
        mint: Pubkey,
        symbol: str,
        entry_price: float,
        quantity: float,
        buy_fee_lamports: int | None,
        take_profit_percentage: float | None,
        stop_loss_percentage: float | None,
        trailing_stop_percentage: float | None,
        max_hold_time: int | None,
    ) -> "Position":
        """Create a position from a successful buy transaction.

        Args:
            mint: Token mint address
            symbol: Token symbol
            entry_price: Price at which position was entered
            quantity: Quantity of tokens purchased
            take_profit_percentage: Take profit percentage (0.5 = 50% profit)
            stop_loss_percentage: Stop loss percentage (0.2 = 20% loss)
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

        return cls(
            mint=mint,
            symbol=symbol,
            entry_price=entry_price,
            quantity=quantity,
            entry_time=datetime.utcnow(),
            take_profit_price=take_profit_price,
            stop_loss_price=stop_loss_price,
            max_hold_time=max_hold_time,
            buy_fee_lamports=buy_fee_lamports,
            trailing_stop_percentage=trailing_stop_percentage,
            highest_price=entry_price,
        )

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

        # Check take profit
        if self.take_profit_price and current_price >= self.take_profit_price:
            return True, ExitReason.TAKE_PROFIT

        # Check stop loss
        if self.stop_loss_price and current_price <= self.stop_loss_price:
            return True, ExitReason.STOP_LOSS

        # Check trailing stop (if configured)
        if self.trailing_stop_percentage is not None and self.highest_price is not None:
            trailing_limit = self.highest_price * (1 - self.trailing_stop_percentage)
            if current_price <= trailing_limit:
                return True, ExitReason.TRAILING_STOP

        # Check max hold time
        if self.max_hold_time:
            elapsed_time = (datetime.utcnow() - self.entry_time).total_seconds()
            if elapsed_time >= self.max_hold_time:
                return True, ExitReason.MAX_HOLD_TIME

        return False, None

    def close_position(self, exit_price: float, exit_reason: ExitReason, sell_fee_lamports: int | None = None) -> None:
        """Close the position with exit details.

        Args:
            exit_price: Price at which position was exited
            exit_reason: Reason for exit
        """
        self.is_active = False
        self.exit_price = exit_price
        self.exit_reason = exit_reason
        self.exit_time = datetime.utcnow()
        self.sell_fee_lamports = sell_fee_lamports

    def get_pnl(self, current_price: float | None = None) -> dict:
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

        price_change = price_to_use - self.entry_price
        price_change_pct = (price_change / self.entry_price) * 100
        gross_pnl_sol = price_change * self.quantity

        buy_fee_lamports = int(self.buy_fee_lamports or 0)
        sell_fee_lamports = int(self.sell_fee_lamports or 0)

        if self.is_active:
            net_unrealized_pnl_sol = gross_pnl_sol - (buy_fee_lamports / LAMPORTS_PER_SOL)
            return {
                "entry_price": self.entry_price,
                "current_price": price_to_use,
                "price_change": price_change,
                "price_change_pct": price_change_pct,
                "unrealized_pnl_sol": net_unrealized_pnl_sol,
                "quantity": self.quantity,
                "buy_fee_lamports": buy_fee_lamports,
                "total_fees_lamports": buy_fee_lamports,
                "total_fees_sol": buy_fee_lamports / LAMPORTS_PER_SOL,
            }
        else:
            total_fees_lamports = buy_fee_lamports + sell_fee_lamports
            net_realized_pnl_sol = gross_pnl_sol - (total_fees_lamports / LAMPORTS_PER_SOL)
            return {
                "entry_price": self.entry_price,
                "current_price": price_to_use,
                "price_change": price_change,
                "price_change_pct": price_change_pct,
                "realized_pnl_sol": net_realized_pnl_sol,
                "quantity": self.quantity,
                "buy_fee_lamports": buy_fee_lamports,
                "sell_fee_lamports": sell_fee_lamports,
                "total_fees_lamports": total_fees_lamports,
                "total_fees_sol": total_fees_lamports / LAMPORTS_PER_SOL,
            }

    def __str__(self) -> str:
        """String representation of position."""
        if self.is_active:
            status = "ACTIVE"
        elif self.exit_reason:
            status = f"CLOSED ({self.exit_reason.value})"
        else:
            status = "CLOSED (UNKNOWN)"
        return f"Position({self.symbol}: {self.quantity:.6f} @ {self.entry_price:.8f} SOL - {status})"
