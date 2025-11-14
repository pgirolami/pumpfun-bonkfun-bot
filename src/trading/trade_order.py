"""Trade order classes that encapsulate order parameters through the trading flow."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from interfaces.core import TokenInfo


class OrderState(Enum):
    """Order execution state."""
    
    UNSENT = "unsent"  # Transaction has not been sent
    SENT = "sent"  # Transaction has been sent
    CONFIRMED = "confirmed"  # Transaction was confirmed successfully
    FAILED = "failed"  # Transaction failed


@dataclass
class Order(ABC):
    """Base order class with common fields."""
    
    # Identity
    token_info: TokenInfo  # Contains mint, symbol, platform
    
    # Order state
    state: OrderState = OrderState.UNSENT  # Execution state of the order
    
    # Price information (calculated)
    token_price_sol: float | None = None
    
    # Transaction parameters (calculated)
    priority_fee: int | None = None
    compute_unit_limit: int | None = None
    account_data_size_limit: int | None = None
    
    # Transaction execution (populated after send)
    tx_signature: str | None = None
    
    # Fees (populated after execution)
    transaction_fee_raw: int | None = None
    protocol_fee_raw: int | None = None
    creator_fee_raw: int | None = None

    block_ts: int | None = None
    
    @property
    @abstractmethod
    def trade_type(self) -> str:
        """Return the trade type: 'buy' or 'sell'."""
        pass
    
    @property
    def mint(self) -> str:
        """Get the token mint address."""
        return str(self.token_info.mint)
    
    @property
    def platform(self) -> str:
        """Get the platform name."""
        return self.token_info.platform.value


@dataclass
class BuyOrder(Order):
    """Buy order with buy-specific parameters."""
    
    # Input
    sol_amount_raw: int = 0  # SOL to spend in lamports
    max_sol_amount_raw: int | None = None  # Max SOL cost with slippage tolerance
    
    # Calculated during preparation
    token_amount_raw: int | None = None  # Expected tokens (before slippage)
    minimum_token_swap_amount_raw: int | None = None  # Min tokens with slippage
    
    # Execution state
    slippage_failed: bool = False  # Flag for dry-run slippage simulation
    
    @property
    def trade_type(self) -> str:
        return "buy"
    
    @property
    def expected_token_amount_raw(self) -> int | None:
        """Expected token amount for database storage."""
        return self.token_amount_raw
    
    @property
    def expected_sol_amount_raw(self) -> int | None:
        """Expected SOL amount for database storage."""
        return self.sol_amount_raw


@dataclass
class SellOrder(Order):
    """Sell order with sell-specific parameters."""
    
    # Input
    token_amount_raw: int = 0  # Tokens to sell
    
    # Calculated during preparation
    expected_sol_swap_amount_raw: int | None = None  # Expected SOL from sale
    minimum_sol_swap_amount_raw: int | None = None  # Min SOL with slippage
    
    @property
    def trade_type(self) -> str:
        return "sell"
    
    @property
    def expected_token_amount_raw(self) -> int | None:
        """Expected token amount for database storage."""
        return self.token_amount_raw
    
    @property
    def expected_sol_amount_raw(self) -> int | None:
        """Expected SOL amount for database storage."""
        return self.expected_sol_swap_amount_raw