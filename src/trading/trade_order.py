"""Trade order classes that encapsulate order parameters through the trading flow."""

from dataclasses import dataclass
from interfaces.core import TokenInfo


@dataclass
class BuyOrder:
    """Buy order with buy-specific parameters."""
    
    # Identity
    token_info: TokenInfo  # Contains mint, symbol, platform
    
    # Input
    sol_amount_raw: int  # SOL to spend in lamports
    
    # Price information (calculated)
    token_price_sol: float | None = None
    
    # Transaction parameters (calculated)
    priority_fee: int | None = None
    compute_unit_limit: int | None = None
    
    # Transaction execution (populated after send)
    tx_signature: str | None = None
    
    # Fees (populated after execution)
    transaction_fee_raw: int | None = None
    platform_fee_raw: int | None = None
    
    # Calculated during preparation
    token_amount_raw: int | None = None  # Expected tokens (before slippage)
    minimum_token_swap_amount_raw: int | None = None  # Min tokens with slippage


@dataclass
class SellOrder:
    """Sell order with sell-specific parameters."""
    
    # Identity
    token_info: TokenInfo  # Contains mint, symbol, platform
    
    # Input
    token_amount_raw: int  # Tokens to sell
    
    # Price information (calculated)
    token_price_sol: float | None = None
    
    # Transaction parameters (calculated)
    priority_fee: int | None = None
    compute_unit_limit: int | None = None
    
    # Transaction execution (populated after send)
    tx_signature: str | None = None
    
    # Fees (populated after execution)
    transaction_fee_raw: int | None = None
    platform_fee_raw: int | None = None
    
    # Calculated during preparation
    expected_sol_swap_amount_raw: int | None = None  # Expected SOL from sale
    minimum_sol_swap_amount_raw: int | None = None  # Min SOL with slippage