"""TokenTradeTracker for real-time price tracking via PumpPortal trade events."""

import time
from typing import Any

from core.pubkeys import LAMPORTS_PER_SOL, TOKEN_DECIMALS
from utils.logger import get_logger

logger = get_logger(__name__)


class TokenTradeTracker:
    """Stateful tracker for a single token's bonding curve state via PumpPortal trades.
    
    Tracks virtual reserves from PumpPortal trade messages and provides
    real-time price calculations without RPC delays.
    """

    def __init__(
        self, 
        mint: str,
        initial_virtual_sol_reserves: int,
        initial_virtual_token_reserves: int
    ):
        """Initialize tracker with reserves from token creation.
        
        Args:
            mint: Token mint address (used as identifier)
            initial_virtual_sol_reserves: Virtual SOL reserves in lamports (from vSolInBondingCurve)
            initial_virtual_token_reserves: Virtual token reserves in raw units (from vTokensInBondingCurve)
        """
        self.mint = mint
        self.virtual_sol_reserves = initial_virtual_sol_reserves
        self.virtual_token_reserves = initial_virtual_token_reserves
        self.last_update_timestamp = time.time()

        logger.debug(f"[{str(self.mint)[:8]}] __init()__ -> (Init) Virtual token reserves: {self.virtual_token_reserves}, Virtual sol reserves: {self.virtual_sol_reserves} => price={self.calculate_price():.10f} SOL")
    
    def apply_trade(self, trade_data: dict[str, Any]) -> None:
        """Update reserves from PumpPortal trade message.
        
        Args:
            trade_data: Trade message from PumpPortal containing updated reserves
        """
        # Extract updated reserves from trade message
        self.virtual_sol_reserves = int(trade_data.get("vSolInBondingCurve")*LAMPORTS_PER_SOL)
        self.virtual_token_reserves = int(trade_data.get("vTokensInBondingCurve")*10**TOKEN_DECIMALS)
        
        # Update reserves
        self.last_update_timestamp = time.time()
        
        logger.debug(f"[{str(self.mint)[:8]}] apply_trade() -> (Trades) Virtual token reserves: {self.virtual_token_reserves}, Virtual sol reserves: {self.virtual_sol_reserves}")
    
    def calculate_price(self) -> float:
        """Get current price from cached reserves.
        
        Returns:
            Current price in SOL per token
            
        Raises:
            RuntimeError: If tracker not initialized
        """
        if not self.is_initialized():
            raise RuntimeError(f"Tracker for {self.mint} not initialized")
        
        if self.virtual_token_reserves == 0:
            logger.warning(f"[{str(self.mint)[:8]}] Zero token reserves for {self.mint}, returning 0 price")
            return 0.0
        
        price_lamports = self.virtual_sol_reserves / self.virtual_token_reserves
        price = price_lamports * (10**TOKEN_DECIMALS) / LAMPORTS_PER_SOL
        age = time.time() - self.last_update_timestamp
        logger.debug(f"[{str(self.mint)[:8]}] calculate_price() -> current price {price:.10f} SOL. Last trade {age:.2f}s ago")
        return price
    
    def is_stale(self, max_age_seconds: float) -> bool:
        """Check if tracker data is stale or not initialized.
        
        Args:
            max_age_seconds: Maximum age in seconds before considered stale (unused)
            
        Returns:
            True if not initialized, False if initialized (reserves are always current via PumpPortal)
        """
        return not self.is_initialized()
    
    def is_initialized(self) -> bool:
        """Check if tracker has received at least one update.
        
        Returns:
            True if initialized with reserve data
        """
        return (
            self.virtual_sol_reserves is not None and 
            self.virtual_token_reserves is not None 
        )
    
    def get_mint(self) -> str:
        """Get token mint address.
        
        Returns:
            Token mint address
        """
        return self.mint
    
    
    def get_reserves(self) -> tuple[int | None, int | None]:
        """Get current reserves.
        
        Returns:
            Tuple of (virtual_sol_reserves, virtual_token_reserves) or (None, None) if not initialized
        """
        result= (self.virtual_token_reserves,self.virtual_sol_reserves)    
        logger.debug(f"[{str(self.mint)[:8]}] get_reserves() -> {result}")
        return result

    def get_last_update_time(self) -> float | None:
        """Get timestamp of last update.
        
        Returns:
            Unix timestamp of last update or None if not initialized
        """
        return self.last_update_timestamp
