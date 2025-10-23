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
    
    def __init__(self, mint: str, initial_data: dict[str, Any] | None = None):
        """Initialize tracker for a token.
        
        Args:
            mint: Token mint address (used as identifier)
            initial_data: Optional initial reserves from token creation message
        """
        self.mint = mint
        
        # Initialize from creation data if provided
        if initial_data:
            self.virtual_sol_reserves = initial_data.get("vSolInBondingCurve")
            self.virtual_token_reserves = initial_data.get("vTokensInBondingCurve")
            self.last_update_timestamp = time.time()
            logger.debug(f"[{str(self.mint)[:8]}] Initialized tracker for {self.mint} with reserves: {self.virtual_sol_reserves} SOL, {self.virtual_token_reserves} tokens")
        else:
            self.virtual_sol_reserves = None
            self.virtual_token_reserves = None
            self.last_update_timestamp = None
            logger.debug(f"[{str(self.mint)[:8]}] Created tracker for {self.mint} (lazy initialization)")
    
    def apply_trade(self, trade_data: dict[str, Any]) -> None:
        """Update reserves from PumpPortal trade message.
        
        Args:
            trade_data: Trade message from PumpPortal containing updated reserves
        """
        # Extract updated reserves from trade message
        v_sol = trade_data.get("vSolInBondingCurve")
        v_tokens = trade_data.get("vTokensInBondingCurve")
        
        if v_sol is None or v_tokens is None:
            logger.warning(f"[{str(self.mint)[:8]}] Trade message missing reserve data for {self.mint}: {trade_data}")
            return
        
        # Update reserves
        self.virtual_sol_reserves = v_sol
        self.virtual_token_reserves = v_tokens
        self.last_update_timestamp = time.time()
        
        logger.info(f"[{str(self.mint)[:8]}] (Trades) Virtual token reserves: {v_tokens}, Virtual sol reserves: {v_sol}")
    
    def get_current_price(self) -> float:
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
        
        price = self.virtual_sol_reserves / self.virtual_token_reserves
        age = time.time() - self.last_update_timestamp
        logger.info(f"[{str(self.mint)[:8]}] Current price {price:.10f} SOL age: {age:.2f}s")
        return price
    
    def is_stale(self, max_age_seconds: float) -> bool:
        """Check if tracker data is stale or not initialized.
        
        Args:
            max_age_seconds: Maximum age in seconds before considered stale
            
        Returns:
            True if stale or not initialized
        """
        if not self.is_initialized():
            return True
        
        age = time.time() - self.last_update_timestamp
        return age > max_age_seconds
    
    def is_initialized(self) -> bool:
        """Check if tracker has received at least one update.
        
        Returns:
            True if initialized with reserve data
        """
        return (
            self.virtual_sol_reserves is not None and 
            self.virtual_token_reserves is not None and
            self.last_update_timestamp is not None
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
        result= (int(self.virtual_token_reserves*10**TOKEN_DECIMALS),int(self.virtual_sol_reserves*LAMPORTS_PER_SOL))    
        # return int(self.virtual_token_reserves),int(self.virtual_sol_reserves)    
        logger.info(f"[{str(self.mint)[:8]}] Reserves: {result}")
        return result

    def get_last_update_time(self) -> float | None:
        """Get timestamp of last update.
        
        Returns:
            Unix timestamp of last update or None if not initialized
        """
        return self.last_update_timestamp
