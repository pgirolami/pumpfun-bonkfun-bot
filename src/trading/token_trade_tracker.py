"""TokenTradeTracker for real-time price tracking via PumpPortal trade events."""

import time
from typing import Any

from core.pubkeys import LAMPORTS_PER_SOL, TOKEN_DECIMALS
from interfaces.core import TokenInfo
from solders.pubkey import Pubkey
from utils.logger import get_logger


logger = get_logger(__name__)


class TokenTradeTracker:
    """Stateful tracker for a single token's bonding curve state via PumpPortal trades.
    
    Tracks virtual reserves from PumpPortal trade messages and provides
    real-time price calculations without RPC delays.
    """

    def __init__(self, token_info: TokenInfo):
        """Initialize tracker with token information from creation.
        
        Args:
            token_info: Token information containing reserves, creator, and initial buy data
        """
        self.mint = token_info.mint
        self.virtual_sol_reserves = token_info.virtual_sol_reserves
        self.virtual_token_reserves = token_info.virtual_token_reserves
        self.creator_public_key = str(token_info.creator)
        self.last_update_timestamp = time.time()
        
        # Creator tracking (raw token units)
        # Use the initial buy amount from token creation
        initial_buy_decimal = token_info.initial_buy_token_amount_decimal or 0.0
        self.creator_token_swap_in = int(initial_buy_decimal * 10**TOKEN_DECIMALS)
        self.creator_token_swap_out = 0    # Tokens sold by creator (negative)
 
        logger.debug(f"[{str(self.mint)[:8]}] __init()__ -> (Init) Virtual token reserves: {self.virtual_token_reserves}, Virtual sol reserves: {self.virtual_sol_reserves} => price={self.calculate_price():.10f} SOL")
    
    def apply_trade(self, trade_data: dict[str, Any]) -> None:
        """Update reserves from PumpPortal trade message.
        
        Args:
            trade_data: Trade message from PumpPortal containing updated reserves
        """
        # Extract updated reserves from trade message
        self.virtual_sol_reserves = int(trade_data.get("vSolInBondingCurve")*LAMPORTS_PER_SOL)
        self.virtual_token_reserves = int(trade_data.get("vTokensInBondingCurve")*10**TOKEN_DECIMALS)
        
        # Track creator trades
        trader_public_key = str(trade_data.get("traderPublicKey", ""))
        # logger.info(f"[{str(self.mint)[:8]}] apply_trade() -> trader_public_key= {trader_public_key} vs self.creator_public_key= {self.creator_public_key}")
        if trader_public_key == self.creator_public_key:
            token_amount_decimal = trade_data.get("tokenAmount", 0.0)
            token_amount_raw = int(token_amount_decimal * 10**TOKEN_DECIMALS)
            tx_type = trade_data.get("txType")
            
            if tx_type == "buy" or tx_type == "create":
                self.creator_token_swap_in += token_amount_raw
                logger.info(f"[{str(self.mint)[:8]}] Creator [{self.creator_public_key[:8]}] buy: +{token_amount_raw} tokens (swap_in: {self.creator_token_swap_in}, swap_out: {self.creator_token_swap_out})")
            elif tx_type == "sell":
                self.creator_token_swap_out -= token_amount_raw
                logger.info(f"[{str(self.mint)[:8]}] Creator [{self.creator_public_key[:8]}] sell: +{token_amount_raw} tokens (swap_in: {self.creator_token_swap_in}, swap_out: {self.creator_token_swap_out})")
            else:
                logger.warning(f"[{str(self.mint)[:8]}] Creator [{self.creator_public_key[:8]}] unknown trade type: {tx_type}")

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
            raise RuntimeError(f"Tracker for {str(self.mint)} not initialized")
        
        if self.virtual_token_reserves == 0:
            logger.warning(f"[{str(self.mint)[:8]}] Zero token reserves, returning 0 price")
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
    
    def get_mint(self) -> Pubkey:
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
    
    def get_creator_swaps(self) -> tuple[int, int]:
        """Get creator's token swap amounts.
        
        Returns:
            Tuple of (swap_in, swap_out) amounts in raw token units
        """
        return (self.creator_token_swap_in, self.creator_token_swap_out)
    
    def get_creator_net_position(self) -> int:
        """Get creator's net token position.
        
        Returns:
            Net tokens held by creator (positive = holding, negative = short)
        """
        return self.creator_token_swap_in + self.creator_token_swap_out
