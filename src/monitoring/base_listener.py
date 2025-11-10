"""
Base class for WebSocket token listeners - now platform-agnostic.
"""

from abc import ABC, abstractmethod
import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

from interfaces.core import Platform, TokenInfo
from trading.token_trade_tracker import TokenTradeTracker
from utils.logger import get_logger

logger = get_logger(__name__)


class BaseTokenListener(ABC):
    """Base abstract class for token listeners - now platform-agnostic."""

    def __init__(self, platform: Platform | None = None):
        """Initialize the listener with optional platform specification.

        Args:
            platform: Platform to monitor (if None, monitor all platforms)
        """
        self.platform = platform
        
        # Trade tracking state - index by mint
        self._trade_trackers: dict[str, TokenTradeTracker] = {}  # mint -> tracker

    @abstractmethod
    async def listen_for_messages(
        self,
        token_callback: Callable[[TokenInfo], Awaitable[None]],
        match_string: str | None = None,
        creator_address: str | None = None,
    ) -> None:
        """
        Listen for new token creations and trade messages.

        Args:
            token_callback: Callback function for new tokens
            match_string: Optional string to match in token name/symbol
            creator_address: Optional creator address to filter by
        """
        pass

    def should_process_token(self, token_info: TokenInfo) -> bool:
        """Check if a token should be processed based on platform filter.

        Args:
            token_info: Token information

        Returns:
            True if token should be processed
        """
        if self.platform is None:
            return True  # Process all platforms
        return token_info.platform == self.platform
    
    async def subscribe_token_trades(self, token_info: TokenInfo) -> None:
        """Subscribe to trade tracking for a token.
        
        Args:
            token_info: Token information containing reserves and creator data
        """
        # Create tracker with token information from creation
        tracker = TokenTradeTracker(token_info)
        self._trade_trackers[str(token_info.mint)] = tracker  # Index by mint
        
        # Subclasses should override to send WebSocket subscription
        logger.debug(f"Calling _send_trade_subscription() for {token_info.mint}")
        await self._send_trade_subscription(str(token_info.mint))
    
    async def unsubscribe_token_trades(self, mint: str) -> None:
        """Unsubscribe from trade tracking for a token.
        
        Args:
            mint: Token mint address
        """
        if mint in self._trade_trackers:
            # Remove from mappings
            del self._trade_trackers[mint]
            
            # Send unsubscribe message to WebSocket
            await self._send_trade_unsubscription(mint)
    
    
    def get_trade_tracker_by_mint(self, mint: str) -> TokenTradeTracker | None:
        """Get trade tracker by mint address.
        
        Args:
            mint: Token mint address
            
        Returns:
            TokenTradeTracker if found, None otherwise
        """
        return self._trade_trackers.get(mint)
    
    def _handle_trade_message(self, timestamp: float, trade_data: dict[str, Any]) -> None:
        """Process trade message and update tracker.
        
        Args:
            trade_data: Trade message from PumpPortal (or other source)
        """
        mint = trade_data.get("mint")
        if not mint:
            return
        
        tracker = self._trade_trackers.get(mint)
        if tracker:
            tracker.apply_trade(trade_data, timestamp)
    
    async def _send_trade_subscription(self, mint: str) -> None:
        """Send WebSocket subscription for token trades.
        
        Subclasses should override to implement platform-specific subscription.
        
        Args:
            mint: Token mint address to subscribe to
        """
        # Default implementation - subclasses should override
        pass
    
    async def _send_trade_unsubscription(self, mint: str) -> None:
        """Send WebSocket unsubscription for token trades.
        
        Subclasses should override to implement platform-specific unsubscription.
        
        Args:
            mint: Token mint address to unsubscribe from
        """
        # Default implementation - subclasses should override
        pass
