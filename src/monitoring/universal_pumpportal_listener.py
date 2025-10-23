"""
Universal PumpPortal listener that works with multiple platforms.
"""

import asyncio
import json
from collections.abc import Awaitable, Callable

import websockets

from interfaces.core import Platform, TokenInfo
from monitoring.base_listener import BaseTokenListener
from utils.logger import get_logger

logger = get_logger(__name__)


class UniversalPumpPortalListener(BaseTokenListener):
    """Universal PumpPortal listener that works with multiple platforms."""

    def __init__(
        self,
        pumpportal_url: str = "wss://pumpportal.fun/api/data",
        platforms: list[Platform] | None = None,
    ):
        """Initialize universal PumpPortal listener.

        Args:
            pumpportal_url: PumpPortal WebSocket URL
            platforms: List of platforms to monitor (if None, monitor all supported platforms)
        """
        super().__init__()
        self.pumpportal_url = pumpportal_url
        self.ping_interval = 20  # seconds
        
        # Track subscribed mints for reconnection
        self._subscribed_mints: set[str] = set()
        
        # Store websocket connection for immediate trade subscriptions
        self._websocket = None

        # Get platform-specific processors
        from platforms.letsbonk.pumpportal_processor import LetsBonkPumpPortalProcessor
        from platforms.pumpfun.pumpportal_processor import PumpFunPumpPortalProcessor

        # Create processor instances
        all_processors = [
            PumpFunPumpPortalProcessor(),
            LetsBonkPumpPortalProcessor(),
        ]

        # Filter processors based on requested platforms
        if platforms is None:
            self.processors = all_processors
        else:
            self.processors = [p for p in all_processors if p.platform in platforms]

        # Build mapping of pool names to processors for quick lookup
        self.pool_to_processors: dict[str, list] = {}
        for processor in self.processors:
            for pool_name in processor.supported_pool_names:
                if pool_name not in self.pool_to_processors:
                    self.pool_to_processors[pool_name] = []
                self.pool_to_processors[pool_name].append(processor)

        logger.info(
            f"Initialized Universal PumpPortal listener for platforms: {[p.platform.value for p in self.processors]}"
        )
        logger.info(f"Monitoring pools: {list(self.pool_to_processors.keys())}")

    async def listen_for_messages(
        self,
        token_callback: Callable[[TokenInfo], Awaitable[None]],
        match_string: str | None = None,
        creator_address: str | None = None,
    ) -> None:
        """Listen for new token creations and trade messages using PumpPortal WebSocket.

        Args:
            token_callback: Callback function for new tokens
            match_string: Optional string to match in token name/symbol
            creator_address: Optional creator address to filter by
        """
        while True:
            try:
                async with websockets.connect(self.pumpportal_url) as self._websocket:
                    await self._subscribe_to_new_tokens()
                    # Resubscribe to any previously subscribed bonding curves
                    await self._resubscribe_to_trades()
                    ping_task = asyncio.create_task(self._ping_loop())

                    try:
                        while True:
                            message = await self._wait_for_message()
                            if not message:
                                continue

                            # Check if it's a token creation or trade message
                            if message.get("txType") == "create":
                                token_info = await self._process_token_creation(message, match_string, creator_address)
                                if token_info:
                                    await token_callback(token_info)
                            elif message.get("txType") in ["buy", "sell"]:
                                # Process trade message
                                self._handle_trade_message(message)

                    except websockets.exceptions.ConnectionClosed:
                        logger.warning(
                            "PumpPortal WebSocket connection closed. Reconnecting..."
                        )
                        asyncio.sleep(0.5)
                    finally:
                        ping_task.cancel()
                        try:
                            await ping_task
                        except asyncio.CancelledError:
                            pass

            except Exception:
                logger.exception("PumpPortal WebSocket connection error")
                logger.info("Reconnecting in 5 seconds...")
                await asyncio.sleep(5)

    async def _subscribe_to_new_tokens(self) -> None:
        """Subscribe to new token events from PumpPortal."""
        subscription_message = json.dumps({"method": "subscribeNewToken", "params": []})

        await self._websocket.send(subscription_message)
        logger.info("Subscribed to PumpPortal new token events")

    async def _ping_loop(self) -> None:
        """Keep connection alive with pings."""
        try:
            while True:
                await asyncio.sleep(self.ping_interval)
                try:
                    pong_waiter = await self._websocket.ping()
                    await asyncio.wait_for(pong_waiter, timeout=10)
                except TimeoutError:
                    logger.warning("Ping timeout - PumpPortal server not responding")
                    # Force reconnection
                    await self._websocket.close()
                    return
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("Ping error")

    
    async def _resubscribe_to_trades(self) -> None:
        """Resubscribe to trade tracking for all previously subscribed mints."""
        for mint in self._subscribed_mints:
            await self._send_actual_trade_subscription(mint)
    
    async def _process_token_creation(
        self, 
        message: dict, 
        match_string: str | None, 
        creator_address: str | None
    ) -> TokenInfo | None:
        """Process token creation message and apply filters.
        
        Args:
            message: Token creation message
            match_string: Optional string to match in token name/symbol
            creator_address: Optional creator address to filter by
            
        Returns:
            TokenInfo if token passes filters, None otherwise
        """
        # Get pool name to determine which processor to use
        pool_name = message.get("pool", "").lower()
        if pool_name not in self.pool_to_processors:
            logger.debug(f"Ignoring token from unsupported pool: {pool_name}")
            return None

        # Try each processor that supports this pool
        for processor in self.pool_to_processors[pool_name]:
            if processor.can_process(message):
                token_info = processor.process_token_data(message)
                if token_info:
                    logger.debug(
                        f"Successfully processed token using {processor.platform.value} processor"
                    )
                    
                    # Apply filters
                    if match_string and not (
                        match_string.lower() in token_info.name.lower()
                        or match_string.lower() in token_info.symbol.lower()
                    ):
                        logger.warning(
                            f"Token does not match filter '{match_string}'. Skipping..."
                        )
                        return None

                    if creator_address:
                        creator_str = (
                            str(token_info.creator)
                            if token_info.creator
                            else ""
                        )
                        user_str = (
                            str(token_info.user) if token_info.user else ""
                        )
                        if creator_address not in [creator_str, user_str]:
                            logger.info(
                                f"Token not created by {creator_address}. Skipping..."
                            )
                            return None
                    
                    logger.debug(
                        f"New token detected: {token_info.name} ({token_info.symbol}) on {token_info.platform.value}"
                    )
                    return token_info

        logger.debug(f"No processor could handle token data from pool {pool_name}")
        return None
    
    async def _wait_for_message(self) -> dict | None:
        """Wait for any message from PumpPortal WebSocket.
        
        Returns:
            Message dict if received, None otherwise
        """
        try:
            response = await asyncio.wait_for(self._websocket.recv(), timeout=30)
            data = json.loads(response)
            
            # Log all received messages for debugging
            logger.debug(f"Raw WebSocket message: {response}")
            logger.debug(f"Parsed message: {json.dumps(data, indent=2)}")

            # Handle different message formats from PumpPortal
            message = None
            if "method" in data and data["method"] == "newToken":
                # Standard newToken method format
                params = data.get("params", [])
                if params and len(params) > 0:
                    message = params[0]
            elif "signature" in data and "mint" in data and "pool" in data:
                # Direct message format
                message = data
            elif "message" in data:
                # Handle subscription acknowledgements
                return None  # Don't process acknowledgements as token messages

            return message

        except TimeoutError:
            logger.debug("No data received from PumpPortal for 30 seconds")
        except websockets.exceptions.ConnectionClosed:
            logger.warning("PumpPortal WebSocket connection closed")
            raise
        except json.JSONDecodeError:
            logger.exception("Failed to decode PumpPortal message")
        except Exception:
            logger.exception("Error processing PumpPortal WebSocket message")

        return None
    
    async def _send_trade_subscription(self, mint: str) -> None:
        """Send WebSocket subscription for token trades.
        
        Args:
            mint: Token mint address to subscribe to
        """
        logger.debug(f"In _send_trade_subscription() for {mint}")
        
        # Store for reconnection tracking
        self._subscribed_mints.add(mint)
        
        # Send immediately if websocket is available
        if self._websocket:
            await self._send_actual_trade_subscription(mint)
        else:
            logger.warn(f"WebSocket not available, will resubscribe on reconnection for mint: {mint}")
    
    async def _send_actual_trade_subscription(self, mint: str) -> None:
        """Actually send the WebSocket subscription for token trades.
        
        Args:
            mint: Token mint address to subscribe to
        """
        subscribe_msg = {
            "method": "subscribeTokenTrade",
            "keys": [mint]
        }

        await self._websocket.send(json.dumps(subscribe_msg))
        logger.debug(f"Sent trade subscription for mint: {mint} -> {json.dumps(subscribe_msg, indent=2)}")
    
    async def _send_trade_unsubscription(self, mint: str) -> None:
        """Send WebSocket unsubscription for token trades.
        
        Args:
            mint: Token mint address to unsubscribe from
        """
        if self._websocket:
            unsubscribe_msg = {
                "method": "unsubscribeTokenTrade",
                "keys": [mint]
            }
            await self._websocket.send(json.dumps(unsubscribe_msg))
            logger.debug(f"Sent trade unsubscription for mint: {mint}")
        else:
            logger.debug(f"WebSocket not available, cannot unsubscribe from mint: {mint}")
    
    
