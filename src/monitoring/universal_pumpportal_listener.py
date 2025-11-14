"""
Universal PumpPortal listener that works with multiple platforms.
"""

import asyncio
import json
from collections.abc import Awaitable, Callable
import time

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
        excluded_wallets: set[str] | None = None,
        database_manager=None,
    ):
        """Initialize universal PumpPortal listener.

        Args:
            pumpportal_url: PumpPortal WebSocket URL
            platforms: List of platforms to monitor (if None, monitor all supported platforms)
            excluded_wallets: Set of wallet addresses whose trades should be offset (default None/empty set)
            database_manager: Optional database manager for storing PumpPortal messages
        """
        super().__init__(excluded_wallets=excluded_wallets)
        self.database_manager = database_manager
        self.pumpportal_url = pumpportal_url
        self.ping_interval = 3  # seconds - faster connection loss detection
        
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
        async for websocket in websockets.connect(self.pumpportal_url):
            self._websocket = websocket
            logger.info("Connected to PumpPortal WebSocket")
            
            # Always resubscribe to new tokens
            await self._subscribe_to_new_tokens()
            
            # Resubscribe to all existing trade tracking
            await self._resubscribe_to_trades()
            
            # Start ping loop in background
            ping_task = asyncio.create_task(self._ping_loop())

            try:
                async for message in websocket:
                    timestamp = time.time()
                    try:
                        data = json.loads(message)
                        # Check if it's a token creation or trade message
                        if data.get("txType") == "create":
                            token_info = await self._process_token_creation(data, match_string, creator_address)
                            if token_info:
                                logger.debug("Token creation message processed")
                                await token_callback(token_info)
                            else:
                                logger.debug(f"Token creation message not processed: {data}")
                        elif data.get("txType") in ["buy", "sell"]:
                            # Process trade message
                            self._handle_trade_message(timestamp, data)
                        elif data.get("message"):
                            # Handle acknowledgement messages
                            logger.debug(f"Received acknowledgement: {data['message']}")
                            continue
                        else:
                            logger.debug(f"Unknown message type: {data}")
                            
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse message as JSON: {message}")
                        continue
                    except Exception:
                        logger.exception("Error processing PumpPortal WebSocket message")
                        continue

            except websockets.exceptions.ConnectionClosed:
                logger.warning("PumpPortal WebSocket connection closed. Reconnecting...")
            finally:
                ping_task.cancel()
                try:
                    await ping_task
                except asyncio.CancelledError:
                    pass

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
                    await asyncio.wait_for(pong_waiter, timeout=2)
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
            logger.info(f"Ignoring token from unsupported pool: {pool_name}")
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
        if not self._websocket:
            logger.warning(f"WebSocket not available, cannot subscribe to {mint}")
            return
            
        subscribe_msg = {
            "method": "subscribeTokenTrade",
            "keys": [mint]
        }

        try:
            await self._websocket.send(json.dumps(subscribe_msg))
            logger.info(f"Sent trade subscription for mint: {mint}")
        except websockets.exceptions.ConnectionClosed:
            logger.warning(f"WebSocket closed while subscribing to {mint}")
        except Exception as e:
            logger.exception(f"Failed to send trade subscription for {mint}: {e}")
    
    async def _send_trade_unsubscription(self, mint: str) -> None:
        """Send WebSocket unsubscription for token trades.
        
        Args:
            mint: Token mint address to unsubscribe from
        """
        if not self._websocket:
            logger.debug(f"WebSocket not available, cannot unsubscribe from mint: {mint}")
            return
            
        try:
            unsubscribe_msg = {
                "method": "unsubscribeTokenTrade",
                "keys": [mint]
            }
            await self._websocket.send(json.dumps(unsubscribe_msg))
            logger.debug(f"Sent trade unsubscription for mint: {mint}")
        except websockets.exceptions.ConnectionClosed:
            logger.debug(f"WebSocket closed while unsubscribing from {mint}")
        except Exception as e:
            logger.exception(f"Failed to send trade unsubscription for {mint}: {e}")

        self._subscribed_mints.remove(mint)

    def _handle_trade_message(self, timestamp: float, trade_data: dict) -> None:
        """Process trade message, write to database, and update tracker.
        
        Args:
            timestamp: Unix timestamp when the trade occurred
            trade_data: Trade message from PumpPortal
        """
        # Call parent method first to maintain existing trade tracking functionality
        super()._handle_trade_message(timestamp, trade_data)

#        logger.info(f"In _handle_trade_message() for {trade_data}")

        # Write to database if database_manager is available
        if not self.database_manager:
            return
        
        # Create async task to write to database (non-blocking)
        try:
            asyncio.create_task(
                self._write_pumpportal_message_to_db(timestamp, trade_data)
            )
        except Exception as e:
            logger.exception(f"Failed to create task for writing PumpPortal message to database: {e}")
    
    async def _write_pumpportal_message_to_db(self, timestamp: float, trade_data: dict) -> None:
        """Write PumpPortal message to database.
        
        Args:
            timestamp: Unix timestamp when the trade occurred
            trade_data: Trade message from PumpPortal
        """
        try:
            # Extract required fields from trade message
            mint = trade_data.get("mint")
            if not mint:
                return
            
            tx_type = trade_data.get("txType", "")
            if tx_type not in ["buy", "sell", "create"]:
                return
            
            # Extract reserves and swap amounts
            v_sol = trade_data.get("vSolInBondingCurve")
            v_tokens = trade_data.get("vTokensInBondingCurve")
            sol_amount = trade_data.get("solAmount")
            token_amount = trade_data.get("tokenAmount")
            
            # Extract pool, trader public key, and signature directly from message
            pool = trade_data.get("pool")
            trader_public_key = trade_data.get("traderPublicKey")
            signature = trade_data.get("signature")
            
            # Determine platform from pool name (still needed for queries)
            platform = pool.lower() if pool else ""
            
            if v_sol is None or v_tokens is None:
                logger.warning(f"Missing reserves in trade message for {mint}, skipping database write")
                return
                        
            # Calculate price from reserves: virtual_sol_reserves / (virtual_token_reserves / 10^TOKEN_DECIMALS)
            # This gives SOL per token
            if v_tokens > 0:
                price_reserves_decimal = v_sol / v_tokens
            else:
                logger.debug(f"Zero token reserves for {mint}, skipping database write")
                return
            
            # Calculate price from swap amounts: sol_amount_swapped / (token_amount_swapped / 10^TOKEN_DECIMALS)
            price_swap_decimal = None
            if sol_amount is not None and token_amount is not None and token_amount > 0:
                price_swap_decimal = sol_amount  / token_amount
            
            # Convert timestamp to milliseconds
            timestamp_ms = int(timestamp * 1000)
            
            # Write to database
            await self.database_manager.insert_pumpportal_message(
                mint=mint,
                platform=platform,
                timestamp=timestamp_ms,
                message_type=tx_type,
                virtual_sol_reserves=v_sol,
                virtual_token_reserves=v_tokens,
                sol_amount_swapped=sol_amount,
                token_amount_swapped=token_amount,
                price_reserves_decimal=price_reserves_decimal,
                price_swap_decimal=price_swap_decimal,
                pool=pool,
                trader_public_key=trader_public_key,
                signature=signature,
            )
        except Exception as e:
            logger.exception(f"Failed to write PumpPortal message to database: {e}")

    
