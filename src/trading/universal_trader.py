"""
Universal trading coordinator that works with any platform.
Cleaned up to remove all platform-specific hardcoding.
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path
from time import monotonic, time

import uvloop
from solders.pubkey import Pubkey

from cleanup.modes import (
    handle_cleanup_after_failure,
    handle_cleanup_after_sell,
    handle_cleanup_post_session,
)
from core.client import SolanaClient
from core.priority_fee.manager import PriorityFeeManager
from core.wallet import Wallet
from interfaces.core import Platform, TokenInfo
from monitoring.listener_factory import ListenerFactory
from platforms import get_platform_implementations
from trading.base import TradeResult
from trading.platform_aware import PlatformAwareBuyer, PlatformAwareSeller
from trading.position import ExitReason, Position
from utils.logger import get_logger
from database.models import PositionConverter

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

logger = get_logger(__name__)


class UniversalTrader:
    """Universal trading coordinator that works with any supported platform."""

    def __init__(
        self,
        rpc_endpoint: str,
        wss_endpoint: str,
        wallet: Wallet,
        buy_amount: float,
        buy_slippage: float,
        sell_slippage: float,
        # Platform configuration
        platform: Platform | str = Platform.PUMP_FUN,
        # Listener configuration
        listener_type: str = "logs",
        geyser_endpoint: str | None = None,
        geyser_api_token: str | None = None,
        geyser_auth_type: str = "x-token",
        pumpportal_url: str = "wss://pumpportal.fun/api/data",
        # Trading configuration
        extreme_fast_mode: bool = False,
        extreme_fast_token_amount: int = 30,
        # Exit strategy configuration
        exit_strategy: str = "time_based",
        take_profit_percentage: float | None = None,
        stop_loss_percentage: float | None = None,
        max_hold_time: int | None = None,
        price_check_interval: int = 10,
        # Priority fee configuration
        enable_dynamic_priority_fee: bool = False,
        enable_fixed_priority_fee: bool = True,
        fixed_priority_fee: int = 200_000,
        extra_priority_fee: float = 0.0,
        hard_cap_prior_fee: int = 200_000,
        # Retry and timeout settings
        max_retries: int = 3,
        wait_time_after_creation: int = 15,
        wait_time_after_buy: int = 15,
        wait_time_before_new_token: int = 15,
        max_token_age: int | float = 0.001,
        token_wait_timeout: int = 30,
        # Cleanup settings
        cleanup_mode: str = "disabled",
        cleanup_force_close_with_burn: bool = False,
        cleanup_with_priority_fee: bool = False,
        # Trading filters
        match_string: str | None = None,
        bro_address: str | None = None,
        marry_mode: bool = False,
        yolo_mode: bool = False,
        # Compute unit configuration
        compute_units: dict | None = None,
        # Testing configuration
        testing: dict | None = None,
        # Database configuration
        database_manager = None,
    ):
        """Initialize the universal trader."""
        # Core components
        self.solana_client = SolanaClient(rpc_endpoint)
        self.wallet = wallet
        self.priority_fee_manager = PriorityFeeManager(
            client=self.solana_client,
            enable_dynamic_fee=enable_dynamic_priority_fee,
            enable_fixed_fee=enable_fixed_priority_fee,
            fixed_fee=fixed_priority_fee,
            extra_fee=extra_priority_fee,
            hard_cap=hard_cap_prior_fee,
        )
        
        # Database manager
        self.database_manager = database_manager
        
        # Generate run ID for this trader instance
        self.run_id = str(int(time() * 1000))
        
        # Trading parameters (needed for validation)
        self.buy_amount = buy_amount

        # Platform setup
        if isinstance(platform, str):
            self.platform = Platform(platform)
        else:
            self.platform = platform

        logger.info(f"Initialized Universal Trader for platform: {self.platform.value} with run_id={self.run_id}")

        # Validate platform support
        try:
            from platforms import platform_factory

            if not platform_factory.registry.is_platform_supported(self.platform):
                raise ValueError(f"Platform {self.platform.value} is not supported")
        except Exception:
            logger.exception("Platform validation failed")
            raise

        # Get platform-specific implementations
        self.platform_implementations = get_platform_implementations(
            self.platform, self.solana_client
        )

        # Store compute unit configuration
        self.compute_units = compute_units or {}

        # Extract testing configuration
        dry_run = False
        dry_run_wait_time = 0.5
        if testing:
            dry_run = testing.get('dry_run', False)
            dry_run_wait_time = testing.get('dry_run_wait_time_seconds', 0.5)

        # Create platform-aware traders based on mode
        if dry_run:
            from trading.dry_run_platform_aware import (
                DryRunPlatformAwareBuyer,
                DryRunPlatformAwareSeller,
            )
            logger.info("Initializing DRY-RUN mode traders")
            
            self.buyer = DryRunPlatformAwareBuyer(
                self.solana_client,
                self.wallet,
                self.priority_fee_manager,
                buy_amount,
                buy_slippage,
                max_retries,
                extreme_fast_token_amount,
                extreme_fast_mode,
                dry_run_wait_time=dry_run_wait_time,
                compute_units=self.compute_units,
                database_manager=self.database_manager,
            )
            
            self.seller = DryRunPlatformAwareSeller(
                self.solana_client,
                self.wallet,
                self.priority_fee_manager,
                sell_slippage,
                max_retries,
                dry_run_wait_time=dry_run_wait_time,
                compute_units=self.compute_units,
                database_manager=self.database_manager,
            )
        else:
            if self.buy_amount>0.01:
                raise ValueError("Buy amount must NOT be greater than 0.01 SOL in live mode")

            # Create platform-aware traders
            self.buyer = PlatformAwareBuyer(
                self.solana_client,
                self.wallet,
                self.priority_fee_manager,
                buy_amount,
                buy_slippage,
                max_retries,
                extreme_fast_token_amount,
                extreme_fast_mode,
                compute_units=self.compute_units,
            )

            self.seller = PlatformAwareSeller(
                self.solana_client,
                self.wallet,
                self.priority_fee_manager,
                sell_slippage,
                max_retries,
                compute_units=self.compute_units,
            )

        # Initialize the appropriate listener with platform filtering
        self.token_listener = ListenerFactory.create_listener(
            listener_type=listener_type,
            wss_endpoint=wss_endpoint,
            geyser_endpoint=geyser_endpoint,
            geyser_api_token=geyser_api_token,
            geyser_auth_type=geyser_auth_type,
            pumpportal_url=pumpportal_url,
            platforms=[self.platform],  # Only listen for our platform
        )

        # Trading parameters
        self.buy_slippage = buy_slippage
        self.sell_slippage = sell_slippage
        self.max_retries = max_retries
        self.extreme_fast_mode = extreme_fast_mode
        self.extreme_fast_token_amount = extreme_fast_token_amount

        # Exit strategy parameters
        self.exit_strategy = exit_strategy.lower()
        self.take_profit_percentage = take_profit_percentage
        self.stop_loss_percentage = stop_loss_percentage
        self.trailing_stop_percentage: float | None = None
        if self.exit_strategy == "trailing":
            # Use stop_loss_percentage input as trailing percentage if provided
            self.trailing_stop_percentage = stop_loss_percentage
        self.max_hold_time = max_hold_time
        self.price_check_interval = price_check_interval

        # Timing parameters
        self.wait_time_after_creation = wait_time_after_creation
        self.wait_time_after_buy = wait_time_after_buy
        self.wait_time_before_new_token = wait_time_before_new_token
        self.max_token_age = max_token_age
        self.token_wait_timeout = token_wait_timeout

        # Cleanup parameters
        self.cleanup_mode = cleanup_mode
        self.cleanup_force_close_with_burn = cleanup_force_close_with_burn
        self.cleanup_with_priority_fee = cleanup_with_priority_fee

        # Trading filters/modes
        self.match_string = match_string
        self.bro_address = bro_address
        self.marry_mode = marry_mode
        self.yolo_mode = yolo_mode

        # State tracking
        self.traded_mints: set[Pubkey] = set()
        self.token_queue: asyncio.Queue = asyncio.Queue()
        self.processing: bool = False
        self.processed_tokens: set[str] = set()
        self.token_timestamps: dict[str, float] = {}

    async def start(self) -> None:
        """Start the trading bot and listen for new tokens."""
        logger.info(f"Starting Universal Trader for {self.platform.value}")
        logger.info(
            f"Match filter: {self.match_string if self.match_string else 'None'}"
        )
        logger.info(
            f"Creator filter: {self.bro_address if self.bro_address else 'None'}"
        )
        logger.info(f"Marry mode: {self.marry_mode}")
        logger.info(f"YOLO mode: {self.yolo_mode}")
        logger.info(f"Exit strategy: {self.exit_strategy}")

        if self.exit_strategy == "tp_sl":
            logger.info(
                f"Take profit: {self.take_profit_percentage * 100 if self.take_profit_percentage else 'None'}%"
            )
            logger.info(
                f"Stop loss: {self.stop_loss_percentage * 100 if self.stop_loss_percentage else 'None'}%"
            )
            logger.info(
                f"Max hold time: {self.max_hold_time if self.max_hold_time else 'None'} seconds"
            )

        logger.info(f"Max token age: {self.max_token_age} seconds")

        try:
            health_resp = await self.solana_client.get_health()
            logger.info(f"RPC warm-up successful (getHealth passed: {health_resp})")
        except Exception as e:
            logger.warning(f"RPC warm-up failed: {e!s}")

        try:
            # Choose operating mode based on yolo_mode
            if not self.yolo_mode:
                # Single token mode: process one token and exit
                logger.info(
                    "Running in single token mode - will process one token and exit"
                )
                token_info = await self._wait_for_token()
                if token_info:
                    await self._handle_token(token_info)
                    logger.info("Finished processing single token. Exiting...")
                else:
                    logger.info(
                        f"No suitable token found within timeout period ({self.token_wait_timeout}s). Exiting..."
                    )
            else:
                # Continuous mode: process tokens until interrupted
                logger.info(
                    "Running in continuous mode - will process tokens until interrupted"
                )
                processor_task = asyncio.create_task(self._process_token_queue())

                try:
                    await self.token_listener.listen_for_tokens(
                        lambda token: self._queue_token(token),
                        self.match_string,
                        self.bro_address,
                    )
                except Exception:
                    logger.exception("Token listening stopped due to error")
                finally:
                    processor_task.cancel()
                    try:
                        await processor_task
                    except asyncio.CancelledError:
                        pass

        except Exception:
            logger.exception("Trading stopped due to error")

        finally:
            await self._cleanup_resources()
            logger.info("Universal Trader has shut down")

    async def _wait_for_token(self) -> TokenInfo | None:
        """Wait for a single token to be detected."""
        # Create a one-time event to signal when a token is found
        token_found = asyncio.Event()
        found_token = None

        async def token_callback(token: TokenInfo) -> None:
            nonlocal found_token
            token_key = str(token.mint)

            # Only process if not already processed and fresh
            if token_key not in self.processed_tokens:
                # Record when the token was discovered
                self.token_timestamps[token_key] = monotonic()
                found_token = token
                self.processed_tokens.add(token_key)
                token_found.set()

        listener_task = asyncio.create_task(
            self.token_listener.listen_for_tokens(
                token_callback,
                self.match_string,
                self.bro_address,
            )
        )

        # Wait for a token with a timeout
        try:
            logger.info(
                f"Waiting for a suitable token (timeout: {self.token_wait_timeout}s)..."
            )
            await asyncio.wait_for(token_found.wait(), timeout=self.token_wait_timeout)
            logger.info(f"Found token: {found_token.symbol} ({found_token.mint})")
            return found_token
        except TimeoutError:
            logger.info(
                f"Timed out after waiting {self.token_wait_timeout}s for a token"
            )
            return None
        finally:
            listener_task.cancel()
            try:
                await listener_task
            except asyncio.CancelledError:
                pass

    async def _cleanup_resources(self) -> None:
        """Perform cleanup operations before shutting down."""
        if self.traded_mints:
            try:
                logger.info(f"Cleaning up {len(self.traded_mints)} traded token(s)...")
                await handle_cleanup_post_session(
                    self.solana_client,
                    self.wallet,
                    list(self.traded_mints),
                    self.priority_fee_manager,
                    self.cleanup_mode,
                    self.cleanup_with_priority_fee,
                    self.cleanup_force_close_with_burn,
                )
            except Exception:
                logger.exception("Error during cleanup")

        old_keys = {k for k in self.token_timestamps if k not in self.processed_tokens}
        for key in old_keys:
            self.token_timestamps.pop(key, None)

        await self.solana_client.close()

    async def _queue_token(self, token_info: TokenInfo) -> None:
        """Queue a token for processing if not already processed."""
        token_key = str(token_info.mint)

        if token_key in self.processed_tokens:
            logger.debug(f"Token {token_info.symbol} already processed. Skipping...")
            return

        # Record timestamp when token was discovered
        self.token_timestamps[token_key] = monotonic()

        await self.token_queue.put(token_info)
        logger.info(
            f"Queued new token: {token_info.symbol} ({token_info.mint}) on {token_info.platform.value}"
        )

    async def _process_token_queue(self) -> None:
        """Continuously process tokens from the queue, only if they're fresh."""
        while True:
            try:
                token_info = await self.token_queue.get()
                token_key = str(token_info.mint)

                # Check if token is still "fresh"
                current_time = monotonic()
                token_age = current_time - self.token_timestamps.get(
                    token_key, current_time
                )

                if token_age > self.max_token_age:
                    logger.info(
                        f"Skipping token {token_info.symbol} - too old ({token_age:.1f}s > {self.max_token_age}s)"
                    )
                    continue

                self.processed_tokens.add(token_key)

                logger.info(
                    f"Processing fresh token: {token_info.symbol} (age: {token_age:.1f}s)"
                )
                await self._handle_token(token_info)

            except asyncio.CancelledError:
                logger.info("Token queue processor was cancelled")
                break
            except Exception:
                logger.exception("Error in token queue processor")
            finally:
                self.token_queue.task_done()

    async def _handle_token(self, token_info: TokenInfo) -> None:
        """Handle a new token creation event."""
        try:
            # Validate that token is for our platform
            if token_info.platform != self.platform:
                logger.warning(
                    f"Token platform mismatch: expected {self.platform.value}, got {token_info.platform.value}"
                )
                return

            # Wait for pool/curve to stabilize (unless in extreme fast mode)
            if not self.extreme_fast_mode:
                logger.info(
                    f"Waiting for {self.wait_time_after_creation} seconds for the pool/curve to stabilize..."
                )
                await asyncio.sleep(self.wait_time_after_creation)

            # Buy token
            logger.info(
                f"Buying {self.buy_amount:.6f} SOL worth of {token_info.symbol} on {token_info.platform.value}..."
            )
            buy_result = await self.buyer.execute(token_info)
            logger.info(
                f"Buy result is {buy_result}"
            )

            # Create position immediately after buy (regardless of success/failure)
            entry_ts = buy_result.block_time or int(time() * 1000)
            if buy_result.success:
                # Successful buy - create active position
                position = Position.create_from_buy_result(
                    mint=token_info.mint,
                    platform=token_info.platform,
                    entry_net_price_decimal=buy_result.net_price_sol_decimal(),
                    quantity=buy_result.token_swap_amount_decimal(),
                    token_swapin_amount_raw=buy_result.token_swap_amount_raw,
                    entry_ts=entry_ts,
                    transaction_fee_raw=buy_result.transaction_fee_raw,
                    platform_fee_raw=buy_result.platform_fee_raw,
                    exit_strategy=self.exit_strategy,
                    buy_amount=self.buy_amount,
                    total_net_sol_swapout_amount_raw=buy_result.net_sol_swap_amount_raw(),  # Raw value (negative for buys)
                    take_profit_percentage=self.take_profit_percentage,
                    stop_loss_percentage=self.stop_loss_percentage,
                    trailing_stop_percentage=self.trailing_stop_percentage,
                    max_hold_time=self.max_hold_time,
                )
            else:
                # Failed buy - create inactive position with None values
                position = Position(
                    mint=token_info.mint,
                    platform=token_info.platform,
                    entry_net_price_decimal=None,  # No actual entry
                    token_quantity_decimal=None,  # No tokens acquired
                    total_token_swapin_amount_raw=None,  # No tokens acquired
                    total_token_swapout_amount_raw=None,  # No tokens acquired
                    entry_ts=entry_ts,
                    exit_strategy=self.exit_strategy,
                    is_active=False,  # Mark as inactive
                    exit_reason=ExitReason.FAILED_BUY,
                    transaction_fee_raw=buy_result.transaction_fee_raw,  # Still incurred fees
                    platform_fee_raw=buy_result.platform_fee_raw,  # Still incurred fees
                    buy_amount=self.buy_amount,  # Still intended to buy this amount
                    total_net_sol_swapout_amount_raw=0,  # No SOL spent
                    total_net_sol_swapin_amount_raw=0,   # No SOL received
                )

            if buy_result.success:
                await self._handle_successful_buy(token_info, buy_result, position)
            else:
                await self._handle_failed_buy(token_info, buy_result, position)

            # Only wait for next token in yolo mode
            if self.yolo_mode:
                logger.info(
                    f"YOLO mode enabled. Waiting {self.wait_time_before_new_token} seconds before looking for next token..."
                )
                await asyncio.sleep(self.wait_time_before_new_token)

        except Exception:
            logger.exception(f"Error handling token {token_info.symbol}")

    async def _handle_successful_buy(
        self, token_info: TokenInfo, buy_result: TradeResult, position: Position
    ) -> None:
        """Handle successful token purchase."""
        logger.info(
            f"Successfully bought {token_info.symbol} on {token_info.platform.value}"
        )
        logger.info(f"buy_result.tx_signature: {buy_result.tx_signature}")
        self.traded_mints.add(token_info.mint)
        
        # Persist to database if available
        if self.database_manager:
            try:
                # Insert token info (will be ignored if already exists)
                await self.database_manager.insert_token_info(token_info)
                
                # Insert position
                position_id = await self.database_manager.insert_position(position)
                
                # Insert buy trade
                await self.database_manager.insert_trade(
                    trade_result=buy_result,
                    mint=str(token_info.mint),
                    position_id=position_id,
                    trade_type="buy",
                    run_id=self.run_id,
                )
                
                logger.debug(f"Persisted buy trade and position to database")
            except Exception as e:
                logger.exception(f"Failed to persist buy trade to database: {e}")

        # Choose exit strategy
        if not self.marry_mode:
            if self.exit_strategy == "tp_sl":
                await self._handle_tp_sl_exit(token_info, position)
            elif self.exit_strategy == "trailing":
                await self._handle_trailing_exit(token_info, position)
            elif self.exit_strategy == "time_based":
                await self._handle_time_based_exit(token_info, position)
            elif self.exit_strategy == "manual":
                logger.info("Manual exit strategy - position will remain open")
        else:
            logger.info("Marry mode enabled. Skipping sell operation.")

    async def _handle_failed_buy(
        self, token_info: TokenInfo, buy_result: TradeResult, position: Position
    ) -> None:
        """Handle failed token purchase."""
        logger.error(f"Failed to buy {token_info.symbol}: {buy_result.error_message}")
        
        # Persist failed trade to database if available
        if self.database_manager:
            try:
                # Insert token info (will be ignored if already exists)
                await self.database_manager.insert_token_info(token_info)
                
                # Insert position
                position_id = await self.database_manager.insert_position(position)
                
                # Insert failed buy trade
                await self.database_manager.insert_trade(
                    trade_result=buy_result,
                    mint=str(token_info.mint),
                    position_id=position_id,  # Now has a position!
                    trade_type="buy",
                    run_id=self.run_id,
                )
                
                logger.debug(f"Persisted failed buy trade and position to database")
            except Exception as e:
                logger.exception(f"Failed to persist failed buy trade to database: {e}")
        
        # Close ATA if enabled
        await handle_cleanup_after_failure(
            self.solana_client,
            self.wallet,
            token_info.mint,
            self.priority_fee_manager,
            self.cleanup_mode,
            self.cleanup_with_priority_fee,
            self.cleanup_force_close_with_burn,
        )

    async def _handle_tp_sl_exit(
        self, token_info: TokenInfo, position: Position
    ) -> None:
        """Handle take profit/stop loss exit strategy."""
        logger.info(f"Created position: {position}")
        if position.take_profit_price:
            logger.info(f"Take profit target: {position.take_profit_price:.8f} SOL")
        if position.stop_loss_price:
            logger.info(f"Stop loss target: {position.stop_loss_price:.8f} SOL")

        # Monitor position until exit condition is met
        await self._monitor_position_until_exit(token_info, position)

    async def _handle_trailing_exit(
        self, token_info: TokenInfo, position: Position
    ) -> None:
        """Handle trailing stop exit strategy (no fixed take profit)."""
        logger.info(f"Created trailing position: {position}")
        if position.trailing_stop_percentage is not None:
            logger.info(
                f"Trailing stop: {position.trailing_stop_percentage * 100:.2f}% (updates with highs)"
            )

        await self._monitor_position_until_exit(token_info, position)

    async def _handle_time_based_exit(self, token_info: TokenInfo, position: Position) -> None:
        """Handle legacy time-based exit strategy."""
        logger.info(f"Waiting for {self.wait_time_after_buy} seconds before selling...")
        await asyncio.sleep(self.wait_time_after_buy)

        logger.info(f"Selling {token_info.symbol}...")
        sell_result: TradeResult = await self.seller.execute(token_info, position)

        if sell_result.success:
            logger.info(f"Successfully sold {token_info.symbol}")

            # Close ATA if enabled
            await handle_cleanup_after_sell(
                self.solana_client,
                self.wallet,
                token_info.mint,
                self.priority_fee_manager,
                self.cleanup_mode,
                self.cleanup_with_priority_fee,
                self.cleanup_force_close_with_burn,
            )
        else:
            logger.error(
                f"Failed to sell {token_info.symbol}: {sell_result.error_message}"
            )

    async def _monitor_position_until_exit(
        self, token_info: TokenInfo, position: Position
    ) -> None:
        """Monitor a position until exit conditions are met."""
        logger.info(
            f"Starting position monitoring (check interval: {self.price_check_interval}s)"
        )

        # Get pool address for price monitoring using platform-agnostic method
        pool_address = self._get_pool_address(token_info)
        curve_manager = self.platform_implementations.curve_manager

        while position.is_active:
            try:
                # Get current price from pool/curve
                current_price = await curve_manager.calculate_price(pool_address)

                # Update highest_price if this is a new high
                if current_price > (position.highest_price or 0):
                    position.highest_price = current_price
                    
                    # Update position in database with new highest price
                    if self.database_manager:
                        try:
                            await self.database_manager.update_position(position)
                            logger.debug(f"Updated highest_price to {current_price:.8f} SOL in database")
                        except Exception as e:
                            logger.exception(f"Failed to update position in database: {e}")

                # Check if position should be exited
                should_exit, exit_reason = position.should_exit(current_price)

                if should_exit and exit_reason:
                    logger.info(f"Exit condition met: {exit_reason.value}")
                    logger.info(f"Current onchain price: {current_price:.8f} SOL")

                    # Log PnL before exit
                    pnl = position._get_pnl(current_price)
                    logger.info(f"PNL: {pnl}")

                    # Execute sell
                    sell_result = await self.seller.execute(token_info, position)
                    logger.info(f"Sell result: {sell_result}")

                    if sell_result.success:
                        # Close position with actual exit price
                        position.close_position(sell_result, exit_reason)

                        logger.info(
                            f"Successfully exited position: {exit_reason.value}"
                        )
                        
                        # Persist sell trade and update position
                        if self.database_manager:
                            try:
                                # Update position in database
                                await self.database_manager.update_position(position)
                                
                                # Insert sell trade
                                position_id = PositionConverter.generate_position_id(
                                    position.mint, position.platform, position.entry_ts
                                )
                                await self.database_manager.insert_trade(
                                    trade_result=sell_result,
                                    mint=str(token_info.mint),
                                    position_id=position_id,
                                    trade_type="sell",
                                    run_id=self.run_id,
                                )
                                
                                logger.debug(f"Persisted sell trade and updated position in database")
                            except Exception as e:
                                logger.exception(f"Failed to persist sell trade to database: {e}")

                        # Log final PnL
                        final_pnl = position._get_pnl()
                        logger.info(f"Final net PnL: {final_pnl}")

                        # Close ATA if enabled
                        await handle_cleanup_after_sell(
                            self.solana_client,
                            self.wallet,
                            token_info.mint,
                            self.priority_fee_manager,
                            self.cleanup_mode,
                            self.cleanup_with_priority_fee,
                            self.cleanup_force_close_with_burn,
                        )
                    else:
                        logger.error(
                            f"Failed to exit position: {sell_result.error_message}"
                        )
                        # Keep monitoring in case sell can be retried
                        # Do not break; continue monitoring loop for another attempt
                    
                    if sell_result.success:
                        # Exit monitoring loop after successful sell
                        break
                else:
                    # Log current status
                    pnl = position._get_pnl(current_price)
                    logger.info(
                        f"Position status: {current_price:.8f} SOL ({pnl['net_price_change_pct']:+.2f}%)"
                    )

                # Wait before next price check
                await asyncio.sleep(self.price_check_interval)

            except Exception:
                logger.exception("Error monitoring position")
                await asyncio.sleep(
                    self.price_check_interval
                )  # Continue monitoring despite errors

    def _get_pool_address(self, token_info: TokenInfo) -> Pubkey:
        """Get the pool/curve address for price monitoring using platform-agnostic method."""
        address_provider = self.platform_implementations.address_provider

        # Use platform-specific logic to get the appropriate address
        if hasattr(token_info, "bonding_curve") and token_info.bonding_curve:
            return token_info.bonding_curve
        elif hasattr(token_info, "pool_state") and token_info.pool_state:
            return token_info.pool_state
        else:
            # Fallback to deriving the address using platform provider
            return address_provider.derive_pool_address(token_info.mint)

# Backward compatibility alias
PumpTrader = UniversalTrader  # Legacy name for backward compatibility
