"""
Universal trading coordinator that works with any platform.
Cleaned up to remove all platform-specific hardcoding.
"""

import asyncio
from time import monotonic, time
from typing import Any

import uvloop
from solders.pubkey import Pubkey

from cleanup.modes import cleanup_after_sell
from core.client import SolanaClient
from core.priority_fee.manager import PriorityFeeManager
from core.wallet import Wallet
from database.models import PositionConverter
from interfaces.core import Platform, TokenInfo
from monitoring.listener_factory import ListenerFactory
from platforms import get_platform_implementations
from trading.base import TradeResult
from trading.platform_aware import PlatformAwareBuyer, PlatformAwareSeller
from trading.position import ExitReason, Position
from utils.logger import get_logger
from utils.volatility import VolatilityCalculator, calculate_take_profit_adjustment

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
        # Exit strategy configuration
        exit_strategy: str = "time_based",
        take_profit_percentage: float | None = None,
        stop_loss_percentage: float | None = None,
        trailing_stop_percentage: float | None = None,
        max_hold_time: int | None = None,
        max_no_price_change_time: int | None = None,
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
        database_manager=None,
        # Parallel position configuration
        max_active_mints: int = 1,
        # Blockhash caching configuration
        blockhash_update_interval: float = 10.0,
        # Volatility-based adjustments
        enable_volatility_adjustment: bool = False,
        volatility_window_seconds: float = 5.0,
        volatility_thresholds: dict | None = None,
        volatility_tp_adjustments: dict | None = None,
        # Insufficient gain exit condition
        min_gain_percentage: float | None = None,
        min_gain_time_window: int = 2,
        # Trade tracking configuration
        enable_trade_tracking: bool = False,
        trade_staleness_threshold: float = 30.0,
    ):
        """Initialize the universal trader."""
        # Core components
        self.solana_client = SolanaClient(rpc_endpoint, blockhash_update_interval)
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

        # Platform constants (loaded from blockchain at startup)
        self.platform_constants: dict[str, Any] = {}

        # Trading parameters (needed for validation)
        self.buy_amount = buy_amount

        # Platform setup
        if isinstance(platform, str):
            self.platform = Platform(platform)
        else:
            self.platform = platform

        logger.info(
            f"Initialized Universal Trader for platform: {self.platform.value} with run_id={self.run_id}"
        )

        # Validate platform support
        try:
            from platforms import platform_factory

            if not platform_factory.registry.is_platform_supported(self.platform):
                raise ValueError(f"Platform {self.platform.value} is not supported")
        except Exception:
            logger.exception("Platform validation failed")
            raise

        # Store compute unit configuration
        self.compute_units = compute_units or {}

        # Extract testing configuration
        self.dry_run = False
        dry_run_wait_time = 0.5
        if testing:
            self.dry_run = testing.get("dry_run", False)
            dry_run_wait_time = testing.get("dry_run_wait_time_seconds", 0.5)

        # Trade tracking configuration
        self.enable_trade_tracking = enable_trade_tracking
        self.trade_staleness_threshold = trade_staleness_threshold

        # Initialize the appropriate listener only if trade tracking is enabled
        logger.info(f"Creating token listener for trade tracking (type: {listener_type})")
        self.token_listener = ListenerFactory.create_listener(
            listener_type=listener_type,
            wss_endpoint=wss_endpoint,
            geyser_endpoint=geyser_endpoint,
            geyser_api_token=geyser_api_token,
            geyser_auth_type=geyser_auth_type,
            pumpportal_url=pumpportal_url,
            platforms=[self.platform],  # Only listen for our platform
        )
        if self.token_listener is None:
            logger.error(f"Failed to create token listener for type: {listener_type}")
        else:
            logger.info(f"Successfully created token listener: {type(self.token_listener).__name__}")

        # Get platform-specific implementations (after token_listener is created)
        self.platform_implementations = get_platform_implementations(
            self.platform, self.solana_client, 
            listener=self.token_listener if self.enable_trade_tracking else None,
            trade_staleness_threshold=self.trade_staleness_threshold
        )

        # Create platform-aware traders based on mode
        if self.dry_run:
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
                extreme_fast_mode,
                curve_manager=self.platform_implementations.curve_manager,
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
                curve_manager=self.platform_implementations.curve_manager,
                dry_run_wait_time=dry_run_wait_time,
                compute_units=self.compute_units,
                database_manager=self.database_manager,
            )
        else:
            if self.buy_amount > 0.01:
                raise ValueError(
                    "Buy amount must NOT be greater than 0.01 SOL in live mode"
                )

            # Create platform-aware traders
            self.buyer = PlatformAwareBuyer(
                self.solana_client,
                self.wallet,
                self.priority_fee_manager,
                buy_amount,
                buy_slippage,
                max_retries,
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


        # Trading parameters
        self.buy_slippage = buy_slippage
        self.sell_slippage = sell_slippage
        self.max_retries = max_retries
        self.extreme_fast_mode = extreme_fast_mode

        # Exit strategy parameters
        self.exit_strategy = exit_strategy.lower()
        self.take_profit_percentage = take_profit_percentage
        self.stop_loss_percentage = stop_loss_percentage
        self.trailing_stop_percentage = trailing_stop_percentage
        self.max_hold_time = max_hold_time
        self.max_no_price_change_time = max_no_price_change_time
        self.price_check_interval = price_check_interval

        # Volatility-based adjustments
        self.enable_volatility_adjustment = enable_volatility_adjustment
        self.volatility_window_seconds = volatility_window_seconds
        self.volatility_thresholds = volatility_thresholds or {
            'low': 0.05,    # < 5% per window
            'medium': 0.1,  # 5-10% per window
        }
        self.volatility_tp_adjustments = volatility_tp_adjustments or {
            'low': 0.0,      # No adjustment
            'medium': 0.25,  # Reduce by 25%
            'high': 0.45,    # Reduce by 45%
        }

        # Insufficient gain exit condition
        self.min_gain_percentage = min_gain_percentage
        self.min_gain_time_window = min_gain_time_window
        

        # Timing parameters
        self.wait_time_after_creation = wait_time_after_creation
        self.wait_time_after_buy = wait_time_after_buy
        self.wait_time_before_new_token = wait_time_before_new_token
        self.max_token_age = max_token_age
        self.token_wait_timeout = token_wait_timeout

        # Cleanup parameters
        self.cleanup_force_close_with_burn = cleanup_force_close_with_burn
        self.cleanup_with_priority_fee = cleanup_with_priority_fee

        # Trading filters/modes
        self.match_string = match_string
        self.bro_address = bro_address
        self.marry_mode = marry_mode
        self.yolo_mode = yolo_mode

        # State tracking
        self.active_mints: set[Pubkey] = set()  # Renamed from traded_mints
        self.reserved_mints: set[Pubkey] = set()  # Track reserved buy slots
        self.token_queue: asyncio.Queue[TokenInfo] = asyncio.Queue[TokenInfo]()
        self.processing: bool = False
        self.processed_tokens: set[str] = set()
        self.token_timestamps: dict[str, float] = {}

        # Parallel position tracking
        self.max_active_mints = max_active_mints
        self.position_tasks: dict[str, asyncio.Task] = {}
        self.position_slot_available = asyncio.Event()

    def _mint_prefix(self, mint: Pubkey) -> str:
        """Get short mint prefix for logging."""
        return str(mint)[:8]

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
        logger.info(f"Max no price change time: {self.max_no_price_change_time} seconds")

        try:
            health_resp = await self.solana_client.get_health()
            logger.info(f"RPC warm-up successful (getHealth passed: {health_resp})")
        except Exception as e:
            logger.warning(f"RPC warm-up failed: {e!s}")

        # Load platform-specific constants from blockchain
        try:
            await self._load_platform_constants()
        except Exception as e:
            logger.exception(f"Failed to load platform constants: {e}")
            # Continue without constants - some operations may fail but bot can still run

        # Resume monitoring of any active positions from database before listening
        try:
            await self._resume_active_positions()
        except Exception:
            logger.exception("Failed to resume active positions from database")

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
                    await self.token_listener.listen_for_messages(
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
            self.token_listener.listen_for_messages(
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
        # Cancel all active position monitoring tasks
        if self.position_tasks:
            logger.info(
                f"Cancelling {len(self.position_tasks)} active position task(s)..."
            )
            for task in self.position_tasks.values():
                task.cancel()

            # Wait for all tasks to finish
            await asyncio.gather(*self.position_tasks.values(), return_exceptions=True)
            logger.info("All position tasks cancelled")

        # DO NOT cleanup tokens on interrupt - user can recover manually
        total_interrupted = len(self.active_mints) + len(self.reserved_mints)
        if total_interrupted > 0:
            logger.warning(
                f"{total_interrupted} position(s) interrupted ({len(self.active_mints)} active, {len(self.reserved_mints)} reserved). "
                f"Tokens NOT burned - manual recovery possible."
            )

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

        # Check if mint is already reserved or active
        if token_info.mint in self.reserved_mints or token_info.mint in self.active_mints:
            logger.debug(f"Token {token_info.symbol} already reserved/active. Skipping...")
            return

        # Enforce max_active_mints globally (including resumed positions and reserved slots)
        total_slots = len(self.active_mints) + len(self.reserved_mints)
        if total_slots >= self.max_active_mints:
            # logger.info(
            #     f"[{self._mint_prefix(token_info.mint)}] Skipping new token - at capacity ({total_slots}/{self.max_active_mints})"
            # )
            return

        # Reserve the buy slot immediately
        self.reserved_mints.add(token_info.mint)
        
        # Record timestamp when token was discovered
        self.token_timestamps[token_key] = monotonic()

        await self.token_queue.put(token_info)
        logger.debug(
            f"Queued new token: {token_info.symbol} ({token_info.mint}) on {token_info.platform.value} (slot reserved)"
        )

    async def _process_token_queue(self) -> None:
        """Process tokens concurrently up to max_active_mints limit."""
        while True:
            try:
                # Wait for capacity if at limit (YOLO mode only)
                if self.yolo_mode:
                    total_slots = len(self.active_mints) + len(self.reserved_mints)
                    while total_slots >= self.max_active_mints:
                        self.position_slot_available.clear()
                        await self.position_slot_available.wait()
                        total_slots = len(self.active_mints) + len(self.reserved_mints)

                # Get next token from queue
                token_info = await self.token_queue.get()
                token_key = str(token_info.mint)

                # Check freshness
                current_time = monotonic()
                token_age = current_time - self.token_timestamps.get(
                    token_key, current_time
                )

                if token_age > self.max_token_age:
                    logger.debug(
                        f"[{self._mint_prefix(token_info.mint)}] Skipping - too old ({token_age:.1f}s)"
                    )
                    continue

                self.processed_tokens.add(token_key)

                # Spawn concurrent task
                task = asyncio.create_task(self._handle_token_wrapper(token_info))
                self.position_tasks[token_key] = task

            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Error in token queue processor")
            finally:
                self.token_queue.task_done()

    async def _handle_token_wrapper(self, token_info: TokenInfo) -> None:
        """Wrapper around _handle_token that manages position lifecycle."""
        mint_key = str(token_info.mint)
        try:
            await self._handle_token(token_info)
        finally:
            # Cleanup: Remove from both reserved and active tracking
            # Remove from reserved if still present (may have been cleaned up in failed buy handler)
            if token_info.mint in self.reserved_mints:
                self.reserved_mints.remove(token_info.mint)
                logger.debug(f"[{self._mint_prefix(token_info.mint)}] Released reserved slot in wrapper cleanup")
            
            # Only remove from active if present (buy might have failed)
            if token_info.mint in self.active_mints:
                self.active_mints.remove(token_info.mint)

            if mint_key in self.position_tasks:
                del self.position_tasks[mint_key]

            # Signal that a slot is available for next token
            if self.yolo_mode:
                self.position_slot_available.set()

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
                    f"[{self._mint_prefix(token_info.mint)}] Waiting for {self.wait_time_after_creation} seconds for the pool/curve to stabilize..."
                )
                await asyncio.sleep(self.wait_time_after_creation)

            # Subscribe to trade tracking if enabled (before buying)
            if self.enable_trade_tracking:
                if self.token_listener is None:
                    logger.error(
                        f"[{self._mint_prefix(token_info.mint)}] Trade tracking enabled but token_listener is None"
                    )
                else:
                    try:
                        # Use pump.fun default values (account not yet visible to RPC)
                        await self.token_listener.subscribe_token_trades(
                            mint=str(token_info.mint),  # Use mint as the identifier
                            initial_data={
                                "vSolInBondingCurve": 30.0,  # Pump.fun default virtual SOL reserves
                                "vTokensInBondingCurve": 1073000000.0,  # Pump.fun default virtual token reserves
                            }
                        )
                        logger.debug(
                            f"[{self._mint_prefix(token_info.mint)}] Subscribed to trade tracking for {token_info.symbol}"
                        )
                    except Exception as e:
                        logger.exception(
                            f"[{self._mint_prefix(token_info.mint)}] Failed to subscribe to trade tracking: {e}"
                        )

            # Buy token
            logger.info(
                f"[{self._mint_prefix(token_info.mint)}] Buying {self.buy_amount:.6f} SOL worth of {token_info.symbol} ({str(token_info.mint)})on {token_info.platform.value}..."
            )
            buy_result = await self.buyer.execute(token_info)
            logger.debug(
                f"[{self._mint_prefix(token_info.mint)}] Buy result is {buy_result}"
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
                    max_no_price_change_time=self.max_no_price_change_time,
                    min_gain_percentage=self.min_gain_percentage,
                    min_gain_time_window=self.min_gain_time_window,
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
                    total_net_sol_swapin_amount_raw=0,  # No SOL received
                )

            if buy_result.success:
                await self._handle_successful_buy(token_info, buy_result, position)
            else:
                await self._handle_failed_buy(token_info, buy_result, position)

            # Only wait for next token in yolo mode
            if self.yolo_mode:
                logger.debug(
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
            f"[{self._mint_prefix(token_info.mint)}] Successfully bought {token_info.symbol} on {token_info.platform.value} in transaction {str(buy_result.tx_signature)}"
        )
        logger.info(f"Position: {position}")
        # Move from reserved to active (slot was already reserved when queued)
        if token_info.mint in self.reserved_mints:
            self.reserved_mints.remove(token_info.mint)
        self.active_mints.add(token_info.mint)

        # Subscribe to trade tracking if enabled
        if self.enable_trade_tracking and self.token_listener:
            try:
                # Use pump.fun default values (account not yet visible to RPC)
                await self.token_listener.subscribe_token_trades(
                    mint=str(token_info.mint),  # Use mint as the identifier
                    initial_data={
                        "vSolInBondingCurve": 30.0,  # Pump.fun default virtual SOL reserves
                        "vTokensInBondingCurve": 1073000000.0,  # Pump.fun default virtual token reserves
                    }
                )
                logger.debug(
                    f"[{self._mint_prefix(token_info.mint)}] Subscribed to trade tracking for {token_info.symbol}"
                )
            except Exception as e:
                logger.exception(
                    f"[{self._mint_prefix(token_info.mint)}] Failed to subscribe to trade tracking: {e}"
                )

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

                logger.debug("Persisted buy trade and position to database")
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
        logger.error(
            f"[{self._mint_prefix(token_info.mint)}] Failed to buy {token_info.symbol}: {buy_result.error_message}"
        )

        # Calculate realized PnL for failed buy (loss from fees only)
        pnl_dict = position._get_pnl()
        position.realized_pnl_sol_decimal = pnl_dict["realized_pnl_sol_decimal"]
        position.realized_net_pnl_sol_decimal = pnl_dict["realized_net_pnl_sol_decimal"]
        
        from core.pubkeys import LAMPORTS_PER_SOL
        logger.info(
            f"[{self._mint_prefix(token_info.mint)}] Failed buy PnL: {position.realized_pnl_sol_decimal:.6f} SOL "
            f"(fees: {pnl_dict['total_fees_raw'] / LAMPORTS_PER_SOL:.6f} SOL)"
        )

        # Clean up reserved slot since buy failed
        if token_info.mint in self.reserved_mints:
            self.reserved_mints.remove(token_info.mint)
            logger.debug(f"[{self._mint_prefix(token_info.mint)}] Released reserved slot after failed buy")

        # Unsubscribe from trade tracking if enabled (buy failed)
        if self.enable_trade_tracking and self.token_listener:
            try:
                await self.token_listener.unsubscribe_token_trades(mint=str(token_info.mint))
                logger.info(
                    f"[{self._mint_prefix(token_info.mint)}] Unsubscribed from trade tracking after failed buy"
                )
            except Exception as e:
                logger.exception(
                    f"[{self._mint_prefix(token_info.mint)}] Failed to unsubscribe from trade tracking: {e}"
                )

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

                logger.debug(
                    f"[{self._mint_prefix(token_info.mint)}] Persisted failed buy trade and position to database"
                )
            except Exception as e:
                logger.exception(
                    f"[{self._mint_prefix(token_info.mint)}] Failed to persist failed buy trade to database: {e}"
                )

        # No ATA cleanup needed - buy is atomic, no ATA created

    async def _handle_tp_sl_exit(
        self, token_info: TokenInfo, position: Position
    ) -> None:
        """Handle take profit/stop loss exit strategy."""

        # Monitor position until exit condition is met
        await self._monitor_position_until_exit(token_info, position)

    async def _handle_trailing_exit(
        self, token_info: TokenInfo, position: Position
    ) -> None:
        """Handle trailing stop exit strategy (no fixed take profit)."""

        await self._monitor_position_until_exit(token_info, position)

    async def _handle_time_based_exit(
        self, token_info: TokenInfo, position: Position
    ) -> None:
        """Handle legacy time-based exit strategy."""
        logger.info(
            f"[{self._mint_prefix(token_info.mint)}] Waiting for {self.wait_time_after_buy} seconds before selling..."
        )
        await asyncio.sleep(self.wait_time_after_buy)

        logger.info(
            f"[{self._mint_prefix(token_info.mint)}] Selling {token_info.symbol}..."
        )
        sell_result: TradeResult = await self.seller.execute(token_info, position)

        if sell_result.success:
            logger.info(
                f"[{self._mint_prefix(token_info.mint)}] Successfully sold {token_info.symbol} in transaction {str(sell_result.tx_signature)}"
            )

            # Always cleanup ATA after sell
            await cleanup_after_sell(
                self.solana_client,
                self.wallet,
                token_info.mint,
                self.priority_fee_manager,
                self.cleanup_with_priority_fee,
                self.cleanup_force_close_with_burn,
            )
        else:
            logger.error(
                f"[{self._mint_prefix(token_info.mint)}] Failed to sell {token_info.symbol}: {sell_result.error_message}"
            )

        # Unsubscribe from trade tracking when time-based exit completes
        if self.enable_trade_tracking and self.token_listener:
            try:
                await self.token_listener.unsubscribe_token_trades(mint=str(token_info.mint))
                logger.info(
                    f"[{self._mint_prefix(token_info.mint)}] Unsubscribed from trade tracking after time-based exit"
                )
            except Exception as e:
                logger.exception(
                    f"[{self._mint_prefix(token_info.mint)}] Failed to unsubscribe from trade tracking: {e}"
                )

    async def _monitor_position_until_exit(
        self, token_info: TokenInfo, position: Position
    ) -> None:
        """Monitor a position until exit conditions are met."""
        logger.debug(
            f"[{self._mint_prefix(token_info.mint)}] Starting position monitoring (check interval: {self.price_check_interval}s)"
        )

        pool_address = self._get_pool_address(token_info)
        curve_manager = self.platform_implementations.curve_manager
        
        # Track last price for change detection
        last_price = None
        
        # Initialize volatility calculator if enabled (will be created when first price is available)
        volatility_calculator = None
        volatility_monitoring_started = False
        if self.enable_volatility_adjustment:
            logger.debug(
                f"[{self._mint_prefix(token_info.mint)}] Volatility adjustment enabled (window: {self.volatility_window_seconds}s) - waiting for first price"
            )

        while position.is_active:
            try:
                # Get current price from pool/curve
                current_price = await curve_manager.calculate_price(token_info.mint,token_info.bonding_curve)
                current_timestamp = time()
                current_timestamp_ms = int(current_timestamp * 1000)

                # Set monitoring start time on first price (for time-based exit conditions)
                if position.monitoring_start_ts is None:
                    position.set_monitoring_start_time(current_timestamp_ms)

                # Store price in database if available
                if self.database_manager:
                    try:
                        await self.database_manager.insert_price_history(
                            mint=str(token_info.mint),
                            platform=token_info.platform.value,
                            price_decimal=current_price,
                        )
                    except Exception as e:
                        logger.exception(f"Failed to insert price history: {e}")

                # Calculate volatility and adjust take profit if enabled
                if self.enable_volatility_adjustment:
                    # Initialize volatility calculator on first valid price
                    if not volatility_monitoring_started:
                        volatility_calculator = VolatilityCalculator(self.volatility_window_seconds)
                        volatility_monitoring_started = True
                        logger.info(
                            f"[{self._mint_prefix(token_info.mint)}] Started volatility monitoring with first price: {current_price:.8f} SOL"
                        )
                    
                    # Add current price to volatility calculator
                    volatility_calculator.add_price(current_price, current_timestamp)
                    
                    if volatility_calculator.has_sufficient_data(current_timestamp):
                        previous_value = volatility_calculator._last_volatility
                        volatility_level = volatility_calculator.get_volatility_level(current_timestamp)
                        volatility_value = volatility_calculator.get_cached_volatility()
                        
                        if volatility_value is not None:
                            logger.debug(
                                f"[{self._mint_prefix(token_info.mint)}] Volatility: {volatility_value:.4f} ({volatility_level})"
                            )
                            
                            # Adjust take profit based on volatility
                            if position.take_profit_price and self.take_profit_percentage:
                                original_tp_price = position.entry_net_price_decimal * (1 + self.take_profit_percentage)
                                volatility_adjusted_take_profit_percentage = calculate_take_profit_adjustment(
                                    self.take_profit_percentage,
                                    volatility_level,
                                    self.volatility_tp_adjustments
                                )
                                volatility_adjusted_take_profit_price = position.entry_net_price_decimal * (1 + volatility_adjusted_take_profit_percentage)
                                
                                # Only adjust if the new TP is more conservative (lower)
                                if volatility_adjusted_take_profit_price != position.take_profit_price:
                                    old_tp_price = position.take_profit_price
                                    position.take_profit_price = volatility_adjusted_take_profit_price
                                    logger.info(
                                        f"[{self._mint_prefix(token_info.mint)}] TP adjusted due to {volatility_level} volatility: "
                                        f"{old_tp_price:.10f} -> {volatility_adjusted_take_profit_price:.10f} SOL "
                                        f"(reduction: {((old_tp_price - volatility_adjusted_take_profit_price) / old_tp_price * 100):.1f}%)"
                                    )

                # Update position fields based on current price
                # Update highest_price if this is a new high
                if current_price > (position.highest_price or 0):
                    position.highest_price = current_price
                    logger.debug(f"[{self._mint_prefix(token_info.mint)}] New highest price: {current_price:.8f} SOL")

                # Track price changes for max_no_price_change_time exit condition
                if position.max_no_price_change_time is not None:
                    price_tolerance = 1e-10  # Very small tolerance for floating point precision
                    if (last_price is None or 
                        abs(current_price - last_price) > price_tolerance):
                        position.last_price_change_ts = time()
                        logger.debug(f"[{self._mint_prefix(token_info.mint)}] Price changed to {current_price:.8f} SOL, updating timestamp")
                    last_price = current_price

                # Update position in database on every loop run
                if self.database_manager:
                    try:
                        await self.database_manager.update_position(position)
                        logger.debug(f"[{self._mint_prefix(token_info.mint)}] Updated position in database")
                    except Exception as e:
                        logger.exception(f"Failed to update position in database: {e}")

                # Check if position should be exited
                should_exit, exit_reason = position.should_exit(current_price)

                if should_exit and exit_reason:
                    logger.info(
                        f"[{self._mint_prefix(token_info.mint)}] Exit condition met: {exit_reason.value}"
                    )
                    logger.info(
                        f"[{self._mint_prefix(token_info.mint)}] Current onchain price: {current_price:.10f} SOL"
                    )

                    # Log PnL before exit
                    pnl = position._get_pnl(current_price)
                    logger.info(f"[{self._mint_prefix(token_info.mint)}] PNL: {pnl}")

                    # Execute sell
                    sell_result = await self.seller.execute(token_info, position)
                    logger.debug(
                        f"[{self._mint_prefix(token_info.mint)}] Sell result: {sell_result}"
                    )

                    if sell_result.success:
                        # Close position with actual exit price
                        position.close_position(sell_result, exit_reason)

                        logger.info(
                            f"[{self._mint_prefix(token_info.mint)}] Successfully exited position: {exit_reason.value}"
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

                                logger.debug(
                                    "Persisted sell trade and updated position in database"
                                )
                            except Exception as e:
                                logger.exception(
                                    f"[{self._mint_prefix(token_info.mint)}] Failed to persist sell trade to database: {e}"
                                )

                        # Log final PnL
                        final_pnl = position._get_pnl()
                        logger.info(f"[{self._mint_prefix(token_info.mint)}] Final net PnL: {final_pnl}")

                        # Always cleanup ATA after sell
                        if not self.dry_run:
                            await cleanup_after_sell(
                                self.solana_client,
                                self.wallet,
                                token_info.mint,
                                self.priority_fee_manager,
                                self.cleanup_with_priority_fee,
                                self.cleanup_force_close_with_burn,
                            )
                    else:
                        logger.error(
                            f"[{self._mint_prefix(token_info.mint)}] Failed to exit position: {sell_result.error_message}"
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
                        f"[{self._mint_prefix(token_info.mint)}] Position status: {current_price:.10f} SOL ({pnl['net_price_change_pct']:+.2f}%)"
                    )

            except Exception:
                logger.exception("Error monitoring position, continuing though")
            # Wait before next price check
            await asyncio.sleep(self.price_check_interval)

        # Unsubscribe from trade tracking when monitoring ends
        if self.enable_trade_tracking and self.token_listener:
            try:
                await self.token_listener.unsubscribe_token_trades(mint=str(token_info.mint))
                logger.debug(
                    f"[{self._mint_prefix(token_info.mint)}] Unsubscribed from trade tracking after position monitoring ended"
                )
            except Exception as e:
                logger.exception(
                    f"[{self._mint_prefix(token_info.mint)}] Failed to unsubscribe from trade tracking: {e}"
                )

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

    async def _load_platform_constants(self) -> None:
        """Load platform-specific constants from the blockchain."""
        try:
            # Get platform implementations
            implementations = get_platform_implementations(self.platform, self.solana_client)
            curve_manager = implementations.curve_manager
            
            # Load constants from the curve manager
            self.platform_constants = await curve_manager.get_platform_constants()
            
            logger.info(f"Successfully loaded {len(self.platform_constants)} platform constants for {self.platform.value}")
                            
        except Exception as e:
            logger.exception(f"Failed to load platform constants for {self.platform.value}")
            raise

    async def _resume_active_positions(self) -> None:
        """Load active positions from DB and resume monitoring with current config."""
        if not self.database_manager:
            return

        try:
            positions = await self.database_manager.get_active_positions(
                min_gain_percentage=self.min_gain_percentage,
                min_gain_time_window=self.min_gain_time_window
            )
        except Exception as e:
            logger.exception(f"Error loading active positions: {e}")
            return

        if not positions:
            return

        logger.info(f"Resuming {len(positions)} active position(s) from database")

        for position in positions:
            logger.info(f"Handling {position}")
            try:
                token_info = await self.database_manager.get_token_info(
                    str(position.mint), position.platform.value
                )
            except Exception as e:
                logger.exception(f"Failed to load TokenInfo for {position.mint}: {e}")
                continue

            if not token_info:
                logger.warning(
                    f"Missing TokenInfo in DB for {self._mint_prefix(position.mint)}; skipping resume"
                )
                continue

            # Apply current exit configuration to the loaded position
            position._apply_exit_strategy_config(
                exit_strategy=self.exit_strategy,
                take_profit_percentage=self.take_profit_percentage,
                stop_loss_percentage=self.stop_loss_percentage,
                trailing_stop_percentage=self.trailing_stop_percentage,
                max_hold_time=self.max_hold_time,
                max_no_price_change_time=self.max_no_price_change_time,
            )
            # Track as active and start monitoring task
            self.active_mints.add(position.mint)
            task_key = str(position.mint)
            task = asyncio.create_task(
                self._monitor_resumed_position(token_info, position)
            )
            self.position_tasks[task_key] = task

    def _apply_current_exit_config(self, position: Position) -> None:
        """Override exit strategy thresholds using current bot config."""
        # Use the Position's built-in method to avoid code duplication

        
    async def _monitor_resumed_position(
        self, token_info: TokenInfo, position: Position
    ) -> None:
        """Wrapper to monitor a resumed position and cleanup on completion."""
        mint_key = str(position.mint)
        try:
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
                logger.info(
                    "Marry mode enabled. Skipping sell operation for resumed position."
                )
        finally:
            # Remove from active tracking when monitoring ends or task is cancelled
            if position.mint in self.active_mints:
                self.active_mints.remove(position.mint)
            if mint_key in self.position_tasks:
                del self.position_tasks[mint_key]
            self.position_slot_available.set()


# Backward compatibility alias
PumpTrader = UniversalTrader  # Legacy name for backward compatibility
