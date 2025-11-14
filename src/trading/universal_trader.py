"""
Universal trading coordinator that works with any platform.
Cleaned up to remove all platform-specific hardcoding.
"""

import asyncio
from time import monotonic, time
from typing import Any

from database.manager import DatabaseManager
import uvloop
from solders.pubkey import Pubkey

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
from trading.position_monitor import PositionMonitor
from utils.logger import get_logger
from utils.volatility import VolatilityCalculator, calculate_take_profit_adjustment

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

logger = get_logger(__name__)


class UniversalTrader:
    """Universal trading coordinator that works with any supported platform."""

    def __init__(
        self,
        rpc_config: dict,
        wallet: Wallet,
        buy_amount: float,
        buy_slippage: float,
        sell_slippage: float,
        max_wallet_loss_percentage: float | None = None,
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
        wait_time_before_new_token: int = 15,
        max_token_age: int | float = 0.001,
        token_wait_timeout: int = 30,
        # Cleanup settings
        cleanup_force_close_with_burn: bool = False,
        cleanup_with_priority_fee: bool = False,
        # Trading filters
        match_string: str | None = None,
        bro_address: str | None = None,
        max_buys: int | None = None,
        min_initial_buy_sol: float = 1.0,
        # Compute unit configuration
        compute_units: dict | None = None,
        # Testing configuration
        testing: dict | None = None,
        offset_wallets: list[str] | None = None,
        # Database configuration
        database_manager: DatabaseManager | None = None,
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
        # Market quality configuration
        market_quality_config: dict | None = None,
    ):
        """Initialize the universal trader."""
        # Core components
        self.solana_client = SolanaClient(rpc_config, blockhash_update_interval)
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

        # Convert offset_wallets to set for efficient lookup, but only if dry_run is enabled
        excluded_wallets = set(offset_wallets) if (offset_wallets and self.dry_run) else set()
        if offset_wallets and not self.dry_run:
            logger.warning(
                "offset_wallets is configured but dry_run is disabled. "
                "Excluded wallet offsets will only be active in dry_run mode."
            )

        # Initialize the appropriate listener only if trade tracking is enabled
        if self.enable_trade_tracking:
            logger.info(f"Creating token listener for trade tracking (type: {listener_type})")
            self.token_listener = ListenerFactory.create_listener(
                listener_type=listener_type,
                wss_endpoint=rpc_config["wss_endpoint"],
                geyser_endpoint=geyser_endpoint,
                geyser_api_token=geyser_api_token,
                geyser_auth_type=geyser_auth_type,
                pumpportal_url=pumpportal_url,
                platforms=[self.platform],  # Only listen for our platform
                excluded_wallets=excluded_wallets,
                database_manager=self.database_manager,
            )
            if self.token_listener is None:
                logger.error(f"Failed to create token listener for type: {listener_type}")
            else:
                logger.info(f"Successfully created token listener: {type(self.token_listener).__name__}")
        else:
            self.token_listener = None
            logger.info("Trade tracking disabled, no token listener created")

        # Market quality configuration
        if market_quality_config and market_quality_config.get("enabled", False):
            from trading.market_quality import MarketQualityController

            lookback_minutes = market_quality_config.get("lookback_minutes")
            exploration_probability = market_quality_config.get("exploration_probability")
            min_trades_for_analysis = market_quality_config.get("min_trades_for_analysis")
            algorithm = market_quality_config.get("algorithm", "win_rate")

            if lookback_minutes is None:
                raise ValueError("market_quality.lookback_minutes is required when enabled")
            if exploration_probability is None:
                raise ValueError(
                    "market_quality.exploration_probability is required when enabled"
                )
            if min_trades_for_analysis is None:
                raise ValueError(
                    "market_quality.min_trades_for_analysis is required when enabled"
                )

            self.market_quality_controller = MarketQualityController(
                database_manager=database_manager,
                lookback_minutes=lookback_minutes,
                exploration_probability=exploration_probability,
                min_trades_for_analysis=min_trades_for_analysis,
                algorithm=algorithm,
            )
            logger.info(
                f"Market quality controller enabled "
                f"(algorithm: {algorithm}, lookback: {lookback_minutes} min, "
                f"exploration: {exploration_probability:.2%}, "
                f"min_trades: {min_trades_for_analysis})"
            )
        else:
            self.market_quality_controller = None
            logger.info("Market quality controller disabled")

        # Store RPC endpoints for listener creation
        self.wss_endpoint = rpc_config["wss_endpoint"]

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
            if self.buy_amount > 0.1:
                raise ValueError(
                    "Buy amount must NOT be greater than 0.1 SOL in live mode"
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
        self.wait_time_before_new_token = wait_time_before_new_token
        self.max_token_age = max_token_age
        self.token_wait_timeout = token_wait_timeout

        # Cleanup parameters
        self.cleanup_force_close_with_burn = cleanup_force_close_with_burn
        self.cleanup_with_priority_fee = cleanup_with_priority_fee

        # Trading filters/modes
        self.match_string = match_string
        self.bro_address = bro_address
        self.max_buys = max_buys
        self.min_initial_buy_sol = min_initial_buy_sol
        self.buy_count = 0  # Counter for number of successful buys

        # State tracking
        self.token_queue: asyncio.Queue[TokenInfo] = asyncio.Queue[TokenInfo]()
        self.processing: bool = False
        self.token_timestamps: dict[str, float] = {}
        self._queued_tokens: set[str] = set()  # Track tokens already queued to prevent duplicates
        self._stop_event = asyncio.Event()  # Event to signal stopping
        self._main_stop_event = asyncio.Event()  # Event to signal main loop stopping
        self._max_buys_reached = False  # Flag to track if max_buys limit reached

        # Parallel position tracking
        self.max_active_mints = max_active_mints
        self.position_tasks: dict[str, asyncio.Task] = {}
        
        # Wallet balance monitoring
        self._balance_monitoring_task: asyncio.Task | None = None
        self._initial_wallet_balance_sol: float | None = None  # Store initial balance for drawdown calculation
        
        # Circuit breaker configuration
        self.max_wallet_loss_percentage = max_wallet_loss_percentage
        self._circuit_breaker_triggered: bool = False

    def _mint_prefix(self, mint: Pubkey) -> str:
        """Get short mint prefix for logging."""
        return str(mint)[:8]

    async def _log_wallet_balance(self) -> None:
        """Log the current SOL balance for this trader's wallet."""
        try:
            balance_lamports = await self.solana_client.get_sol_balance(self.wallet.pubkey)
            balance_sol = balance_lamports / 1_000_000_000  # Convert lamports to SOL
            
            # Store initial balance on first log
            if self._initial_wallet_balance_sol is None:
                self._initial_wallet_balance_sol = balance_sol
                logger.info(f"Initial wallet balance: {balance_sol} SOL")
            
            # Calculate increase
            increase_sol = balance_sol - self._initial_wallet_balance_sol
            increase_percentage = (increase_sol / self._initial_wallet_balance_sol) * 100 if self._initial_wallet_balance_sol > 0 else 0
            
            # Format increase display
            increase_text = f" | Increase: {increase_sol} SOL ({increase_percentage:+.2f}%)"
            
            # Check circuit breaker
            self._check_circuit_breaker(balance_sol)
            
            logger.info(
                f"Wallet {self.wallet.pubkey} balance: "
                f"{balance_sol} SOL ({balance_lamports:,} lamports){increase_text}"
            )
            
            # Store in database if available
            if self.database_manager:
                try:
                    await self.database_manager.insert_wallet_balance(
                        wallet_pubkey=str(self.wallet.pubkey),
                        balance_sol=balance_sol,
                        balance_lamports=balance_lamports,
                        run_id=self.run_id,
                    )
                except Exception as e:
                    logger.exception(f"Failed to store wallet balance in database: {e}")
                    
        except Exception as e:
            logger.error(f"Failed to get wallet balance: {e}")

    def _check_circuit_breaker(self, current_balance_sol: float) -> bool:
        """Check if circuit breaker should be triggered based on wallet loss.
        
        Args:
            current_balance_sol: Current wallet balance in SOL
            
        Returns:
            True if circuit breaker is triggered (stop buying), False otherwise
        """
        # If already triggered, stay triggered
        if self._circuit_breaker_triggered:
            return True
        
        # If no threshold configured, circuit breaker is disabled
        if self.max_wallet_loss_percentage is None:
            return False
        
        # If no initial balance yet, cannot check
        if self._initial_wallet_balance_sol is None:
            return False
        
        # Calculate current loss percentage
        loss_percentage = ((self._initial_wallet_balance_sol - current_balance_sol) / self._initial_wallet_balance_sol) * 100
        
        # Trigger if loss exceeds threshold
        if loss_percentage >= self.max_wallet_loss_percentage:
            self._circuit_breaker_triggered = True
            logger.warning(
                f"CIRCUIT BREAKER TRIGGERED: Wallet loss {loss_percentage:.2f}% exceeds threshold "
                f"{self.max_wallet_loss_percentage:.2f}%. Stopping new token purchases."
            )
            return True
        
        return False

    async def _start_balance_monitoring(self) -> None:
        """Start background wallet balance monitoring."""
        logger.debug("Starting wallet balance monitoring (every 60 seconds)")
        
        # Log initial balance
        await self._log_wallet_balance()
        
        # Start background monitoring task
        self._balance_monitoring_task = asyncio.create_task(self._balance_monitoring_loop())

    async def _balance_monitoring_loop(self) -> None:
        """Background loop for wallet balance monitoring."""
        while not self._stop_event.is_set():
            try:
                await asyncio.sleep(60)  # Check every minute
                await self._log_wallet_balance()
            except asyncio.CancelledError:
                logger.debug("Balance monitoring task cancelled")
                break
            except Exception as e:
                logger.exception(f"Error in balance monitoring loop: {e}")
                await asyncio.sleep(5)  # Short delay before retrying

    async def _stop_balance_monitoring(self) -> None:
        """Stop background wallet balance monitoring."""
        if self._balance_monitoring_task:
            self._balance_monitoring_task.cancel()
            try:
                await self._balance_monitoring_task
            except asyncio.CancelledError:
                pass
            self._balance_monitoring_task = None
            logger.debug("Stopped wallet balance monitoring")

    async def start(self) -> None:
        """Start the trading bot and listen for new tokens."""
        logger.info(f"Starting Universal Trader for {self.platform.value}")
        logger.info(
            f"Match filter: {self.match_string if self.match_string else 'None'}"
        )
        logger.info(
            f"Creator filter: {self.bro_address if self.bro_address else 'None'}"
        )
        logger.info(f"Max buys: {self.max_buys if self.max_buys else 'unlimited'}")

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
            # Start wallet balance monitoring
            await self._start_balance_monitoring()
            
            # Start continuous trading with max_buys limit
            logger.info(f"Starting continuous trading (max_buys: {self.max_buys})")
            processor_task = asyncio.create_task(self._process_token_queue())

            try:
                # Create a task for the listener
                listener_task = asyncio.create_task(
                    self.token_listener.listen_for_messages(
                        lambda token: self._queue_token(token),
                        self.match_string,
                        self.bro_address,
                    )
                )
                
                # Wait for either the listener to stop or the main stop event
                done, pending = await asyncio.wait(
                    [listener_task, asyncio.create_task(self._main_stop_event.wait())],
                    return_when=asyncio.FIRST_COMPLETED
                )
                
                # Cancel the remaining task
                for task in pending:
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
                        
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


    async def _cleanup_resources(self) -> None:
        """Perform cleanup operations before shutting down."""
        # Stop wallet balance monitoring
        await self._stop_balance_monitoring()
        
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
        total_interrupted = len(self.position_tasks)
        if total_interrupted > 0:
            logger.warning(
                f"{total_interrupted} position(s) interrupted. "
                f"Tokens NOT burned - manual recovery possible."
            )

        # old_keys = {k for k in self.token_timestamps if k not in self.processed_tokens}
        # for key in old_keys:
        #     self.token_timestamps.pop(key, None)

        await self.solana_client.close()

    async def _queue_token(self, token_info: TokenInfo) -> None:
        """Queue a token for processing if not already processed."""

        # Check if we should stop
        if self._stop_event.is_set():
            logger.info("Stop event set, ignoring new tokens")
            return
            
        token_key = str(token_info.mint)

        # Prevent duplicate queuing (can happen during WebSocket reconnection)
        if token_key in self._queued_tokens:
            logger.warning(
                f"[{self._mint_prefix(token_info.mint)}] Token already queued, skipping duplicate"
            )
            return

        # Check capacity using active position tasks
        if len(self.position_tasks) >= self.max_active_mints:
            logger.info(
                f"[{self._mint_prefix(token_info.mint)}] Skipping new token - at capacity ({len(self.position_tasks)}/{self.max_active_mints})"
            )
            return

        # Check circuit breaker
        if self._circuit_breaker_triggered:
            logger.warning(
                f"[{self._mint_prefix(token_info.mint)}] Skipping token due to circuit breaker (wallet loss threshold exceeded)"
            )
            return

        if token_info.initial_buy_sol_amount_decimal < self.min_initial_buy_sol:
            logger.info(
                f"[{self._mint_prefix(token_info.mint)}] Skipping token from creator [{str(token_info.creator)[:8]}] because initial SOL amount is too low {token_info.initial_buy_sol_amount_decimal} SOL (min: {self.min_initial_buy_sol} SOL)"
            )
            return

        # Record timestamp when token was discovered
        self.token_timestamps[token_key] = monotonic()
        
        # Mark as queued to prevent duplicates
        self._queued_tokens.add(token_key)

        await self.token_queue.put(token_info)
        logger.debug(
            f"Queued new token: {token_info.symbol} ({token_info.mint}) on {token_info.platform.value} (slot reserved)"
        )

    async def _process_token_queue(self) -> None:
        """Process tokens concurrently up to max_active_mints limit."""
        while True:
            token_info = None
            task_done_called = False
            task_spawned = False
            try:
                # Get next token from queue
                token_info = await self.token_queue.get()
                token_key = str(token_info.mint)

                # Check freshness
                current_time = monotonic()
                token_age = current_time - self.token_timestamps.get(
                    token_key
                )
                self.token_timestamps.pop(token_key, None)
                if token_age > self.max_token_age:
                    logger.debug(
                        f"[{self._mint_prefix(token_info.mint)}] Skipping - too old ({token_age:.1f}s)"
                    )
                    # Remove from queued set since we're skipping it (not processing)
                    self._queued_tokens.discard(token_key)
                    # Mark task as done before continuing
                    self.token_queue.task_done()
                    task_done_called = True
                    continue


                # Spawn concurrent task
                task = asyncio.create_task(self._handle_token(token_info))
                self.position_tasks[token_key] = task
                task_spawned = True
                logger.debug(f"_process_token_queue() [{self._mint_prefix(token_info.mint)}] Created position task")
                
                # Mark task as done after successfully spawning the task
                # Note: We don't discard from _queued_tokens here - _handle_token_wrapper will do it
                self.token_queue.task_done()
                task_done_called = True
            except asyncio.CancelledError:
                # Mark task as done if we got a token before cancellation and haven't called it yet
                if token_info is not None and not task_done_called:
                    self.token_queue.task_done()
                # Discard from queued set if we didn't spawn a task (prevents leak)
                if token_info is not None and not task_spawned:
                    self._queued_tokens.discard(str(token_info.mint))
                break
            except Exception:
                logger.exception("Error in token queue processor")
                # Mark task as done even on error to maintain queue consistency
                if token_info is not None and not task_done_called:
                    self.token_queue.task_done()
                # Discard from queued set if we didn't spawn a task (prevents leak)
                if token_info is not None and not task_spawned:
                    self._queued_tokens.discard(str(token_info.mint))

    async def _handle_token(self, token_info: TokenInfo) -> None:
        """Handle a new token creation event."""
        try:
            # Validate that token is for our platform
            if token_info.platform != self.platform:
                logger.warning(
                    f"Token platform mismatch: expected {self.platform.value}, got {token_info.platform.value}"
                )
                return

            # Persist to database if available
            if self.database_manager:
                try:
                    # Insert token info (will be ignored if already exists)
                    await self.database_manager.insert_token_info(token_info)
                except Exception as e:
                    logger.exception(f"Failed to persist token info to database: {e}")

            # Wait for pool/curve to stabilize (unless in extreme fast mode)
            if not self.extreme_fast_mode:
                logger.info(
                    f"[{self._mint_prefix(token_info.mint)}] Waiting for {self.wait_time_after_creation} seconds for the pool/curve to stabilize..."
                )
                await asyncio.sleep(self.wait_time_after_creation)

            # Check market quality buy decision (synchronous, uses cached score)
            if self.market_quality_controller:
                should_buy = self.market_quality_controller.should_buy()
                if not should_buy:
                    quality_score = self.market_quality_controller.get_cached_quality_score()
                    logger.info(
                        f"[{self._mint_prefix(token_info.mint)}] Skipping buy - "
                        f"market quality insufficient (score: {quality_score:.2f})"
                    )
                    return

            # Subscribe to trade tracking if enabled (before buying)
            if self.enable_trade_tracking:
                # Validate that we have virtual reserves for trade tracking
                if token_info.virtual_sol_reserves is None or token_info.virtual_token_reserves is None:
                    logger.error(f"Cannot subscribe to trade tracking: missing virtual reserves for {token_info.symbol}")
                else:
                    try:
                        # Subscribe to trade tracking
                        await self.token_listener.subscribe_token_trades(token_info)
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
            logger.info(
                f"[{self._mint_prefix(token_info.mint)}] Buy result is {buy_result}"
            )

            # Create position immediately after buy (regardless of success/failure)
            entry_ts = (buy_result.block_time or int(time())) * 1000
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
                    tip_fee_raw=buy_result.tip_fee_raw,
                    rent_exemption_amount_raw=buy_result.rent_exemption_amount_raw,
                    unattributed_sol_amount_raw=buy_result.unattributed_sol_amount_raw,
                    exit_strategy="trailing",
                    buy_amount=self.buy_amount,
                    total_net_sol_swapout_amount_raw=buy_result.net_sol_swap_amount_raw,  # Raw value (negative for buys)
                    total_sol_swapout_amount_raw=buy_result.sol_swap_amount_raw,  # Raw value (negative for buys)
                    take_profit_percentage=self.take_profit_percentage,
                    stop_loss_percentage=self.stop_loss_percentage,
                    trailing_stop_percentage=self.trailing_stop_percentage,
                    max_hold_time=self.max_hold_time,
                    max_no_price_change_time=self.max_no_price_change_time,
                    min_gain_percentage=self.min_gain_percentage,
                    min_gain_time_window=self.min_gain_time_window,
                )
            else:
                position_id = Position.generate_position_id(token_info.mint, token_info.platform, entry_ts)
                # Failed buy - create inactive position with None values
                position = Position(
                    position_id=position_id,
                    mint=token_info.mint,
                    platform=token_info.platform,
                    entry_net_price_decimal=None,  # No actual entry
                    token_quantity_decimal=None,  # No tokens acquired
                    total_token_swapin_amount_raw=None,  # No tokens acquired
                    total_token_swapout_amount_raw=None,  # No tokens acquired
                    entry_ts=entry_ts,
                    exit_ts=entry_ts,   #yes, = entry_ts
                    exit_strategy="trailing",
                    is_active=False,  # Mark as inactive
                    exit_reason=ExitReason.FAILED_BUY,
                    transaction_fee_raw=buy_result.transaction_fee_raw,  # Still incurred fees
                    platform_fee_raw=buy_result.platform_fee_raw,  # Still incurred fees
                    tip_fee_raw=buy_result.tip_fee_raw,  # Still incurred fees
                    rent_exemption_amount_raw=buy_result.rent_exemption_amount_raw,
                    unattributed_sol_amount_raw=buy_result.unattributed_sol_amount_raw,
                    buy_amount=self.buy_amount,  # Still intended to buy this amount
                    total_net_sol_swapout_amount_raw=0,  # No SOL spent
                    total_net_sol_swapin_amount_raw=0,  # No SOL received
                    total_sol_swapout_amount_raw=buy_result.sol_swap_amount_raw,  # SOL spent
                    total_sol_swapin_amount_raw=0,  # No SOL received
                )
            if self.database_manager:
                try:
                    # Insert position
                    await self.database_manager.insert_position(position)

                    logger.debug("Persisted position to database")
                except Exception as e:
                    logger.exception(f"Failed to persist position to database: {e}")

            # Increment buy counter regardless of success/failure
            self.buy_count += 1
            logger.info(f"Buy count: {self.buy_count}/{self.max_buys if self.max_buys else 'unlimited'}")
            
            # Check if we've reached max_buys
            if self.max_buys and self.buy_count >= self.max_buys:
                logger.info(f"Reached max_buys limit ({self.max_buys}). Will stop after all positions close...")
                self._max_buys_reached = True
                self._stop_event.set()  # Stop accepting new tokens

            if buy_result.success:
                await self._handle_successful_buy(token_info, buy_result, position)
            else:
                await self._handle_failed_buy(token_info, buy_result, position)

            if self._stop_event.is_set():
                return

            # Wait before looking for next token
            logger.debug(
                f"Waiting {self.wait_time_before_new_token} seconds before looking for next token..."
            )
            await asyncio.sleep(self.wait_time_before_new_token)

        except Exception:
            logger.exception(f"Error handling token {token_info.symbol}")
        finally:
            # Cleanup: Remove from position tasks
            if token_info.mint in self.position_tasks:
                logger.debug(f"_handle_token_wrapper() [{self._mint_prefix(token_info.mint)}] Removed position task")
                del self.position_tasks[token_info.mint]
            # Remove from queued set after processing completes (allows same token to be processed again if it comes in later)
            self._queued_tokens.discard(token_info.mint)


    async def _handle_successful_buy(
        self, token_info: TokenInfo, buy_result: TradeResult, position: Position
    ) -> None:
        """Handle successful token purchase."""
        logger.info(
            f"[{self._mint_prefix(token_info.mint)}] Successfully bought {token_info.symbol} on {token_info.platform.value} in transaction {str(buy_result.tx_signature)}"
        )
        logger.info(f"Position: {position}")
        


        # Persist to database if available
        if self.database_manager:
            try:

                await self.database_manager.update_position(position)

                # Insert buy trade
                await self.database_manager.insert_trade(
                    trade_result=buy_result,
                    mint=str(token_info.mint),
                    position_id=position.position_id,
                    trade_type="buy",
                    run_id=self.run_id,
                )

                logger.debug("Persisted buy trade to database")
            except Exception as e:
                logger.exception(f"Failed to persist buy trade to database: {e}")

        # Execute trailing exit strategy
        await self._monitor_position_until_exit(token_info, position)

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
        logger.debug(
            f"[{self._mint_prefix(token_info.mint)}] Failed buy PnL: {position.realized_pnl_sol_decimal:.6f} SOL "
            f"(fees: {pnl_dict['total_fees_raw'] / LAMPORTS_PER_SOL:.6f} SOL)"
        )


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
                await self.database_manager.update_position(position)

                # Insert failed buy trade
                await self.database_manager.insert_trade(
                    trade_result=buy_result,
                    mint=str(token_info.mint),
                    position_id=position.position_id,
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

        # Update market quality score asynchronously (event-driven)
        # Position is already marked as inactive with FAILED_BUY exit reason
        if self.market_quality_controller:
            asyncio.create_task(
                self.market_quality_controller.update_quality_score(self.run_id)
            )

    async def _monitor_position_until_exit(
        self, token_info: TokenInfo, position: Position
    ) -> None:
        """Monitor a position until exit conditions are met using event-driven architecture."""
        curve_manager = self.platform_implementations.curve_manager
        
        # Create position monitor instance
        position_monitor = PositionMonitor(
            position=position,
            token_info=token_info,
            curve_manager=curve_manager,
            seller=self.seller,
            price_check_interval=self.price_check_interval,
            database_manager=self.database_manager,
            token_listener=self.token_listener if self.enable_trade_tracking else None,
            run_id=self.run_id,
            enable_trade_tracking=self.enable_trade_tracking,
            enable_volatility_adjustment=self.enable_volatility_adjustment,
            volatility_window_seconds=self.volatility_window_seconds,
            volatility_tp_adjustments=self.volatility_tp_adjustments,
            take_profit_percentage=self.take_profit_percentage,
            market_quality_controller=self.market_quality_controller,
        )
        
        # Start monitoring - this will handle all event-driven logic
        await position_monitor.monitor()

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
                exit_strategy="trailing",
                take_profit_percentage=self.take_profit_percentage,
                stop_loss_percentage=self.stop_loss_percentage,
                trailing_stop_percentage=self.trailing_stop_percentage,
                max_hold_time=self.max_hold_time,
                max_no_price_change_time=self.max_no_price_change_time,
            )
            
            # Subscribe to trade tracking for resumed positions if enabled
            if (self.enable_trade_tracking):
                await self.token_listener.subscribe_token_trades(token_info)
                logger.debug(
                    f"[{self._mint_prefix(position.mint)}] Subscribed to trade tracking for resumed position {token_info.symbol}"
                )
            
            # Track as active and start monitoring task
            task_key = str(position.mint)
            task = asyncio.create_task(
                self._monitor_resumed_position(token_info, position)
            )
            self.position_tasks[task_key] = task
            logger.info(f"_resume_active_positions() [{self._mint_prefix(token_info.mint)}] Created position task")

    def _apply_current_exit_config(self, position: Position) -> None:
        """Override exit strategy thresholds using current bot config."""
        # Use the Position's built-in method to avoid code duplication

        
    async def _monitor_resumed_position(
        self, token_info: TokenInfo, position: Position
    ) -> None:
        """Wrapper to monitor a resumed position and cleanup on completion."""
        mint_key = str(position.mint)
        try:
            await self._monitor_position_until_exit(token_info, position)
        finally:
            # Unsubscribe from trade tracking if enabled
            if self.enable_trade_tracking and self.token_listener:
                try:
                    await self.token_listener.unsubscribe_token_trades(mint=str(position.mint))
                    logger.debug(f"[{self._mint_prefix(position.mint)}] Unsubscribed from trade tracking for resumed position")
                except Exception as e:
                    logger.debug(f"[{self._mint_prefix(position.mint)}] Failed to unsubscribe from trade tracking: {e}")
            
            # Remove from position tasks when monitoring ends or task is cancelled
            if mint_key in self.position_tasks:
                del self.position_tasks[mint_key]
                logger.info(f"[{self._mint_prefix(position.mint)}] Removed position task")
            
            # Check if we should stop the trader after all positions close
            if self._max_buys_reached and len(self.position_tasks) == 0:
                logger.info("All positions have closed after reaching max_buys limit. Stopping trader...")
                # This will cause the main loop to exit
                self._main_stop_event.set()


# Backward compatibility alias
PumpTrader = UniversalTrader  # Legacy name for backward compatibility
