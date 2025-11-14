"""Event-driven position monitoring with timestamped events."""

import asyncio
from datetime import datetime
import time
from typing import Any

from database.manager import DatabaseManager
from interfaces.core import CurveManager, TokenInfo
from monitoring.base_listener import BaseTokenListener
from trading.market_quality import MarketQualityController
from trading.platform_aware import PlatformAwareSeller
from trading.position import ExitReason, Position
from utils.logger import get_logger
from utils.volatility import VolatilityCalculator, calculate_take_profit_adjustment

logger = get_logger(__name__)

INFINITY = float('inf')

class PositionMonitor:
    """Event-driven monitor for a single trading position.
    
    One instance per position. Waits for trade events and time tick events,
    evaluates exit conditions using timestamps from events, and executes exits.
    """

    def __init__(
        self,
        position: Position,
        token_info: TokenInfo,
        curve_manager: CurveManager,
        seller: PlatformAwareSeller,
        price_check_interval: int,
        token_listener: BaseTokenListener,
        # Optional dependencies
        database_manager: DatabaseManager | None = None,
        run_id: str | None = None,
        enable_volatility_adjustment: bool = False,
        volatility_window_seconds: float = 5.0,
        volatility_tp_adjustments: dict | None = None,
        take_profit_percentage: float | None = None,
        market_quality_controller: MarketQualityController | None = None,
    ):
        """Initialize position monitor.
        
        Args:
            position: Position to monitor
            token_info: Token information
            curve_manager: Curve manager for price calculations
            seller: Seller instance for executing sells
            price_check_interval: Interval in seconds for time tick events
            token_listener: Token listener for trade tracking
            database_manager: Optional database manager
            run_id: Optional run ID for database operations
            enable_volatility_adjustment: Whether volatility adjustment is enabled
            volatility_window_seconds: Volatility calculation window
            volatility_tp_adjustments: Volatility-based TP adjustments
            take_profit_percentage: Take profit percentage for volatility adjustments
            market_quality_controller: Optional market quality controller
            mint_prefix_fn: Optional function to get mint prefix for logging
        """
        self.position = position
        self.token_info = token_info
        self.curve_manager = curve_manager
        self.seller = seller
        self.price_check_interval = price_check_interval
        self.database_manager = database_manager
        self.token_listener = token_listener
        self.run_id = run_id
        self.enable_volatility_adjustment = enable_volatility_adjustment
        self.volatility_window_seconds = volatility_window_seconds
        self.volatility_tp_adjustments = volatility_tp_adjustments or {}
        self.take_profit_percentage = take_profit_percentage
        self.market_quality_controller = market_quality_controller
        self._mint_prefix = (lambda mint: str(mint)[:8])
        
        # Time tick event
        self._time_tick_event = asyncio.Event()
        self._last_time_tick_timestamp: float | None = None
        
        # Background task for time ticks
        self._time_tick_task: asyncio.Task | None = None
        
        # Volatility calculator
        self._volatility_calculator: VolatilityCalculator | None = None
        self._volatility_monitoring_started = False

    async def monitor(self) -> None:
        """Start monitoring position until exit conditions are met."""
        logger.debug(
            f"[{self._mint_prefix(self.token_info.mint)}] Starting event-driven position monitoring (check interval: {self.price_check_interval}s)"
        )
        # Get trade tracker
        tracker = self.token_listener.get_trade_tracker_by_mint(str(self.token_info.mint))    
        #Do not initialize position's last price to tracker timestamp because our time-based exit thresholds are from the time of the buy
        
        # Start time tick background task
        self._time_tick_task = asyncio.create_task(self._time_tick_loop())
        
        try:
            while self.position.is_active:
                try:
                    # Wait for either trade event or time tick event
                    time_tick_wait_task = asyncio.create_task(self._time_tick_event.wait())
                    wait_tasks = [time_tick_wait_task]
                    if tracker:
                        trade_wait_task = asyncio.create_task(tracker.price_update_event.wait())
                        wait_tasks.append(trade_wait_task)
                    
                    done, pending = await asyncio.wait(
                        wait_tasks,
                        return_when=asyncio.FIRST_COMPLETED
                    )
                    
                    # Cancel pending tasks
                    for task in pending:
                        task.cancel()
                        try:
                            await task
                        except asyncio.CancelledError:
                            pass
                    
                    # Determine which event fired by checking the event objects themselves
                    # (not the tasks, since cancelled tasks are also "done")
                    trade_timestamp: float | None = None
                    tick_timestamp: float | None = None
                    
                    # Check which event is actually set (before clearing)
                    if tracker and tracker.price_update_event.is_set():
                        trade_timestamp = tracker.get_last_update_timestamp()

                    if self._time_tick_event.is_set():
                        tick_timestamp = self._last_time_tick_timestamp
                    
                    event_timestamp = min(trade_timestamp or INFINITY,tick_timestamp or INFINITY)
                    logger.debug(f"[{self._mint_prefix(self.token_info.mint)}] Event timestamp: {datetime.fromtimestamp(event_timestamp):%Y-%m-%d %H:%M:%S.%f}")

                    # Reset both events to avoid processing stale events on next iteration
                    if tracker:
                        tracker.price_update_event.clear()
                    self._time_tick_event.clear()
                    
                    # Process the event
                    should_exit = await self._process_event(event_timestamp)
                    
                    if should_exit:
                        break
                        
                except Exception:
                    logger.exception("Error in position monitoring event loop, continuing")
            
        finally:
            # Cleanup
            if self._time_tick_task:
                self._time_tick_task.cancel()
                try:
                    await self._time_tick_task
                except asyncio.CancelledError:
                    pass
            
            # Unsubscribe from trade tracking
            try:
                await self.token_listener.unsubscribe_token_trades(mint=str(self.token_info.mint))
                logger.debug(
                    f"[{self._mint_prefix(self.token_info.mint)}] Unsubscribed from trade tracking after position monitoring ended"
                )
            except Exception as e:
                logger.exception(
                    f"[{self._mint_prefix(self.token_info.mint)}] Failed to unsubscribe from trade tracking: {e}"
                )

    async def _time_tick_loop(self) -> None:
        """Background loop that emits time tick events at regular intervals."""
        while self.position.is_active:
            try:
                await asyncio.sleep(self.price_check_interval)
                if self.position.is_active:
                    # In live mode, use time.time(). In replay mode, replay system would inject timestamp differently
                    self._last_time_tick_timestamp = time.time()
                    self._time_tick_event.set()
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Error in time tick loop, continuing")

    async def _process_event(self, event_timestamp: float) -> bool:
        """Process a single event (trade or time tick).
        
        Args:
            event_timestamp: Timestamp from the event
            
        Returns:
            True if position should exit (and monitoring should stop)
        """
        # Get current price
        current_price = await self.curve_manager.calculate_price(
            self.token_info.mint,
            self.token_info.bonding_curve
        )
        current_timestamp_ms = int(event_timestamp * 1000)
        
        # Set monitoring start time on first price (for time-based exit conditions)
        if self.position.monitoring_start_ts is None:
            self.position.set_monitoring_start_time(current_timestamp_ms)
        
        # Calculate volatility and adjust take profit if enabled
        if self.enable_volatility_adjustment:
            # Initialize volatility calculator on first valid price
            if not self._volatility_monitoring_started:
                self._volatility_calculator = VolatilityCalculator(self.volatility_window_seconds)
                self._volatility_monitoring_started = True
                logger.debug(
                    f"[{self._mint_prefix(self.token_info.mint)}] Started volatility monitoring with first price: {current_price} SOL"
                )
            
            # Add current price to volatility calculator
            self._volatility_calculator.add_price(current_price, event_timestamp)
            
            if self._volatility_calculator.has_sufficient_data(event_timestamp):
                volatility_level = self._volatility_calculator.get_volatility_level(event_timestamp)
                volatility_value = self._volatility_calculator.get_cached_volatility()
                
                if volatility_value is not None:
                    logger.debug(
                        f"[{self._mint_prefix(self.token_info.mint)}] Volatility: {volatility_value:.4f} ({volatility_level})"
                    )
                    
                    # Adjust take profit based on volatility
                    if self.position.take_profit_price and self.take_profit_percentage:
                        original_tp_price = self.position.entry_net_price_decimal * (1 + self.take_profit_percentage)
                        volatility_adjusted_take_profit_percentage = calculate_take_profit_adjustment(
                            self.take_profit_percentage,
                            volatility_level,
                            self.volatility_tp_adjustments
                        )
                        volatility_adjusted_tp_price = self.position.entry_net_price_decimal * (1 + volatility_adjusted_take_profit_percentage)
                        
                        # Only adjust if the new TP is more conservative (lower)
                        if volatility_adjusted_tp_price != self.position.take_profit_price:
                            old_tp_price = self.position.take_profit_price
                            self.position.take_profit_price = volatility_adjusted_tp_price
                            logger.info(
                                f"[{self._mint_prefix(self.token_info.mint)}] Take profit price adjusted due to {volatility_level} volatility: "
                                f"{old_tp_price} -> {volatility_adjusted_tp_price} SOL "
                                f"(reduction: {((old_tp_price - volatility_adjusted_tp_price) / old_tp_price * 100):.1f}%)"
                            )
        
        # Update position fields based on current price
        # Update highest_price if this is a new high
        if current_price > (self.position.highest_price or 0):
            self.position.highest_price = current_price
            logger.debug(f"[{self._mint_prefix(self.token_info.mint)}] New highest price: {current_price} SOL")
        
        # Update price change timestamp from trade tracker
        # The tracker tracks when price actually changes (only on reserve updates, not time ticks)
        if self.position.max_no_price_change_time is not None:
            tracker = self.token_listener.get_trade_tracker_by_mint(str(self.token_info.mint))
            if tracker:
                tracker_timestamp = tracker.get_last_price_change_timestamp()
                if tracker_timestamp is not None:
                    # Update position timestamp from tracker (only when tracker has valid timestamp)
                    if self.position.last_price_change_ts < tracker_timestamp:
                        self.position.last_price_change_ts = tracker_timestamp
                        logger.debug(f"[{self._mint_prefix(self.token_info.mint)}] Updated last_price_change_ts from tracker: {tracker_timestamp}")
        
        # Update creator tracking from trade tracker
        tracker = self.token_listener.get_trade_tracker_by_mint(str(self.token_info.mint))
        if tracker:
            creator_swaps = tracker.get_creator_swaps()
            self.position.update_creator_tracking(
                creator_swaps[0], 
                creator_swaps[1]
            )
        
        # Update position in database on every event
        if self.database_manager:
            try:
                await self.database_manager.update_position(self.position)
                logger.debug(f"[{self._mint_prefix(self.token_info.mint)}] Updated position in database")
            except Exception as e:
                logger.exception(f"Failed to update position in database: {e}")
        
        # Log current status
        pnl = self.position._get_pnl(current_price)
        logger.info(
            f"[{self._mint_prefix(self.token_info.mint)}] Position's price on chain: {current_price} SOL ({pnl['net_price_change_pct']:+.2f}%)"
        )
        
        # Check if position should be exited (using event timestamp)
        should_exit, exit_reason = self.position.should_exit(current_price, event_timestamp)
        
        if should_exit and exit_reason:
            logger.info(
                f"[{self._mint_prefix(self.token_info.mint)}] Exit condition met: {exit_reason.value}"
            )
            logger.info(
                f"[{self._mint_prefix(self.token_info.mint)}] Current onchain price: {current_price} SOL"
            )
            
            # Log PnL before exit
            pnl = self.position._get_pnl(current_price)
            logger.info(f"[{self._mint_prefix(self.token_info.mint)}] PNL: {pnl}")
            
            # Execute sell
            sell_result = await self.seller.execute(self.token_info, self.position)
            logger.info(
                f"[{self._mint_prefix(self.token_info.mint)}] Sell result: {sell_result}"
            )
            
            if sell_result.success:
                exit_ts = (sell_result.block_time or int(event_timestamp)) * 1000
                self.position.exit_ts = exit_ts
                # Close position with actual exit price
                self.position.close_position(sell_result, exit_reason)
                
                logger.info(
                    f"[{self._mint_prefix(self.token_info.mint)}] Successfully exited position: {exit_reason.value}"
                )
                
                # Persist sell trade and update position
                if self.database_manager:
                    try:
                        # Update position in database
                        await self.database_manager.update_position(self.position)
                        
                        # Insert sell trade
                        await self.database_manager.insert_trade(
                            trade_result=sell_result,
                            mint=str(self.token_info.mint),
                            position_id=self.position.position_id,
                            trade_type="sell",
                            run_id=self.run_id,
                        )
                        
                        logger.debug(
                            "Persisted sell trade and updated position in database"
                        )
                    except Exception as e:
                        logger.exception(
                            f"[{self._mint_prefix(self.token_info.mint)}] Failed to persist sell trade to database: {e}"
                        )
                
                # Log final PnL
                final_pnl = self.position._get_pnl()
                logger.info(f"[{self._mint_prefix(self.token_info.mint)}] Final net PnL: {final_pnl}")
                
                # Update market quality score asynchronously (event-driven)
                if self.market_quality_controller:
                    asyncio.create_task(
                        self.market_quality_controller.update_quality_score(
                            self.run_id
                        )
                    )
                
                return True  # Exit monitoring
            else:
                logger.error(
                    f"[{self._mint_prefix(self.token_info.mint)}] Failed to exit position & stopping monitoring: {sell_result.error_message}"
                )
                return True  # Exit monitoring on failure
        
        return False  # Continue monitoring

