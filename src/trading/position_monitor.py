"""Event-driven position monitoring with timestamped events."""

import asyncio
from datetime import datetime
from time import time as current_time
import time
from typing import Any, Callable

from database.manager import DatabaseManager
from interfaces.core import CurveManager, TokenInfo
from monitoring.base_listener import BaseTokenListener
from trading.base import TradeResult
from trading.market_quality import MarketQualityController
from trading.platform_aware import PlatformAwareBuyer, PlatformAwareSeller
from trading.position import ExitReason, Position
from trading.trade_order import BuyOrder, OrderState
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
        buyer: PlatformAwareBuyer,
        seller: PlatformAwareSeller,
        price_check_interval: int,
        token_listener: BaseTokenListener,
        # Buy configuration
        buy_amount: float,
        buy_slippage: float,
        wait_time_after_creation: int,
        max_buy_time: float | None = None,
        # Optional dependencies
        database_manager: DatabaseManager | None = None,
        run_id: str | None = None,
        enable_volatility_adjustment: bool = False,
        volatility_window_seconds: float = 5.0,
        volatility_tp_adjustments: dict | None = None,
        take_profit_percentage: float | None = None,
        stop_loss_percentage: float | None = None,
        market_quality_controller: MarketQualityController | None = None,
        # Optional callbacks
        on_buy_sent: Callable[[], None] | None = None,
        on_buy_confirmed: Callable[[], None] | None = None,
    ):
        """Initialize position monitor.
        
        Args:
            position: Position to monitor
            token_info: Token information
            curve_manager: Curve manager for price calculations
            buyer: Buyer instance for executing buys
            seller: Seller instance for executing sells
            price_check_interval: Interval in seconds for time tick events
            token_listener: Token listener for trade tracking
            buy_amount: SOL amount to buy
            buy_slippage: Buy slippage tolerance
            wait_time_after_creation: Wait time before buying (seconds)
            database_manager: Optional database manager
            run_id: Optional run ID for database operations
            enable_volatility_adjustment: Whether volatility adjustment is enabled
            volatility_window_seconds: Volatility calculation window
            volatility_tp_adjustments: Volatility-based TP adjustments
            take_profit_percentage: Take profit percentage for volatility adjustments
            stop_loss_percentage: Stop loss percentage
            market_quality_controller: Optional market quality controller
            on_buy_sent: Optional callback called when buy order is sent
            on_buy_confirmed: Optional callback called when buy confirms
        """
        self.position = position
        self.token_info = token_info
        self.curve_manager = curve_manager
        self.buyer = buyer
        self.seller = seller
        self.price_check_interval = price_check_interval
        self.database_manager = database_manager
        self.token_listener = token_listener
        self.run_id = run_id
        self.buy_amount = buy_amount
        self.buy_slippage = buy_slippage
        self.wait_time_after_creation = wait_time_after_creation
        self.max_buy_time = max_buy_time
        self.take_profit_percentage = take_profit_percentage
        self.stop_loss_percentage = stop_loss_percentage
        self.enable_volatility_adjustment = enable_volatility_adjustment
        self.volatility_window_seconds = volatility_window_seconds
        self.volatility_tp_adjustments = volatility_tp_adjustments or {}
        self.market_quality_controller = market_quality_controller
        self.on_buy_sent = on_buy_sent
        self.on_buy_confirmed = on_buy_confirmed
        self._mint_prefix = (lambda mint: str(mint)[:8])
        
        # Buy result available event (set when BUY transaction result is available)
        self.buy_result_available_event = asyncio.Event()
        
        # Time tick event
        self._time_tick_event = asyncio.Event()
        self._last_time_tick_timestamp: float | None = None
        
        # Background task for time ticks
        self._time_tick_task: asyncio.Task | None = None
        
        # Volatility calculator
        self._volatility_calculator: VolatilityCalculator | None = None
        self._volatility_monitoring_started = False

    async def start_monitoring(self) -> asyncio.Task:
        """Start monitoring by subscribing to trades, inserting position to DB, and starting monitor loop.
        
        This should be called as soon as PositionMonitor is created. It:
        - Subscribes to trade tracking
        - Inserts position to database
        - Starts the monitor() task with unsubscribe callback
        
        Returns:
            asyncio.Task for the monitor loop
        """
        # Subscribe to trade tracking
        try:
            await self.token_listener.subscribe_token_trades(self.token_info)
            logger.info(
                f"[{self._mint_prefix(self.token_info.mint)}] Subscribed to trade tracking for {self.token_info.symbol}"
            )
        except Exception as e:
            logger.exception(
                f"[{self._mint_prefix(self.token_info.mint)}] Failed to subscribe to trade tracking"
            )
            raise
        
        # Insert position to database
        if self.database_manager:
            try:
                await self.database_manager.insert_position(self.position)
                # logger.info(f"[{self._mint_prefix(self.token_info.mint)}] Persisted position to database")
            except Exception as e:
                logger.exception(f"Failed to persist position to database: {e}")
        
        # Create async task for monitor() with done callback to unsubscribe
        task = asyncio.create_task(self.monitor_with_cleanup())
        return task

    async def monitor_with_cleanup(self) -> None:
        """Monitor position with cleanup on completion.
        
        This method wraps monitor() and ensures trade tracking is unsubscribed
        when monitoring ends, regardless of how it ends (normal exit, exception, cancellation).
        """
        try:
            await self.monitor()
        finally:
            # Unsubscribe from trade tracking when monitoring ends
            try:
                await self.token_listener.unsubscribe_token_trades(mint=str(self.token_info.mint))
                logger.info(
                    f"[{self._mint_prefix(self.token_info.mint)}] Unsubscribed from trade tracking after position monitoring ended"
                )
            except Exception as e:
                logger.exception(
                    f"[{self._mint_prefix(self.token_info.mint)}] Failed to unsubscribe from trade tracking: {e}"
                )

    async def _send_buy(self) -> bool:
        """Execute buy order with business logic.
        
        This method checks business rules (like wait_time_after_creation) and executes
        the buy if conditions are met. It should be called at the beginning of each
        monitoring loop iteration to allow delayed buys.
        
        Returns:
            True if buy was executed successfully, False if buy failed or should be delayed
        """
        # logger.info(f"[{self._mint_prefix(self.token_info.mint)}] _send_buy() called")

        try:
            logger.info(
                f"[{self._mint_prefix(self.token_info.mint)}] Buying {self.buy_amount:.6f} SOL worth of {self.token_info.symbol} ({str(self.token_info.mint)}) on {self.token_info.platform.value}..."
            )
            
            # Send buy order
            buy_order = await self.buyer.prepare_and_send_order(self.token_info)

            self.position.update_from_buy_order(
                buy_order=buy_order,
                take_profit_percentage=self.take_profit_percentage,
                stop_loss_percentage=self.stop_loss_percentage
            )
            
            # Update position in database with buy_order
            if self.database_manager:
                try:
                    await self.database_manager.update_position(self.position)
                    # logger.debug("Persisted position with buy_order to database")
                except Exception as e:
                    logger.exception(f"Failed to persist position to database: {e}")
            
            # Call on_buy_sent callback if provided
            if self.on_buy_sent:
                self.on_buy_sent()
            
            # Start background task to confirm the BUY transaction
            # This runs in parallel with the monitoring loop so we can monitor/sell while buy confirms
            asyncio.create_task(self._confirm_buy_order(buy_order))
            
            return buy_order.state != OrderState.UNSENT
            
        except Exception as e:
            logger.exception(
                f"[{self._mint_prefix(self.token_info.mint)}] Error in buy execution: {e}"
            )
            # Set buy_order state to FAILED on exception
            if self.position.buy_order:
                self.position.buy_order.state = OrderState.FAILED
            else:
                # Create a failed buy order if none exists
                self.position.buy_order = BuyOrder(
                    token_info=self.token_info,
                    sol_amount_raw=int(self.buy_amount * 1_000_000_000),
                    state=OrderState.FAILED
                )
            
            # Update position with failed buy
            self.position.is_active = False
            self.position.exit_reason = ExitReason.FAILED_BUY
            
            # Calculate realized PnL for failed buy (loss from fees only)
            pnl_dict = self.position._get_pnl(buy_order.token_price_sol)
            self.position.realized_pnl_sol_decimal = pnl_dict["realized_pnl_sol_decimal"]
            self.position.realized_net_pnl_sol_decimal = pnl_dict["realized_net_pnl_sol_decimal"]
            
            # Update position in database
            if self.database_manager:
                try:
                    await self.database_manager.update_position(self.position)
                except Exception as e:
                    logger.exception(f"Failed to update position in database: {e}")
            
            # Signal that result is available (even if it's an error)
            self.buy_result_available_event.set()
            
            #Return true to indicate we should exit monitoring
            return True

    async def _confirm_buy_order(self, buy_order:BuyOrder) -> None:
        """Background task to confirm BUY transaction and update position.
        
        Args:
            buy_order: BuyOrder that was sent (state should be SENT)
        """
        buy_result: TradeResult | None = None
        try:
            # Process buy order (confirms transaction and analyzes balance changes)
            logger.debug(
                f"[{self._mint_prefix(self.token_info.mint)}] Going to process {buy_order}"
            )
            buy_result = await self.buyer.process_order(self.position)
            entry_ts = int(current_time() * 1000)
            logger.info(
                f"[{self._mint_prefix(self.token_info.mint)}] Buy result is {buy_result}"
            )
            
            # Update position with actual values from TradeResult
            # The buy_order state is already set to CONFIRMED in process_order()
            self.position.update_from_buy_result(buy_result, entry_ts)
            
        except Exception as e:
            logger.exception(
                f"[{self._mint_prefix(self.token_info.mint)}] Error in buy confirmation"
            )
            # Set buy_order state to FAILED on exception
            buy_order.state = OrderState.FAILED
            self.position.is_active = False
            self.position.exit_reason = ExitReason.FAILED_BUY
            
            # Calculate realized PnL for failed buy (loss from fees only)
            pnl_dict = self.position._get_pnl(buy_order.token_price_sol)
            self.position.realized_pnl_sol_decimal = pnl_dict["realized_pnl_sol_decimal"]
            self.position.realized_net_pnl_sol_decimal = pnl_dict["realized_net_pnl_sol_decimal"]
        finally:
            # Update position in database
            if self.database_manager:
                try:
                    await self.database_manager.update_position(self.position)
                    
                    # Insert buy trade (successful or failed)
                    if buy_result:
                        await self.database_manager.insert_trade(
                            trade_result=buy_result,
                            mint=str(self.token_info.mint),
                            position_id=self.position.position_id,
                            trade_type="buy",
                            run_id=self.run_id,
                        )
                        logger.debug("Persisted buy trade to database")
                except Exception as e:
                    logger.exception(f"Failed to persist buy trade to database: {e}")
            
            # Call on_buy_confirmed callback if provided
            if self.on_buy_confirmed:
                self.on_buy_confirmed()
            
            # Signal that result is available (even if it's an error)
            self.buy_result_available_event.set()

    async def monitor(self) -> None:
        """Start monitoring position until exit conditions are met."""
        # logger.info(
        #     f"[{self._mint_prefix(self.token_info.mint)}] Starting event-driven position monitoring (check interval: {self.price_check_interval}s)"
        # )
        
        # Get trade tracker
        tracker = self.token_listener.get_trade_tracker_by_mint(str(self.token_info.mint))    
        #Do not initialize position's last price to tracker timestamp because our time-based exit thresholds are from the time of the buy
        
        # Start time tick background task
        self._time_tick_task = asyncio.create_task(self._time_tick_loop())
        buy_confirmation_running = False        
        # logger.info(
        #     f"[{self._mint_prefix(self.token_info.mint)}] Starting monitoring loop"
        # )
        try:
            while True:
                
                try:
                    # Wait for either trade event, time tick event, or buy result available event
                    time_tick_wait_task = asyncio.create_task(self._time_tick_event.wait())
                    wait_tasks = [time_tick_wait_task]
                    if tracker:
                        trade_wait_task = asyncio.create_task(tracker.price_update_event.wait())
                        wait_tasks.append(trade_wait_task)

                    # Check if buy_order exists and state is SENT (background BUY confirmation task is running)
                    buy_confirmation_running = (
                        self.position.buy_order is not None 
                        and self.position.buy_order.state == OrderState.SENT
                    )
                    if buy_confirmation_running:
                        buy_result_wait_task = asyncio.create_task(self.buy_result_available_event.wait())
                        wait_tasks.append(buy_result_wait_task)

                    # logger.info(
                    #     f"[{self._mint_prefix(self.token_info.mint)}] monitor() loop waiting for one of {len(wait_tasks)} events"
                    # )
                    done, pending = await asyncio.wait(
                        wait_tasks,
                        return_when=asyncio.FIRST_COMPLETED
                    )
                    # logger.info(
                    #     f"[{self._mint_prefix(self.token_info.mint)}] monitor() loop running because an event fired"
                    # )
                    
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
                    buy_result_available = False
                    
                    # Check which event is actually set (before clearing)
                    if tracker and tracker.price_update_event.is_set():
                        trade_timestamp = tracker.get_last_update_timestamp()

                    if self._time_tick_event.is_set():
                        tick_timestamp = self._last_time_tick_timestamp
                    
                    if self.buy_result_available_event.is_set():
                        buy_result_available = True

                    event_timestamp = min(trade_timestamp or INFINITY, tick_timestamp or INFINITY,time.time())
                    # logger.info(f"[{self._mint_prefix(self.token_info.mint)}] Event timestamp: {datetime.fromtimestamp(event_timestamp):%Y-%m-%d %H:%M:%S.%f}")

                    # Reset events to avoid processing stale events on next iteration
                    if tracker:
                        tracker.price_update_event.clear()
                    self._time_tick_event.clear()
                    if buy_result_available:
                        self.buy_result_available_event.clear()
                    
                    # Check for failed buy before processing exit conditions
                    if self.position.buy_order and self.position.buy_order.state == OrderState.FAILED:
                        # BUY failed and no sell was sent - exit monitoring and close position
                        logger.info(
                            f"[{self._mint_prefix(self.token_info.mint)}] Buy order failed, no sell was sent. Closing position."
                        )
                        # Close position similar to _handle_failed_buy()
                        self.position.is_active = False
                        self.position.exit_reason = ExitReason.FAILED_BUY
                        # Calculate realized PnL for failed buy (loss from fees only)
                        pnl_dict = self.position._get_pnl(self.position.buy_order.token_price_sol)
                        self.position.realized_pnl_sol_decimal = pnl_dict["realized_pnl_sol_decimal"]
                        self.position.realized_net_pnl_sol_decimal = pnl_dict["realized_net_pnl_sol_decimal"]

# SLIPPAGE ERRORS ARE MESSED UP HERE
#              transaction_fee_raw = 45000
#                 platform_fee_raw = 1738605
#                      tip_fee_raw = 5000
#        rent_exemption_amount_raw = 2039280
#      unattributed_sol_amount_raw = 0
#         realized_pnl_sol_decimal = 0.0
#     realized_net_pnl_sol_decimal = 0.0
#                       buy_amount = 0.1
# total_net_sol_swapout_amount_raw = 0
#  total_net_sol_swapin_amount_raw = 139088433
#     total_sol_swapout_amount_raw = -27000
#      total_sol_swapin_amount_raw = 139366108

                        # Update position in database
                        if self.database_manager:
                            try:
                                await self.database_manager.update_position(self.position)
                            except Exception as e:
                                logger.exception(f"Failed to update position in database: {e}")
                            
                        if self.position.sell_order is None:
                            return True  # Exit monitoring
                        else:
                            # BUY failed but sell was already sent - continue monitoring (sell will fail naturally)
                            logger.warning(
                                f"[{self._mint_prefix(self.token_info.mint)}] Buy order failed but sell was already sent. Continuing monitoring."
                            )
                    
                    # Process the event
                    should_exit = await self._process_event(event_timestamp)
                    
                    if should_exit:
                        break
                        
                except Exception:
                    logger.exception("Error in position monitoring event loop, continuing")

            # logger.info(
            #     f"[{self._mint_prefix(self.token_info.mint)}] monitor() loop is ending"
            # )

        finally:
            # Cleanup
            if self._time_tick_task:
                self._time_tick_task.cancel()
                try:
                    await self._time_tick_task
                except asyncio.CancelledError:
                    pass
            
            # Note: Unsubscribe from trade tracking is now handled in start_monitoring() cleanup callback
            logger.debug(
                f"[{self._mint_prefix(self.token_info.mint)}] monitor() loop is finished"
            )


    async def _time_tick_loop(self) -> None:
        """Background loop that emits time tick events at regular intervals."""
        # logger.info(f"[{self._mint_prefix(self.token_info.mint)}] _time_tick_loop() starting")
        while self.position.is_active:
            try:
                await asyncio.sleep(self.price_check_interval)
                self._last_time_tick_timestamp = time.time()
                self._time_tick_event.set()
                # logger.info(f"[{self._mint_prefix(self.token_info.mint)}] _time_tick_loop() has set the event and will sleep again for {self.price_check_interval} seconds")

            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Error in time tick loop, continuing")
        # logger.info(f"[{self._mint_prefix(self.token_info.mint)}] _time_tick_loop() is exiting")

    async def _process_event(self, event_timestamp: float) -> bool:
        """Process a single event (trade or time tick).
        
        Args:
            event_timestamp: Timestamp from the event
            
        Returns:
            True if position should exit (and monitoring should stop)
        """
        # logger.info(f"[{self._mint_prefix(self.token_info.mint)}] _process_event() called")

        # Check max buy time
        if self.max_buy_time:
            # Max buy time is based on token creation time (total time since token was created)
            elapsed_time = (time.time() - self.position.token_creation_timestamp)
            if elapsed_time >= self.max_buy_time:
                logger.info(
                    f"[{self._mint_prefix(self.token_info.mint)}] elapsed_time {elapsed_time} >= max_buy_time {self.max_buy_time} "
                    f"(start_time={datetime.fromtimestamp(self.position.token_creation_timestamp):%Y-%m-%d %H:%M:%S.%f})"
                )
                return True, ExitReason.MAX_BUY_TIME
            # logger.info(f"[{self._mint_prefix(self.token_info.mint)}] elapsed_time {elapsed_time} < max_buy_time {self.max_buy_time}")

        # Handle buy
        # Execute buy at the beginning of each loop iteration if needed
        # This allows business rules (like wait_time_after_creation) to delay the buy
        if self.position.buy_order is None or (self.position.buy_order and self.position.buy_order.state in [OrderState.UNSENT, OrderState.FAILED]):
            buy_order_string="has" if self.position.buy_order else "does not have"
            buy_order_state_string = f"State is {self.position.buy_order.state.value}." if self.position.buy_order else ""
            logger.info(f"[{self._mint_prefix(self.token_info.mint)}] Position {buy_order_string} a buy order. {buy_order_state_string}")

            # Check wait_time_after_creation business rule
            if self.wait_time_after_creation > 0:
                # Calculate time since token creation
                time_since_creation = current_time() - self.token_info.creation_timestamp
                
                if time_since_creation < self.wait_time_after_creation:
                    # Not ready to buy yet, wait more
                    logger.info(
                        f"[{self._mint_prefix(self.token_info.mint)}] It's too early to buy "
                        f"{time_since_creation:.1f}s < {self.wait_time_after_creation}s"
                    )
                    return False
                else:
                    logger.info(f"[{self._mint_prefix(self.token_info.mint)}] wait_time_after_creation is 0 or less, so buying immediately")

            buy_executed = await self._send_buy()
            if not buy_executed:
                # Buy failed, maybe later
                logger.error(
                    f"[{self._mint_prefix(self.token_info.mint)}] Buy execution failed, maybe later"
                )
                return False
        # else:
        #     logger.info(f"[{self._mint_prefix(self.token_info.mint)}] Buy in state {self.position.buy_order.state}, not considering buy")



        if self.position.buy_order is not None and self.position.buy_order.state == OrderState.UNSENT:
            logger.info(
                f"[{self._mint_prefix(self.token_info.mint)}] Buy hasn't been sent yet, not even considering a sell"
            )
            return False
            
        # Get current price
        current_price = await self.curve_manager.calculate_price(
            self.token_info.mint,
            self.token_info.bonding_curve
        )
        current_timestamp_ms = int(event_timestamp * 1000)
        
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
                    if self.position.last_price_change_ts is None or self.position.last_price_change_ts < tracker_timestamp:
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
        

        logger.debug(f"[{self._mint_prefix(self.token_info.mint)}] Position: {self.position}")

        # Log current status
        pnl = self.position._get_pnl(current_price)
        logger.info(
            f"[{self._mint_prefix(self.token_info.mint)}] Position's price on chain: {current_price} SOL ({pnl['net_price_change_pct']}%)"
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
            
            # Execute sell using new flow: prepare_and_send_order then process_order
            sell_order = await self.seller.prepare_and_send_order(self.token_info, self.position)
            self.position.update_from_sell_order(sell_order)
            # Update position in database before calling process_order()
            if self.database_manager:
                try:
                    await self.database_manager.update_position(self.position)
                except Exception as e:
                    logger.exception(f"Failed to update position in database: {e}")
            
            # Process sell order (blocking)
            if sell_order.state == OrderState.FAILED:
                logger.error(
                    f"[{self._mint_prefix(self.token_info.mint)}] Sell order failed already, exiting monitoring"
                )
                return True # Exit monitoring on failure
            
            sell_result = await self.seller.process_order(self.position)
            logger.info(
                f"[{self._mint_prefix(self.token_info.mint)}] Sell result: {sell_result}"
            )
            
            if sell_result.success:
                exit_ts = int(event_timestamp*1000)
                self.position.exit_ts = exit_ts
                # Close position with actual exit price
                self.position.close_position(sell_result, exit_reason)
                
                logger.info(
                    f"[{self._mint_prefix(self.token_info.mint)}] Successfully exited position: {exit_reason.value}"
                )
                
                # Persist sell trade and update position
                if self.database_manager:
                    try:
                        
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
                final_pnl = self.position._get_pnl(self.position.exit_net_price_decimal)
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
        
            try:
                await self.database_manager.update_position(self.position)
            except Exception as e:
                logger.exception(f"Failed to update position in database: {e}")

        return False  # Continue monitoring

