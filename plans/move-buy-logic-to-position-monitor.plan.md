# Refactoring Plan: Move Buy Logic to PositionMonitor

## Overview

Move business logic (buy decision, buy execution, monitoring, sell) from `UniversalTrader` to `PositionMonitor`, keeping infrastructure checks (capacity, circuit breaker, filters) in `UniversalTrader`. Centralize all Position database operations in `PositionMonitor`.

## Architecture Changes

**Before:**
- `UniversalTrader`: Infrastructure checks + buy execution + buy confirmation
- `PositionMonitor`: Monitoring + sell execution
- Database: Position operations split between both classes

**After:**
- `UniversalTrader`: Infrastructure checks only (capacity, circuit breaker, market quality filter)
- `PositionMonitor`: Buy decision + buy execution + monitoring + sell execution
- Database: All Position operations in `PositionMonitor`

## Implementation Steps

### Phase 1: Update PositionMonitor Interface

**File:** `src/trading/position_monitor.py`

1. Add `buyer` parameter to `__init__`:
   - Add `buyer: PlatformAwareBuyer` parameter
   - Store as `self.buyer = buyer`
   - Import `PlatformAwareBuyer` from `trading.platform_aware`

2. Add buy configuration parameters to `__init__`:
   - `buy_amount: float` - SOL amount to buy
   - `buy_slippage: float` - Buy slippage tolerance
   - `wait_time_after_creation: int` - Wait time before buying (seconds)
   - Note: `max_token_age` stays in UniversalTrader as infrastructure check
   - Note: `token_wait_timeout` is not used, ignore it

3. Add optional buy counter callbacks:
   - `on_buy_sent: Callable[[], None] | None = None` - Called when buy order is sent
   - `on_buy_confirmed: Callable[[], None] | None = None` - Called when buy confirms

4. Create the position from the TokenInfo in `__init__`:

5. Keep `buy_result_available_event`:
   - Keep `self.buy_result_available_event = asyncio.Event()` (line 80)
   - Buy stays asynchronous to monitor loop, event signals when buy confirms

### Phase 2: Create Buy Execution Method and Start Monitoring

**File:** `src/trading/position_monitor.py`

1. Create new method `_execute_buy()` that handles buy logic:

```python
async def _execute_buy(self) -> bool:
    """Execute buy order with business logic.
    
    Returns:
        True if buy was executed successfully, False if buy failed
    """
```

Implementation steps (remove wait_time and token age checks - those are infrastructure):
a. Insert position to database (if `database_manager` available)
b. Call `self.buyer.prepare_and_send_order(token_info)` to send buy order
c. Update position with `position.update_from_buy_order(buy_order, take_profit_percentage, stop_loss_percentage)`
d. Update position in database
e. Call `on_buy_sent()` callback if provided
f. Start background task to call `self.buyer.process_order(buy_order)` to confirm transaction
g. In background task: Update position with `position.update_from_buy_result(buy_result, entry_ts)`
h. In background task: Update position in database
i. In background task: Insert buy trade to database via `database_manager.insert_trade()`
j. In background task: Call `on_buy_confirmed()` callback if provided
k. In background task: Set `buy_result_available_event` to signal monitor loop

Error handling:
- If buy fails, set `position.buy_order.state = OrderState.FAILED`
- Update position with `ExitReason.FAILED_BUY`
- Update position in database
- Insert failed buy trade to database
- Set `buy_result_available_event` even on failure

2. Create new method `start_monitoring()`:

```python
async def start_monitoring(self) -> asyncio.Task:
    """Start monitoring by subscribing to trades and starting monitor loop.
    
    Returns:
        asyncio.Task for the monitor loop
    """
```

Implementation:
a. Subscribe to trade tracking via `self.token_listener.subscribe_token_trades(token_info)`
b. Insert the position into the database
c. Create async task for `self.monitor()` with done callback
d. In done callback: Unsubscribe from trade tracking via `self.token_listener.unsubscribe_token_trades()`
e. Return the task (caller can track it)

### Phase 3: Update monitor() Method

**File:** `src/trading/position_monitor.py`

Modify `monitor()` method (starting at line 93):

1. Call `_execute_buy()` at the beginning of the loop, if the position's buy order is None or is in state `CONSIDERING` 
2. Keep `buy_result_available_event` waiting logic in event loop (lines 124-126, 160-164, 173-176) - buy is asynchronous
3. Event loop waits for:
   - Time tick events
   - Trade update events (from tracker)
   - Buy result available event (when buy confirms)
4. Keep buy failure check in monitoring loop (lines 178-219) - check `buy_order.state == OrderState.FAILED` when event fires

### Phase 4: Simplify UniversalTrader._handle_token()

**File:** `src/trading/universal_trader.py`

Modify `_handle_token()` method (starting at line 688):

**Remove:**
1. Buy execution code (lines 748-767):
   - `buyer.prepare_and_send_order()` call
   - `position.update_from_buy_order()` call
   - Position database insert (line 743)
   - Position database update after buy_order (lines 761-766)
   - Buy counter increment (line 769)
   - Max buys check (lines 772-776)

2. Buy confirmation background task (lines 802-805):
   - Remove `_confirm_buy_order()` task creation

3. Trade tracking subscription (lines 718-728):
   - Move to `PositionMonitor.start_monitoring()`

**Keep:**
1. Platform validation (lines 691-696)
2. Token info database insert (lines 698-704)
3. Market quality check (lines 707-716) - infrastructure filter
4. Max active tokens check - infrastructure filter
5. Max buys check - infrastructure filter

**Update:**
1. Position creation (lines 730-739) - keep but don't insert to DB yet
2. PositionMonitor instantiation (lines 780-794) - add new parameters:
   - `buyer=self.buyer`
   - `buy_amount=self.buy_amount`
   - `buy_slippage=self.buy_slippage`
   - `wait_time_after_creation=self.wait_time_after_creation`
   - `on_buy_sent=lambda: self._increment_buy_count()`
   - `on_buy_confirmed=None` (or add if needed)
   - Note: Do NOT pass `max_token_age` or `token_wait_timeout`

3. Replace `asyncio.create_task(position_monitor.monitor())` with:
   - `monitoring_task = await position_monitor.start_monitoring()`
   - This subscribes to trades and starts monitor loop with unsubscribe callback

4. Add helper method for buy counter:
```python
def _increment_buy_count(self) -> None:
    """Increment buy counter and check max_buys limit."""
    self.buy_count += 1
    logger.info(f"Buy count: {self.buy_count}/{self.max_buys if self.max_buys else 'unlimited'}")
    if self.max_buys and self.buy_count >= self.max_buys:
        logger.info(f"Reached max_buys limit ({self.max_buys}). Will stop after all positions close...")
        self._max_buys_reached = True
        self._stop_event.set()
```

### Phase 5: Remove Dead Code from UniversalTrader

**File:** `src/trading/universal_trader.py`

1. Delete `_confirm_buy_order()` method (lines 826-881) - logic moved to `PositionMonitor._execute_buy()`
2. Delete `_handle_failed_buy()` method (lines 914-973) - logic moved to `PositionMonitor._execute_buy()`
3. Delete `_handle_successful_buy()` method (lines 883-912) - no longer used
4. Remove unused imports if any

### Phase 6: Update Position Resumption

**File:** `src/trading/universal_trader.py`

Update `_resume_active_positions()` method (starting at line 1031):

1. Update `_monitor_resumed_position()` to create PositionMonitor with buyer:
   - Pass `buyer=self.buyer` to PositionMonitor
   - Pass buy configuration parameters (buy_amount, buy_slippage, wait_time_after_creation)
   - Note: Resumed positions may not need buy logic, but include it for consistency

2. Update `_monitor_resumed_position()` to use `start_monitoring()` instead of direct `monitor()` call:
   - Replace `await self._monitor_position_until_exit(token_info, position)` 
   - With `await position_monitor.start_monitoring()` where position_monitor is created with buyer
   - Remove trade tracking subscription from `_resume_active_positions()` (line 1076) - PositionMonitor.start_monitoring() handles it

### Phase 7: Database Operations Consolidation

**File:** `src/trading/position_monitor.py`

Ensure all Position database operations are in PositionMonitor:

1. `insert_position()` - in `_execute_buy()` before buy execution
2. `update_position()` - after buy_order, after buy_result, on each monitoring event, after sell
3. `insert_trade()` for buy - in `_execute_buy()` after buy confirms
4. `insert_trade()` for sell - already in `_process_event()` (line 426)

**File:** `src/trading/universal_trader.py`

Remove Position database operations:
1. Remove `insert_position()` call from `_handle_token()` (line 743)
2. Remove `update_position()` calls related to buy (lines 763, 865)
3. Remove `insert_trade()` calls for buy trades (lines 868-874)
4. Keep `insert_token_info()` calls (line 702) - not Position-specific

### Phase 8: Token Age Check Decision

**File:** `src/trading/universal_trader.py`

Keep token age check in `_process_token_queue()` (line 649) as early infrastructure filter.

**File:** `src/trading/position_monitor.py`

Do NOT add token age check in `_execute_buy()` - it's already handled as infrastructure check in UniversalTrader.

### Phase 9: Testing & Validation

1. Test `PositionMonitor._execute_buy()`:
   - Normal buy flow
   - Buy failure handling
   - Database operations
   - Asynchronous buy confirmation

2. Test `PositionMonitor.start_monitoring()`:
   - Trade tracking subscription
   - Monitor loop starts correctly
   - Unsubscribe on completion

3. Test `UniversalTrader._handle_token()`:
   - Infrastructure checks only
   - Buy counter increment via callback

4. Integration tests:
   - Full flow: token discovery → buy → monitor → sell
   - Position resumption from database
   - Concurrent positions
   - Max_buys limit

5. Manual testing:
   - Dry-run mode
   - Live mode (small amounts)
   - Database consistency
   - Log verification

### Phase 10: Code Cleanup

1. Run `ruff format` on modified files
2. Run `ruff check --fix` on modified files
3. Update docstrings:
   - PositionMonitor: reflect new buy responsibilities
   - UniversalTrader: reflect simplified role
4. Add architecture comments explaining the split

## Key Files Modified

- `src/trading/position_monitor.py` - Add buy logic, add start_monitoring(), keep event coordination
- `src/trading/universal_trader.py` - Remove buy logic, simplify to infrastructure checks

## Migration Notes

- Position resumption: Ensure PositionMonitor handles resumed positions correctly
- Buy counter: Use callback mechanism to increment in UniversalTrader
- Database: All Position operations now in PositionMonitor
- Trade tracking: Moved from UniversalTrader to PositionMonitor.start_monitoring()
- Buy stays asynchronous: Monitor loop runs immediately, buy confirms in background and signals via event

