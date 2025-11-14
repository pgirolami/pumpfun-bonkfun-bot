# Pre-Confirmation Sell Transactions

## Problems We'd Run Into

1. **Transaction Ordering**: Solana does NOT have explicit transaction dependencies (unlike Ethereum's nonce system). Transactions are processed independently in the order they arrive at validators. Our approach:

- **Send BUY first**, then wait 500ms, then send SELL
- **Use the same recent blockhash** for both transactions (or sequential blockhashes) so they're processed in the same slot/block if possible
- **Rely on network ordering**: If both transactions arrive at the validator in order, they should process in order
- **Risk**: Network conditions (latency, different validator paths) could cause SELL to arrive/process before BUY, causing SELL to fail with "insufficient funds" or "account not found"
- **No guarantee**: There's no Solana mechanism to ensure BUY processes before SELL - we're relying on implicit ordering through send timing and blockhash usage

2. **Blockhash Reuse**: This is NOT a problem. Solana only rejects transactions as "duplicates" if they have the same signature (same signer + same blockhash + same transaction content). Since BUY and SELL have different instructions and different accounts, they will have different signatures even with the same blockhash. We will reuse the cached blockhash for both transactions.

3. **Token Amount Calculation**: For pump.fun, the token amount is fixed in the buy instruction (unlike Raydium where it's calculated dynamically). The token amount is determined in `_prepare_buy_order()` based on bonding curve state at that moment. If other trades occur between BUY and SELL, the price changes, but our token amount is fixed from the BUY. At the moment, SELLs liquidate the position so we know how many tokens to sell from the buy order.

4. **Position Tracking**: Currently positions are created after BUY confirmation. We need to track "pending" positions with expected token amounts before confirmation.

5. **Error Handling**: If BUY fails, the pending SELL will also fail. We need to track and handle this gracefully without creating orphaned positions.

6. **Account Creation**: For pump.fun, the token account may be created in the BUY transaction. The SELL must reference the same account (derivable from mint + wallet).

## Implementation Plan

### 1. Add Position State Tracking

**File**: `src/trading/trade_order.py`
- Create an `OrderState` enum: 
    - `SENT` : when the decision to buy has been made and the transaction has been sent
    - `CONFIRMED` : when the buy transaction was confirmed
    - `FAILED` : when the buy failed
- Add a `state` field to the base `Order` class with this type

**File**: `src/trading/position.py`

- Keep the `Position.is_active` field as it is today
- Add `buy_order: BuyOrder | None` field to track BUY transaction 
- Add `sell_order: SellOrder | None` field to track SELL transaction
- Make the `_get_pnl()` method and other similar methods compatible with the fact that some fields are still None before the buy is confirmed
- Add a `Position.create_from_token_info(token_info:TokenInfo)` method to be created and that will populate as many fields as possible. `active` should remain False
- Add a `Position.update_from_buy_order(buy_order:BuyOrder)` method that will set the fields that can be. It will most likely be more or less the existing `create_from_buy_result()` method
- Add a `Position.update_from_sell_order(sell_order:SellOrder)` method that will store the `sell_order` in the position and update various fields (similar to how position is updated from sell results today)

### 2. Store position's orders tracking in database (Optional for now)

The database manager needs to be update to persist the buy & sell order fields, they can be stored as a JSON object in a string column in the database
- update_position() needs to overwrite `buy_order` and `sell_order` on every call
- get_position() and similar methods need to read both those fields
- insert_position() doesn't change because we'll insert the position before buying or selling

The `schema.sql` file should be updated to include these two NULLable columns and you should produce a migration file for existing databases. 

### 3. Modify Flow

**File**: `src/trading/platform_aware.py` - `PlatformAwareBuyer.execute()`

- The `execute()` method needs to be split into two methods:
    - a `prepare_and_send_order(token_info: TokenInfo)` method that contains the beginning of `execute()` without wait for confirmation before returning. It should return the `BuyOrder` after setting its `buy_order.state` to `SENT` to indicate we were able to send the transaction.
    - a `process_order(buy_order: BuyOrder)` that contains the rest of the former `execute()` method and returns the same `TradeResult` as today. The balance changes will be obtained from the Transaction object returned by Solana. TokenInfo is available from `buy_order.token_info`, so it doesn't need to be passed as a separate parameter. The state of the order will be set to `FAILED` or `CONFIRMED` based on what happened.

**File**: `src/trading/platform_aware.py` - `PlatformAwareSeller.execute()`
- Split the `execute()` method as we did in PlatformAwareBuyer(), using SellOrder objects instead of BuyOrder:
    - `prepare_and_send_order(token_info: TokenInfo, position: Position)` - prepares and sends sell order, returns `SellOrder` with state `SENT`
    - `process_order(sell_order: SellOrder)` - confirms and processes sell order, returns `TradeResult`. TokenInfo is available from `sell_order.token_info`.

**File**: `src/trading/universal_trader.py` - `_handle_token()`

- When token is detected and decision is made to buy the token:
    - Create the position by calling `Position.create_from_token_info(token_info: TokenInfo)` method (this creates a minimal position with `is_active=False`)
    - Insert the position into the database
    - Call `buyer.prepare_and_send_order(token_info)` to send the BUY order, which returns a `BuyOrder` with state `SENT`
    - Call the position's `update_from_buy_order(buy_order: BuyOrder)` which will set the position's `buy_order` field and other fields
    - Update position in database
    - Create the `PositionMonitor` instance (needed for background BUY confirmation task)
    - Start position monitoring as a background task (don't await it, don't wait for BUY confirmation)
    - Start a background task to confirm the BUY transaction (calls `buyer.process_order(buy_order)`, passes `PositionMonitor` instance)

**File**: `src/trading/position_monitor.py` - `__init__()`

    - Add a `buy_result_available_event: asyncio.Event` field that will be set when the BUY transaction result is available (success or failure)
    - This event will be set by the background BUY confirmation task in `universal_trader.py`

**File**: `src/trading/position_monitor.py` - `monitor()`

    - When entering `monitor()`, check if `position.buy_order` exists and if its state is `SENT`:
        - If so, the background BUY confirmation task is already running (started in `_handle_token()`)
        - The background task will update the position when BUY confirms/fails
    - The monitoring loop should wait for three types of events:
        - Trade events (price updates from tracker)
        - Time tick events (periodic price checks)
        - Buy result available event (`buy_result_available_event`) - when BUY transaction result is available (success or failure)
    - When any of these events fire, process the event and check exit conditions:
        - **Before checking exit conditions or sending a sell**, check if `position.buy_order.state == OrderState.FAILED`:
            - If BUY failed and no sell was sent (`position.sell_order is None`), exit the monitoring loop and close the position (similar to `universal_trader._handle_failed_buy()`)
            - If BUY failed but sell was already sent, continue monitoring (sell will fail naturally)
    - When exit condition is met:
        - Call `seller.prepare_and_send_order(token_info, position)` to send the SELL order, which returns a `SellOrder` with state `SENT`
        - Call `update_from_sell_order(sell_order: SellOrder)` which will store the `sell_order` in the position (`position.sell_order = sell_order`) and update various fields (as is done today already)
        - Update position in database before calling `process_order()`
        - Call `seller.process_order(sell_order)` immediately in a blocking fashion
        - Handle sell result as today (close position, update database, etc.)
    - The sell can be sent before BUY confirms (when exit conditions are met)
    - When SELL confirms, the code path stays exactly the same as it is today

**File**: `src/trading/universal_trader.py` - Background BUY confirmation task

    - This background task (created in `_handle_token()`) calls `buyer.process_order(buy_order)` and has access to the `PositionMonitor` instance:
            - If BUY confirms:
                - Update position with actual values from `balance_changes` (move code from current `_handle_successful_buy()`)
                - Set `buy_order.state = OrderState.CONFIRMED`
                - Update position in database
            - If BUY fails:
                - Set `buy_order.state = OrderState.FAILED`
                - Update position in database
                - The `monitor()` loop will detect this and exit appropriately
            - In both cases (success or failure), set `position_monitor.buy_result_available_event.set()` to signal that BUY processing is complete and result is available


### 4. Keeping within max active positions

**File**: `src/trading/universal_trader.py`

    - Since active positions are now an actual async task, we can remove all the buy counters and just make sure that when a new token is received by the universal_trader and we want to buy it, we don't have more than `max_active_mints` tasks in self.position_tasks
    - we're keeping the max_buys parameter and we'll increment the buy_count after calling PlatformAwareBuyer.prepare_and_send_order()

## Files to Modify

1. `src/trading/universal_trader.py` - Main orchestration logic
2. `src/trading/platform_aware.py` - Return BuyOrder earlier in flow
3. `src/trading/position.py` - Support pending/expected positions
4. `src/trading/trade_order.py` - Ensure BuyOrder contains all needed data
5. `src/trading/position_monitor.py` - Add buy_result_available_event and handle pre-confirmation sells

## Logging and Monitoring

- Log when SELL is sent before BUY confirmation (for tracking and debugging)
- Track success rate of pre-confirmation sells
- Log order state transitions (SENT → CONFIRMED/FAILED)
- Monitor transaction ordering in Solana explorer to verify BUY processes before SELL

## Testing Considerations

- Test with extreme fast mode (no price checks)
- Test with normal mode (price checks)
- Test BUY failure scenarios:
    - BUY fails before SELL is sent → position should close without selling
    - BUY fails after SELL is sent → SELL should fail naturally, position closes
- Test network delays and reordering
- Test multiple simultaneous tokens
- Verify token amounts match between BUY and SELL (token amount is fixed in pump.fun buy instruction)
- Monitor transaction ordering in Solana explorer
- Test that SELL cannot confirm successfully before BUY confirms
- Verify position state transitions (pending → active → closed)
- Test database persistence of buy_order and sell_order fields