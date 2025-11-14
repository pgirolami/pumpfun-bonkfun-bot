# Pre-Confirmation Sell Transactions

## Problems We'd Run Into

1. **Transaction Ordering**: Solana does NOT have explicit transaction dependencies (unlike Ethereum's nonce system). Transactions are processed independently in the order they arrive at validators. Our approach:

- **Send BUY first**, then wait 500ms, then send SELL
- **Use the same recent blockhash** for both transactions (or sequential blockhashes) so they're processed in the same slot/block if possible
- **Rely on network ordering**: If both transactions arrive at the validator in order, they should process in order
- **Risk**: Network conditions (latency, different validator paths) could cause SELL to arrive/process before BUY, causing SELL to fail with "insufficient funds" or "account not found"
- **No guarantee**: There's no Solana mechanism to ensure BUY processes before SELL - we're relying on implicit ordering through send timing and blockhash usage

2. **Blockhash Reuse**: This is NOT a problem. Solana only rejects transactions as "duplicates" if they have the same signature (same signer + same blockhash + same transaction content). Since BUY and SELL have different instructions and different accounts, they will have different signatures even with the same blockhash. We can safely reuse the same blockhash or use a fresh one.

3. **Next Block Processing**: we will get a fresh blockhash before selling - this naturally targets the next block since Solana blocks are ~400ms

4. **Token Amount Calculation**: The token amount is determined in  `_prepare_buy_order()` based on bonding curve state. If other trades occur between BUY and SELL, the price changes, but our token amount is fixed from the BUY. At the moment, SELLs liquidiate the position so we know how many tokens to sell.

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

### 2. Store position's orders tracking in database (Optional for now)

The database manager needs to be update to persist the buy & sell order fields, they can be stored as a JSON object in a string column in the database
- update_position() needs to overwrite `buy_order` and `sell_order` on every call
- get_position() and similar methods need to read both those fields
- insert_position() doesn't change because we'll insert the position before buying or selling

The `schema.sql` file should be updated to include these two NULLable columns and you should produce a migration file for existing databases. 

### 3. Modify Flow

**File**: `src/trading/platform_aware.py` - `PlatformAwareBuyer.execute()`

- The `execute()` method needs to be split into two methods:
    - a `prepare_and_send_order()`method that contains the beginning of `execute()` without wait for confirmation before returning. It should return the `BuyOrder` after setting its `buy_order.state` to `SENT` to indicate we were able to send the transaction.
    - a `process_order(buy_order:BuyOrder)` that contains the rest of the former `execute()` method and returns the same `TradeResult` as today. The balance changes will be obtained from the Transaction object returned by Solana so TokenInfo doesn't need to be passed as a parameter to the method. The state of the order will be set to `FAILED` or `CONFIRMED` based on what happened.

**File**: `src/trading/platform_aware.py` - `PlatformAwareSeller.execute()`
- Split the `execute()` method as we did in PlatformAwareBuyer(), using SellOrder objects instead of BuyOrder

**File**: `src/trading/universal_trader.py` - `_handle_token()`

- When token is detected: Create the position by calling the `Position.create_from_token_info(token_info:TokenInfo)` method 
- If the decision is made to buy the token:
    - insert the position into the database as is
    - Call `buyer.prepare_and_send_order()` to send the BUY order then
        - call the position's `update_from_buy_order(buy_order:BuyOrder)` which will set the position's `buy_order`
        - update position in database
        - no changes to creating the position monitoring and delegating

**File**: `src/trading/universal_trader.py` - `_handle_time_based_exit()`
    - remove this method and everything related to only that method

**File**: `src/trading/position_monitor.py - monitor()`

    - update the code to replace the call to `execute()` so it 
        - calls prepare_and_send_order()
        - stores the sell order in the position and persist it to the database
        - calls process_order() immediately & in a blocking fashion

    - when entering that function, it should `asyncio.create_task()` a  background task to confirm the transaction in the background using the `process_order(buy_order:BuyOrder)` blocking method
        -  If BUY confirms:
            - Update position as we do today, in particular with actual values from `balance_changes` . You should be just moving code here, not writing entirely new code
            - update position in database
        - If BUY fails:
            - if a sell wasn't sent (ie. there is no sell_order on the position) then the `monitor()` loop should exit and close the position in a way similar to universal_tradeer._handle_failed_buy()` (in fact you should probably just move the code)
            - if a sell was sent (= position.sell_order is not None)
                - if the sell order's state is still set to `SENT` then we have nothing to do because the position will close once the monitor() loop runs beyond the sell confirmation call (it blocks on that call)
                - the only other case if for the sell to be confirmed to fail in which case, it will be logged as usual and the position will be closed as it is today 
        - in both cases, this should trigger a `_process_event()` call because it may trigger a sell. It does not matter if the confirmation happens before or after a sell.
    - The loop in `monitor()` continues to work as currently and can trigger a sell based on a variety of criteria: we've just made sure it can happen before the buy confirms. For example, the time-based exit will be done through the `min_gain_time_window` : I can just set the min_gain to be 1000 and the window to be 0.5 to sell 500ms after the buy
        - We do not need to track the SELL was sent before BUY confirmed because the SELL cannot confirm successfully before the BUY confirms
    - When SELL confirms, the code path stays exactly the same as it is today

### 4. Keeping within max active positions

**File**: `src/trading/universal_trader.py`

    - Since active positions are now an actual async task, we can remove all the buy counters and just make sure that when a new token is received by the universal_trader and we want to buy it, we don't have more than `max_active_mints` tasks in self.position_tasks
    - we're keeping the max_buys parameter and we'll increment the buy_count after calling PlatformAwareBuyer.prepare_and_send_order()

## Files to Modify

1. `src/trading/universal_trader.py` - Main orchestration logic
2. `src/trading/platform_aware.py` - Return BuyOrder earlier in flow
3. `src/trading/position.py` - Support pending/expected positions
4. `src/trading/trade_order.py` - Ensure BuyOrder contains all needed data
5. `src/core/client.py` - Ensure fresh blockhash usage

## Testing Considerations

- Test with extreme fast mode (no price checks)
- Test with normal mode (price checks)
- Test BUY failure scenarios
- Test network delays and reordering
- Test multiple simultaneous tokens
- Verify token amounts match between BUY and SELL
- Monitor transaction ordering in Solana explorer