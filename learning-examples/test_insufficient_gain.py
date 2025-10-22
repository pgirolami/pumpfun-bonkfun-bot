#!/usr/bin/env python3
"""
Test script for the insufficient gain exit condition.

This script tests the new exit reason that triggers when a position
hasn't gained at least X% within 2 seconds of being bought.
"""

import sys
from pathlib import Path
import time

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from trading.position import Position, ExitReason
from solders.pubkey import Pubkey
from interfaces.core import Platform

def test_insufficient_gain_exit():
    """Test the insufficient gain exit condition."""
    print("Testing Insufficient Gain Exit Condition")
    print("=" * 50)
    
    # Create a test position with min_gain_percentage = 10%
    mint = Pubkey.from_string("11111111111111111111111111111112")  # Dummy mint
    platform = Platform.PUMP_FUN
    entry_price = 0.00010000  # 0.0001 SOL
    current_time = int(time.time() * 1000)  # Current time in milliseconds
    
    position = Position.create_from_buy_result(
        mint=mint,
        platform=platform,
        entry_net_price_decimal=entry_price,
        quantity=1000.0,
        token_swapin_amount_raw=1000000000,
        entry_ts=current_time,
        transaction_fee_raw=5000,
        platform_fee_raw=1000,
        exit_strategy="tp_sl",
        buy_amount=0.001,
        total_net_sol_swapout_amount_raw=100000,  # 0.0001 SOL in lamports
        take_profit_percentage=0.5,  # 50% take profit
        stop_loss_percentage=0.2,    # 20% stop loss
        trailing_stop_percentage=None,
        max_hold_time=60,
        max_no_price_change_time=None,
        min_gain_percentage=0.1,  # 10% minimum gain required
        min_gain_time_window=2,  # 2 second time window
    )
    
    print(f"Position created with min_gain_percentage: {position.min_gain_percentage}")
    print(f"Time window: {position.min_gain_time_window} seconds")
    print(f"Entry price: {entry_price:.8f} SOL")
    print()
    
    # Test 1: Check immediately after creation (should not exit)
    print("Test 1: Immediately after creation")
    should_exit, reason = position.should_exit(entry_price)
    print(f"  Current price: {entry_price:.8f} SOL (same as entry)")
    print(f"  Should exit: {should_exit}, Reason: {reason}")
    print()
    
    # Test 2: Check after 1 second with 5% gain (should not exit yet)
    print("Test 2: After 1 second with 5% gain")
    time.sleep(1.1)  # Wait 1.1 seconds
    price_5_percent = entry_price * 1.05  # 5% gain
    should_exit, reason = position.should_exit(price_5_percent)
    print(f"  Current price: {price_5_percent:.8f} SOL (+5%)")
    print(f"  Should exit: {should_exit}, Reason: {reason}")
    print()
    
    # Test 3: Check after 2+ seconds with 5% gain (should exit - insufficient gain)
    print("Test 3: After 2+ seconds with 5% gain (insufficient)")
    time.sleep(1.1)  # Wait another 1.1 seconds (total ~2.2 seconds)
    should_exit, reason = position.should_exit(price_5_percent)
    print(f"  Current price: {price_5_percent:.8f} SOL (+5%)")
    print(f"  Should exit: {should_exit}, Reason: {reason}")
    print()
    
    # Test 4: Check with 15% gain after 2+ seconds (should not exit - sufficient gain)
    print("Test 4: After 2+ seconds with 15% gain (sufficient)")
    price_15_percent = entry_price * 1.15  # 15% gain
    should_exit, reason = position.should_exit(price_15_percent)
    print(f"  Current price: {price_15_percent:.8f} SOL (+15%)")
    print(f"  Should exit: {should_exit}, Reason: {reason}")
    print()
    
    # Test 5: Check with loss after 2+ seconds (should exit - insufficient gain)
    print("Test 5: After 2+ seconds with loss (insufficient)")
    price_loss = entry_price * 0.95  # 5% loss
    should_exit, reason = position.should_exit(price_loss)
    print(f"  Current price: {price_loss:.8f} SOL (-5%)")
    print(f"  Should exit: {should_exit}, Reason: {reason}")
    print()
    
    print("All tests completed!")

def test_edge_cases():
    """Test edge cases for the insufficient gain condition."""
    print("\nTesting Edge Cases")
    print("=" * 30)
    
    # Test with min_gain_percentage = None (should not check)
    mint = Pubkey.from_string("11111111111111111111111111111112")
    platform = Platform.PUMP_FUN
    entry_price = 0.00010000
    current_time = int(time.time() * 1000)
    
    position = Position.create_from_buy_result(
        mint=mint,
        platform=platform,
        entry_net_price_decimal=entry_price,
        quantity=1000.0,
        token_swapin_amount_raw=1000000000,
        entry_ts=current_time,
        transaction_fee_raw=5000,
        platform_fee_raw=1000,
        exit_strategy="tp_sl",
        buy_amount=0.001,
        total_net_sol_swapout_amount_raw=100000,
        take_profit_percentage=0.5,
        stop_loss_percentage=0.2,
        trailing_stop_percentage=None,
        max_hold_time=60,
        max_no_price_change_time=None,
        min_gain_percentage=None,  # No minimum gain check
        min_gain_time_window=2,  # 2 second time window (not used when disabled)
    )
    
    print(f"Position with min_gain_percentage: {position.min_gain_percentage}")
    
    # Wait 3 seconds and check with no gain
    time.sleep(3.1)
    should_exit, reason = position.should_exit(entry_price)
    print(f"After 3+ seconds with no gain: Should exit: {should_exit}, Reason: {reason}")
    
    print("Edge case tests completed!")

if __name__ == "__main__":
    test_insufficient_gain_exit()
    test_edge_cases()
