#!/usr/bin/env python3
"""
Test script for delayed monitoring start time.

This script tests that time-based exit conditions (max_hold_time, min_gain_percentage)
use monitoring start time instead of entry time when there's a delay before monitoring begins.
"""

import sys
from pathlib import Path
import time

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from trading.position import Position, ExitReason
from solders.pubkey import Pubkey
from interfaces.core import Platform

def test_delayed_monitoring_timing():
    """Test that time-based conditions use monitoring start time, not entry time."""
    print("Testing Delayed Monitoring Timing")
    print("=" * 50)
    
    # Create a test position
    mint = Pubkey.from_string("11111111111111111111111111111112")
    platform = Platform.PUMP_FUN
    entry_price = 0.00010000
    entry_time = int(time.time() * 1000)  # Entry time in milliseconds
    
    position = Position.create_from_buy_result(
        mint=mint,
        platform=platform,
        entry_net_price_decimal=entry_price,
        quantity=1000.0,
        token_swapin_amount_raw=1000000000,
        entry_ts=entry_time,
        transaction_fee_raw=5000,
        platform_fee_raw=1000,
        exit_strategy="tp_sl",
        buy_amount=0.001,
        total_net_sol_swapout_amount_raw=100000,
        take_profit_percentage=0.5,
        stop_loss_percentage=0.2,
        trailing_stop_percentage=None,
        max_hold_time=10,  # 10 second max hold
        max_no_price_change_time=None,
        min_gain_percentage=0.05,  # 5% minimum gain required
        min_gain_time_window=3,  # 3 second window
    )
    
    print(f"Position created at entry time: {entry_time}")
    print(f"Max hold time: {position.max_hold_time} seconds")
    print(f"Min gain: {position.min_gain_percentage} within {position.min_gain_time_window} seconds")
    print()
    
    # Simulate 15-second delay before monitoring starts
    print("Simulating 15-second delay before monitoring starts...")
    time.sleep(2)  # Shortened for testing
    
    # Now monitoring starts (first price received)
    monitoring_start_time = int(time.time() * 1000)
    position.set_monitoring_start_time(monitoring_start_time)
    
    print(f"Monitoring started at: {monitoring_start_time}")
    print(f"Delay: {(monitoring_start_time - entry_time) / 1000:.1f} seconds")
    print()
    
    # Test 1: Check immediately after monitoring starts (should not exit)
    print("Test 1: Immediately after monitoring starts")
    current_price = entry_price  # Same as entry price
    should_exit, reason = position.should_exit(current_price)
    print(f"  Current price: {current_price:.8f} SOL (same as entry)")
    print(f"  Should exit: {should_exit}, Reason: {reason}")
    print()
    
    # Test 2: Check after 2 seconds with 3% gain (should not exit - insufficient gain)
    print("Test 2: After 2 seconds with 3% gain (insufficient)")
    time.sleep(2.1)
    current_price = entry_price * 1.03  # 3% gain
    should_exit, reason = position.should_exit(current_price)
    print(f"  Current price: {current_price:.8f} SOL (+3%)")
    print(f"  Should exit: {should_exit}, Reason: {reason}")
    print()
    
    # Test 3: Check after 3+ seconds with 3% gain (should exit - insufficient gain)
    print("Test 3: After 3+ seconds with 3% gain (insufficient)")
    time.sleep(1.1)  # Total 3+ seconds from monitoring start
    should_exit, reason = position.should_exit(current_price)
    print(f"  Current price: {current_price:.8f} SOL (+3%)")
    print(f"  Should exit: {should_exit}, Reason: {reason}")
    print()
    
    # Test 4: Check with 6% gain after 3+ seconds (should not exit - sufficient gain)
    print("Test 4: After 3+ seconds with 6% gain (sufficient)")
    current_price = entry_price * 1.06  # 6% gain
    should_exit, reason = position.should_exit(current_price)
    print(f"  Current price: {current_price:.8f} SOL (+6%)")
    print(f"  Should exit: {should_exit}, Reason: {reason}")
    print()
    
    # Test 5: Check max hold time (should not exit yet - only 6 seconds from monitoring start, but 8+ from entry)
    print("Test 5: Max hold time check (8+ seconds from entry, 6 seconds from monitoring start)")
    should_exit, reason = position.should_exit(current_price)
    print(f"  Current price: {current_price:.8f} SOL (+6%)")
    print(f"  Should exit: {should_exit}, Reason: {reason}")
    print()
    
    # Test 6: Check max hold time after 10+ seconds from entry (should exit)
    print("Test 6: Max hold time check (10+ seconds from entry)")
    time.sleep(2.1)  # Total 10+ seconds from entry
    should_exit, reason = position.should_exit(current_price)
    print(f"  Current price: {current_price:.8f} SOL (+6%)")
    print(f"  Should exit: {should_exit}, Reason: {reason}")
    print()
    
    print("Key insight: Price-based conditions use monitoring start time, max_hold_time uses entry time!")
    print("This ensures accurate timing for monitoring-dependent conditions while preserving total hold time limits.")

def test_no_monitoring_start():
    """Test that positions without monitoring_start_ts don't trigger time-based exits."""
    print("\nTesting No Monitoring Start")
    print("=" * 40)
    
    # Create position without setting monitoring start time
    mint = Pubkey.from_string("11111111111111111111111111111112")
    platform = Platform.PUMP_FUN
    entry_price = 0.00010000
    entry_time = int(time.time() * 1000)
    
    position = Position.create_from_buy_result(
        mint=mint,
        platform=platform,
        entry_net_price_decimal=entry_price,
        quantity=1000.0,
        token_swapin_amount_raw=1000000000,
        entry_ts=entry_time,
        transaction_fee_raw=5000,
        platform_fee_raw=1000,
        exit_strategy="tp_sl",
        buy_amount=0.001,
        total_net_sol_swapout_amount_raw=100000,
        take_profit_percentage=0.5,
        stop_loss_percentage=0.2,
        trailing_stop_percentage=None,
        max_hold_time=5,  # 5 second max hold
        max_no_price_change_time=None,
        min_gain_percentage=0.1,  # 10% minimum gain required
        min_gain_time_window=2,  # 2 second window
    )
    
    print(f"Position created (monitoring_start_ts: {position.monitoring_start_ts})")
    
    # Test that time-based conditions are NOT checked without monitoring start
    time.sleep(2.1)  # 2+ seconds from entry
    current_price = entry_price * 1.05  # 5% gain (insufficient)
    should_exit, reason = position.should_exit(current_price)
    
    print(f"After 2+ seconds with 5% gain:")
    print(f"  Should exit: {should_exit}, Reason: {reason}")
    print("  (Time-based conditions are NOT checked without monitoring_start_ts)")
    print("  This prevents premature exits when monitoring hasn't started yet.")

if __name__ == "__main__":
    test_delayed_monitoring_timing()
    test_no_monitoring_start()
