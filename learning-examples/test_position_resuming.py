#!/usr/bin/env python3
"""
Test script for position resuming with insufficient gain logic.

This script tests that when positions are loaded from the database,
they correctly use the current bot configuration for min_gain_percentage.
"""

import sys
from pathlib import Path
import time
import asyncio

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from database.models import PositionConverter
from trading.position import Position, ExitReason
from solders.pubkey import Pubkey
from interfaces.core import Platform

def test_position_resuming_with_config():
    """Test that position resuming uses current configuration values."""
    print("Testing Position Resuming with Configuration")
    print("=" * 50)
    
    # Simulate a database row (without min_gain_percentage)
    # This represents an old position that was created before the insufficient gain feature
    db_row = (
        "test-position-1",  # id
        "11111111111111111111111111111112",  # mint
        "pump_fun",  # platform
        0.00010000,  # entry_net_price_decimal
        0.0,  # take_profit_price (not used in from_row)
        1000000000,  # total_token_swapin_amount_raw
        0,  # total_token_swapout_amount_raw
        int(time.time() * 1000) - 5000,  # entry_ts (5 seconds ago)
        "tp_sl",  # exit_strategy
        0.00010000,  # highest_price
        None,  # max_no_price_change_time
        time.time(),  # last_price_change_ts
        1,  # is_active
        None,  # exit_reason
        None,  # exit_net_price_decimal
        None,  # exit_ts
        5000,  # transaction_fee_raw
        1000,  # platform_fee_raw
        0.0,  # realized_pnl_sol_decimal
        0.0,  # realized_net_pnl_sol_decimal
        0.001,  # buy_amount
        100000,  # total_net_sol_swapout_amount_raw
        0,  # total_net_sol_swapin_amount_raw
    )
    
    print("Database row created (simulating old position without min_gain_percentage)")
    print(f"Entry time: {db_row[7]} (5 seconds ago)")
    print()
    
    # Test 1: Resume with min_gain_percentage = 0.1 (10%)
    print("Test 1: Resume with min_gain_percentage = 0.1 (10%)")
    position_with_gain = PositionConverter.from_row(db_row, min_gain_percentage=0.1)
    
    print(f"  Position min_gain_percentage: {position_with_gain.min_gain_percentage}")
    print(f"  Position min_gain_time_window: {position_with_gain.min_gain_time_window}")
    print(f"  Position is_active: {position_with_gain.is_active}")
    
    # Test insufficient gain logic
    current_price = 0.00010500  # 5% gain
    should_exit, reason = position_with_gain.should_exit(current_price)
    print(f"  Current price: {current_price:.8f} SOL (+5%)")
    print(f"  Should exit: {should_exit}, Reason: {reason}")
    print()
    
    # Test 2: Resume with min_gain_percentage = None (disabled)
    print("Test 2: Resume with min_gain_percentage = None (disabled)")
    position_disabled = PositionConverter.from_row(db_row, min_gain_percentage=None)
    
    print(f"  Position min_gain_percentage: {position_disabled.min_gain_percentage}")
    print(f"  Position min_gain_time_window: {position_disabled.min_gain_time_window}")
    
    # Test insufficient gain logic (should not exit)
    should_exit, reason = position_disabled.should_exit(current_price)
    print(f"  Current price: {current_price:.8f} SOL (+5%)")
    print(f"  Should exit: {should_exit}, Reason: {reason}")
    print()
    
    # Test 3: Resume with different min_gain_percentage = 0.05 (5%)
    print("Test 3: Resume with min_gain_percentage = 0.05 (5%)")
    position_low_gain = PositionConverter.from_row(db_row, min_gain_percentage=0.05)
    
    print(f"  Position min_gain_percentage: {position_low_gain.min_gain_percentage}")
    
    # Test insufficient gain logic (should not exit since 5% gain >= 5% required)
    should_exit, reason = position_low_gain.should_exit(current_price)
    print(f"  Current price: {current_price:.8f} SOL (+5%)")
    print(f"  Should exit: {should_exit}, Reason: {reason}")
    
    # Test with 3% gain (should exit)
    current_price_low = 0.00010300  # 3% gain
    should_exit, reason = position_low_gain.should_exit(current_price_low)
    print(f"  Current price: {current_price_low:.8f} SOL (+3%)")
    print(f"  Should exit: {should_exit}, Reason: {reason}")
    print()
    
    print("All position resuming tests completed!")

def test_backward_compatibility():
    """Test that old positions work correctly with new configuration."""
    print("\nTesting Backward Compatibility")
    print("=" * 40)
    
    # Simulate a very old database row (minimal fields)
    old_db_row = (
        "old-position-1",  # id
        "11111111111111111111111111111112",  # mint
        "pump_fun",  # platform
        0.00010000,  # entry_net_price_decimal
        0.0,  # take_profit_price
        1000000000,  # total_token_swapin_amount_raw
        0,  # total_token_swapout_amount_raw
        int(time.time() * 1000) - 10000,  # entry_ts (10 seconds ago)
        "time_based",  # exit_strategy
        0.00010000,  # highest_price
        None,  # max_no_price_change_time
        time.time(),  # last_price_change_ts
        1,  # is_active
        None,  # exit_reason
        None,  # exit_net_price_decimal
        None,  # exit_ts
        5000,  # transaction_fee_raw
        1000,  # platform_fee_raw
        0.0,  # realized_pnl_sol_decimal
        0.0,  # realized_net_pnl_sol_decimal
        0.001,  # buy_amount
        100000,  # total_net_sol_swapout_amount_raw
        0,  # total_net_sol_swapin_amount_raw
    )
    
    print("Old database row created (10 seconds ago)")
    print()
    
    # Resume with current configuration
    position = PositionConverter.from_row(old_db_row, min_gain_percentage=0.15)
    
    print(f"Resumed position with min_gain_percentage: {position.min_gain_percentage}")
    print(f"Entry time: {position.entry_ts} (10 seconds ago)")
    
    # Test with 10% gain (should exit since 10% < 15% required)
    current_price = 0.00011000  # 10% gain
    should_exit, reason = position.should_exit(current_price)
    print(f"Current price: {current_price:.8f} SOL (+10%)")
    print(f"Should exit: {should_exit}, Reason: {reason}")
    
    print("Backward compatibility test completed!")

if __name__ == "__main__":
    test_position_resuming_with_config()
    test_backward_compatibility()
