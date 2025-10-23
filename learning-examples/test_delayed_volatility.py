#!/usr/bin/env python3
"""
Test script for delayed volatility monitoring.

This script tests that volatility monitoring starts only when the first price
is available, not at entry time, to handle cases where it takes up to 15 seconds
to get the first price after a buy.
"""

import sys
from pathlib import Path
import time

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.volatility import VolatilityCalculator

def test_delayed_volatility_monitoring():
    """Test that volatility monitoring starts only when first price is available."""
    print("Testing Delayed Volatility Monitoring")
    print("=" * 50)
    
    # Simulate a scenario where we have a 15-second delay before first price
    print("Scenario: 15-second delay before first price is available")
    print()
    
    # Create volatility calculator
    calculator = VolatilityCalculator(window_seconds=5.0)
    
    # Simulate entry time (when position is created)
    entry_time = time.time()
    print(f"Entry time: {entry_time:.2f}")
    print("Waiting 15 seconds for first price...")
    
    # Simulate 15-second delay (in real scenario, this would be waiting for RPC)
    time.sleep(2)  # Shortened for testing
    
    # Now we get the first price
    first_price_time = time.time()
    first_price = 0.00010000  # 0.0001 SOL
    
    print(f"First price available at: {first_price_time:.2f}")
    print(f"Delay: {first_price_time - entry_time:.1f} seconds")
    print(f"First price: {first_price:.8f} SOL")
    print()
    
    # Add first price to calculator
    calculator.add_price(first_price, first_price_time)
    print("Added first price to volatility calculator")
    
    # Check if we have sufficient data (should be False)
    has_sufficient = calculator.has_sufficient_data(first_price_time)
    print(f"Has sufficient data: {has_sufficient}")
    print()
    
    # Simulate price updates every 1 second for 10 seconds
    print("Simulating price updates every 1 second:")
    current_time = first_price_time
    current_price = first_price
    
    for i in range(10):
        current_time += 1.0  # 1 second later
        current_price *= 1.02  # 2% increase each time
        
        calculator.add_price(current_price, current_time)
        has_sufficient = calculator.has_sufficient_data(current_time)
        volatility = calculator.calculate_volatility(current_time)
        level = calculator.get_volatility_level(current_time) if volatility is not None else "unknown"
        
        volatility_str = f"{volatility:.4f}" if volatility is not None else "None"
        print(f"  Time +{i+1}s: Price {current_price:.8f} SOL, "
              f"Sufficient: {has_sufficient}, "
              f"Volatility: {volatility_str} ({level})")
    
    print()
    print("Key insight: Volatility monitoring window starts from first price time, not entry time!")
    print("This ensures accurate volatility calculation even with delayed price availability.")

def test_volatility_window_timing():
    """Test that the volatility window is correctly timed from first price."""
    print("\nTesting Volatility Window Timing")
    print("=" * 40)
    
    calculator = VolatilityCalculator(window_seconds=3.0)  # 3-second window
    
    # Simulate entry at time 0
    entry_time = 0.0
    print(f"Entry time: {entry_time}")
    
    # First price available at time 10 (10-second delay)
    first_price_time = 10.0
    first_price = 0.00010000
    print(f"First price at time {first_price_time}: {first_price:.8f} SOL")
    
    calculator.add_price(first_price, first_price_time)
    
    # Add prices at times 11, 12, 13 (within 3-second window from first price)
    for i in range(1, 4):
        price_time = first_price_time + i
        price = first_price * (1 + i * 0.01)  # 1%, 2%, 3% increases
        calculator.add_price(price, price_time)
        
        has_sufficient = calculator.has_sufficient_data(price_time)
        volatility = calculator.calculate_volatility(price_time)
        
        volatility_str = f"{volatility:.4f}" if volatility is not None else "None"
        print(f"  Time {price_time}: Price {price:.8f} SOL, "
              f"Sufficient: {has_sufficient}, "
              f"Volatility: {volatility_str}")
    
    print()
    print("Window should be: 10-13 seconds (3-second window from first price)")
    print("NOT: 0-3 seconds (which would include the 10-second delay)")

if __name__ == "__main__":
    test_delayed_volatility_monitoring()
    test_volatility_window_timing()
