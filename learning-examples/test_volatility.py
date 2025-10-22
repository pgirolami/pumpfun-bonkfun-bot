#!/usr/bin/env python3
"""
Test volatility calculation and take profit adjustment logic.

This script demonstrates how the volatility calculator works with different
price scenarios and validates the take profit adjustment logic.
"""

import asyncio
import time
from typing import List

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.volatility import VolatilityCalculator, calculate_take_profit_adjustment


def simulate_price_data(scenario: str) -> List[tuple[float, float]]:
    """Generate simulated price data for different volatility scenarios.
    
    Args:
        scenario: 'stable', 'moderate', or 'extreme'
        
    Returns:
        List of (timestamp, price) tuples
    """
    base_price = 0.0001  # Starting price in SOL
    current_time = time.time()
    prices = []
    
    if scenario == "stable":
        # Low volatility: small, gradual changes
        for i in range(20):
            timestamp = current_time + i * 0.5  # 0.5 second intervals
            # Small random walk with 1-2% changes
            change = 0.01 + (i % 3) * 0.005  # 1-2.5% changes
            if i % 2 == 0:
                base_price *= (1 + change)
            else:
                base_price *= (1 - change * 0.5)
            prices.append((timestamp, base_price))
            
    elif scenario == "moderate":
        # Medium volatility: moderate swings
        for i in range(20):
            timestamp = current_time + i * 0.5
            # Moderate swings with 3-8% changes
            change = 0.03 + (i % 4) * 0.01  # 3-6% changes
            if i % 2 == 0:
                base_price *= (1 + change)
            else:
                base_price *= (1 - change * 0.7)
            prices.append((timestamp, base_price))
            
    elif scenario == "extreme":
        # High volatility: large swings
        for i in range(20):
            timestamp = current_time + i * 0.5
            # Large swings with 10-25% changes
            change = 0.10 + (i % 3) * 0.05  # 10-20% changes
            if i % 2 == 0:
                base_price *= (1 + change)
            else:
                base_price *= (1 - change * 0.8)
            prices.append((timestamp, base_price))
    
    return prices


def test_volatility_calculation():
    """Test volatility calculation with different scenarios."""
    print("=== Testing Volatility Calculation ===\n")
    
    scenarios = ["stable", "moderate", "extreme"]
    
    for scenario in scenarios:
        print(f"--- {scenario.upper()} Volatility Scenario ---")
        
        # Create volatility calculator
        calculator = VolatilityCalculator(window_seconds=5.0)
        
        # Generate price data
        price_data = simulate_price_data(scenario)
        
        print(f"Price data points: {len(price_data)}")
        print(f"Price range: {min(p[1] for p in price_data):.8f} - {max(p[1] for p in price_data):.8f} SOL")
        
        # Feed prices to calculator
        for timestamp, price in price_data:
            calculator.add_price(price, timestamp)
            
            # Calculate volatility when we have enough data
            if calculator.has_sufficient_data():
                volatility = calculator.calculate_volatility()
                level = calculator.get_volatility_level()
                if volatility is not None:
                    print(f"  Time {timestamp:.1f}: Price {price:.8f}, Volatility: {volatility:.4f} ({level})")
                else:
                    print(f"  Time {timestamp:.1f}: Price {price:.8f}, Volatility: calculating... ({level})")
        
        print()


def test_take_profit_adjustment():
    """Test take profit adjustment logic."""
    print("=== Testing Take Profit Adjustment ===\n")
    
    original_tp_percentage = 0.4  # 40% take profit
    
    volatility_levels = ["low", "medium", "high"]
    
    for level in volatility_levels:
        adjusted_tp = calculate_take_profit_adjustment(
            original_tp_percentage, 
            level
        )
        
        reduction_pct = ((original_tp_percentage - adjusted_tp) / original_tp_percentage) * 100
        
        print(f"Volatility Level: {level}")
        print(f"  Original TP: {original_tp_percentage * 100:.1f}%")
        print(f"  Adjusted TP:  {adjusted_tp * 100:.1f}%")
        print(f"  Reduction:    {reduction_pct:.1f}%")
        print()


def test_sliding_window():
    """Test sliding window behavior."""
    print("=== Testing Sliding Window Behavior ===\n")
    
    calculator = VolatilityCalculator(window_seconds=3.0)
    base_time = time.time()
    
    # Add prices over 10 seconds
    for i in range(20):
        timestamp = base_time + i * 0.5
        price = 0.0001 * (1 + i * 0.01)  # Gradually increasing price
        calculator.add_price(price, timestamp)
        
        data_count = calculator.get_data_count()
        has_sufficient = calculator.has_sufficient_data()
        
        print(f"  Time {timestamp:.1f}: Price {price:.8f}, Data points: {data_count}, Sufficient: {has_sufficient}")
        
        if has_sufficient:
            volatility = calculator.calculate_volatility()
            level = calculator.get_volatility_level()
            if volatility is not None:
                print(f"    Volatility: {volatility:.4f} ({level})")
            else:
                print(f"    Volatility: calculating... ({level})")


def test_real_world_scenario():
    """Test with a realistic trading scenario."""
    print("=== Real-World Trading Scenario ===\n")
    
    # Simulate a token that starts stable, then becomes volatile
    calculator = VolatilityCalculator(window_seconds=5.0)
    base_time = time.time()
    base_price = 0.0001
    
    # Phase 1: Stable price for 3 seconds
    print("Phase 1: Stable price (3 seconds)")
    for i in range(6):
        timestamp = base_time + i * 0.5
        price = base_price * (1 + (i % 2) * 0.001)  # Very small changes
        calculator.add_price(price, timestamp)
        
        if calculator.has_sufficient_data():
            volatility = calculator.calculate_volatility()
            level = calculator.get_volatility_level()
            if volatility is not None:
                print(f"  Time {timestamp:.1f}: Price {price:.8f}, Volatility: {volatility:.4f} ({level})")
            else:
                print(f"  Time {timestamp:.1f}: Price {price:.8f}, Volatility: calculating... ({level})")
    
    # Phase 2: Volatile price for 4 seconds
    print("\nPhase 2: Volatile price (4 seconds)")
    for i in range(6, 14):
        timestamp = base_time + i * 0.5
        # Large swings
        change = 0.05 + (i % 3) * 0.02  # 5-9% changes
        if i % 2 == 0:
            price = base_price * (1 + change)
        else:
            price = base_price * (1 - change * 0.8)
        base_price = price  # Update base for next iteration
        
        calculator.add_price(price, timestamp)
        
        if calculator.has_sufficient_data():
            volatility = calculator.calculate_volatility()
            level = calculator.get_volatility_level()
            if volatility is not None:
                print(f"  Time {timestamp:.1f}: Price {price:.8f}, Volatility: {volatility:.4f} ({level})")
                
                # Show take profit adjustment
                if level != "unknown":
                    original_tp = 0.4  # 40%
                    adjusted_tp = calculate_take_profit_adjustment(original_tp, level)
                    reduction = ((original_tp - adjusted_tp) / original_tp) * 100
                    print(f"    TP Adjustment: {original_tp*100:.1f}% -> {adjusted_tp*100:.1f}% (reduction: {reduction:.1f}%)")
            else:
                print(f"  Time {timestamp:.1f}: Price {price:.8f}, Volatility: calculating... ({level})")


async def main():
    """Run all volatility tests."""
    print("Volatility Calculator Test Suite")
    print("=" * 50)
    print()
    
    test_volatility_calculation()
    test_take_profit_adjustment()
    test_sliding_window()
    test_real_world_scenario()
    
    print("All tests completed!")


if __name__ == "__main__":
    asyncio.run(main())
