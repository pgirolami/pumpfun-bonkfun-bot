#!/usr/bin/env python3
"""
Calculate volatility for specific price changes.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.volatility import VolatilityCalculator, calculate_take_profit_adjustment


def calculate_volatility_for_changes(entry_price: float, changes: list[float]) -> None:
    """Calculate volatility for a series of price changes.
    
    Args:
        entry_price: Starting price
        changes: List of percentage changes (e.g., [0.0139, 0.4727] for 1.39%, 47.27%)
    """
    print(f"Entry price: {entry_price:.8f} SOL")
    print(f"Price changes: {[f'{c*100:.2f}%' for c in changes]}")
    print()
    
    # Create volatility calculator with longer window
    calculator = VolatilityCalculator(window_seconds=5.0)
    
    # Simulate price changes over time
    base_time = 1000000000  # Some base timestamp
    
    print("Price progression:")
    print(f"  Time 0: {entry_price:.8f} SOL (entry)")
    
    # Add entry price to calculator first
    calculator.add_price(entry_price, base_time)
    
    for i, change in enumerate(changes):
        # Calculate new price
        new_price = entry_price * (1 + change)
        timestamp = base_time + (i + 1) * 1.0  # 1 second intervals to ensure time span > 0
        
        print(f"  Time {i+1}: {new_price:.8f} SOL (+{change*100:.2f}%)")

        # Add to volatility calculator
        calculator.add_price(new_price, timestamp)
        
        
        # Calculate volatility if we have enough data
        data_count = calculator.get_data_count()
        has_sufficient = calculator.has_sufficient_data(timestamp)
        print(f"    Data points: {data_count}, Sufficient: {has_sufficient}")
        
        if has_sufficient:
            volatility = calculator.calculate_volatility(timestamp)
            level = calculator.get_volatility_level(timestamp)
            
            if volatility is not None:
                print(f"    Volatility: {volatility:.4f} ({level})")
                
                # Show take profit adjustment
                original_tp = 1  # 40% take profit
                adjusted_tp = calculate_take_profit_adjustment(original_tp, level)
                reduction = ((original_tp - adjusted_tp) / original_tp) * 100
                
                print(f"    TP Adjustment: {original_tp*100:.1f}% -> {adjusted_tp*100:.1f}% (reduction: {reduction:.1f}%)")
            else:
                print(f"    Volatility: calculating... ({level})")
        else:
            print(f"    Volatility: insufficient data")
    
    # Add one more data point to ensure we have enough for calculation
    # Calculate final price by applying all changes
    final_price = entry_price
    for change in changes:
        final_price = final_price * (1 + change)
    final_price = final_price * 1.01  # 1% increase
    
    final_timestamp = base_time + len(changes) + 1
    calculator.add_price(final_price, final_timestamp)
    print(f"  Time {len(changes)+1}: {final_price:.8f} SOL (+1.00%)")
    
    # Calculate final volatility
    data_count = calculator.get_data_count()
    has_sufficient = calculator.has_sufficient_data(final_timestamp)
    print(f"    Data points: {data_count}, Sufficient: {has_sufficient}")
    
    if has_sufficient:
        volatility = calculator.calculate_volatility(final_timestamp)
        level = calculator.get_volatility_level(final_timestamp)
        
        if volatility is not None:
            print(f"    Volatility: {volatility:.4f} ({level})")
            
            # Show take profit adjustment
            original_tp = 0.4  # 40% take profit
            adjusted_tp = calculate_take_profit_adjustment(original_tp, level)
            reduction = ((original_tp - adjusted_tp) / original_tp) * 100
            
            print(f"    TP Adjustment: {original_tp*100:.1f}% -> {adjusted_tp*100:.1f}% (reduction: {reduction:.1f}%)")
        else:
            print(f"    Volatility: calculating... ({level})")
    else:
        print(f"    Volatility: insufficient data")
    
    print()
    
    # Final summary
    if calculator.has_sufficient_data(final_timestamp):
        final_volatility = calculator.calculate_volatility( final_timestamp)
        final_level = calculator.get_volatility_level( final_timestamp  )
        
        if final_volatility is not None:
            print(f"Final volatility: {final_volatility:.4f} ({final_level})")
            print(f"Interpretation: {final_volatility*100:.2f}% price change per second on average")
        else:
            print("Final volatility: Could not calculate")
    else:
        print("Final volatility: Insufficient data")


def main():
    """Calculate volatility for the specific example."""
    print("Volatility Calculation for Specific Price Changes")
    print("=" * 60)
    print()
    
    # Your specific example: 1.39% and 47.27% increases
    entry_price = 0.00003  # Example entry price in SOL
    changes = [0.0139, 0.4727, 0.7133, 0.7554, 0.7529, 0.8063]  # 1.39% and 47.27% increases
    
    calculate_volatility_for_changes(entry_price, changes)
    
    print("\n" + "=" * 60)
    print("Additional Analysis:")
    print()
    
    # Let's also show what this means in terms of the actual price movement
    price1 = entry_price * (1 + changes[0])
    price2 = price1 * (1 + changes[1])
    
    print(f"Price progression:")
    print(f"  Entry:  {entry_price:.8f} SOL")
    print(f"  After 1st change (+{changes[0]*100:.2f}%): {price1:.8f} SOL")
    print(f"  After 2nd change (+{changes[1]*100:.2f}%): {price2:.8f} SOL")
    print(f"  Total gain: {((price2 - entry_price) / entry_price * 100):.2f}%")
    
    # Calculate the standard deviation manually to verify
    returns = [changes[0], changes[1]]
    mean_return = sum(returns) / len(returns)
    variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
    std_dev = (variance ** 0.5)
    
    print(f"\nManual calculation:")
    print(f"  Returns: {[f'{r*100:.2f}%' for r in returns]}")
    print(f"  Mean return: {mean_return*100:.2f}%")
    print(f"  Standard deviation: {std_dev:.4f} ({std_dev*100:.2f}%)")
    print(f"  This represents the volatility per time period")


if __name__ == "__main__":
    main()
