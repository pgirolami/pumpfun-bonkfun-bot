#!/usr/bin/env python3
"""
Simple volatility calculation for specific price changes.
"""

import math


def calculate_volatility_manual(returns):
    """Calculate volatility manually for a list of returns.
    
    Args:
        returns: List of percentage returns (e.g., [0.0139, 0.4727] for 1.39%, 47.27%)
        
    Returns:
        Tuple of (volatility, level)
    """
    if len(returns) < 2:
        return None, "insufficient_data"
    
    # Calculate mean return
    mean_return = sum(returns) / len(returns)
    
    # Calculate variance
    variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
    
    # Calculate standard deviation (volatility)
    std_dev = math.sqrt(variance)
    
    # Classify volatility level
    if std_dev < 0.05:  # < 5%
        level = "low"
    elif std_dev < 0.15:  # 5-15%
        level = "medium"
    else:  # > 15%
        level = "high"
    
    return std_dev, level


def calculate_take_profit_adjustment(original_tp_percentage, volatility_level):
    """Calculate adjusted take profit percentage based on volatility level."""
    adjustments = {
        'low': 0.0,      # No adjustment
        'medium': 0.25,  # Reduce by 25%
        'high': 0.45,    # Reduce by 45%
    }
    
    adjustment_factor = adjustments.get(volatility_level, 0.0)
    adjusted_percentage = original_tp_percentage * (1 - adjustment_factor)
    
    # Ensure we don't go below 5% take profit
    return max(adjusted_percentage, 0.05)


def main():
    """Calculate volatility for the specific example."""
    print("Volatility Calculation for Price Changes: 1.39% and 47.27%")
    print("=" * 60)
    print()
    
    # Your specific example
    returns = [0.0139, 0.4727, 0.7133, 0.7554, 0.7529, 0.8063]  # 1.39% and 47.27% increases
    
    print(f"Price changes: {[f'{r*100:.2f}%' for r in returns]}")
    print()
    
    # Calculate volatility
    volatility, level = calculate_volatility_manual(returns)
    
    if volatility is not None:
        print(f"Volatility: {volatility:.4f} ({volatility*100:.2f}%)")
        print(f"Level: {level}")
        print()
        
        # Show take profit adjustment
        original_tp = 0.4  # 40% take profit
        adjusted_tp = calculate_take_profit_adjustment(original_tp, level)
        reduction = ((original_tp - adjusted_tp) / original_tp) * 100
        
        print(f"Take Profit Adjustment:")
        print(f"  Original TP: {original_tp*100:.1f}%")
        print(f"  Adjusted TP: {adjusted_tp*100:.1f}%")
        print(f"  Reduction: {reduction:.1f}%")
        print()
        
        # Show what this means in practice
        entry_price = 0.0001  # Example entry price
        original_tp_price = entry_price * (1 + original_tp)
        adjusted_tp_price = entry_price * (1 + adjusted_tp)
        
        print(f"Practical Example (Entry: {entry_price:.8f} SOL):")
        print(f"  Original TP price: {original_tp_price:.8f} SOL")
        print(f"  Adjusted TP price: {adjusted_tp_price:.8f} SOL")
        print(f"  Price difference: {((original_tp_price - adjusted_tp_price) / original_tp_price * 100):.1f}% lower")
        
    else:
        print("Insufficient data for volatility calculation")
    
    print()
    print("=" * 60)
    print("Analysis:")
    print(f"  Mean return: {sum(returns)/len(returns)*100:.2f}%")
    print(f"  Standard deviation: {volatility*100:.2f}%")
    print(f"  This represents the average deviation from the mean return")
    print(f"  Higher volatility means more unpredictable price movements")


if __name__ == "__main__":
    main()
