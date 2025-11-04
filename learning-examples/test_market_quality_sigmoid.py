"""
Learning example to visualize sigmoid transformation for market quality scores.

This script tests the sigmoid function used in the avg_pnl algorithm
for normalized PnL values from -1.0 to 1.0.
"""

import math


def calculate_quality_score(avg_normalized_pnl: float) -> tuple[float, float]:
    """Calculate sigmoid value and quality score for a given normalized PnL.
    
    Args:
        avg_normalized_pnl: Average normalized PnL (PnL / amount_spent)
        
    Returns:
        Tuple of (sigmoid_value, quality_score)
    """
    # Linear function:
    # - normalized_pnl >= 0.01 → quality_score = 1.0
    # - normalized_pnl <= -0.01 → quality_score = 0.0
    # - -0.01 < normalized_pnl < 0.01 → linear interpolation
    
    if avg_normalized_pnl >= 0.01:
        quality_score = 1.0
        sigmoid_value = 1.0
    elif avg_normalized_pnl <= -0.01:
        quality_score = 0.0
        sigmoid_value = 0.0
    else:
        # Linear interpolation: y = (x + 0.01) / 0.02
        # At x = -0.01: y = 0.0
        # At x = 0.0: y = 0.5
        # At x = 0.01: y = 1.0
        quality_score = (avg_normalized_pnl + 0.01) / 0.02
        sigmoid_value = quality_score  # Not a sigmoid anymore, but keeping for compatibility
    
    return sigmoid_value, quality_score


def main() -> None:
    """Print sigmoid values and quality scores for normalized PnL from -1.0 to 1.0."""
    print("=" * 80)
    print("Market Quality Sigmoid Transformation Test")
    print("=" * 80)
    print()
    print(f"{'Normalized PnL':<18} {'Function Type':<18} {'Function Value':<18} {'Quality Score':<18}")
    print("-" * 80)
    
    # Test values from -1.0 to 1.0 with step 0.01
    for normalized_pnl in range(-100, 110, 1):
        normalized_pnl_float = normalized_pnl / 1000.0
        sigmoid_value, quality_score = calculate_quality_score(normalized_pnl_float)
        
        # Calculate linear coefficient for display (only meaningful for -0.01 < x < 0.01)
        if normalized_pnl_float >= 0.01:
            sigmoid_input_str = "     N/A"
        elif normalized_pnl_float <= -0.01:
            sigmoid_input_str = "     N/A"
        else:
            # Linear function: y = (x + 0.01) / 0.02
            sigmoid_input_str = "  linear"
        
        print(
            f"{normalized_pnl_float:>8.3f}         "
            f"{sigmoid_input_str}         "
            f"{sigmoid_value:>8.6f}         "
            f"{quality_score:>8.6f}"
        )
    
    print()
    print("=" * 80)
    print("Key Points:")
    print("  - normalized_pnl >= 0.01  → quality_score = 1.0 (full buy)")
    print("  - normalized_pnl = 0.0    → quality_score = 0.5 (half buy)")
    print("  - normalized_pnl <= -0.01 → quality_score = 0.0 (no buy)")
    print("  - Linear interpolation between -0.01 and 0.01")
    print("=" * 80)


if __name__ == "__main__":
    main()

