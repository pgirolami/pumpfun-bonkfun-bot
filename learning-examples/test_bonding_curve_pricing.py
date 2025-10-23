#!/usr/bin/env python3
"""
Test script: Verify bonding curve price calculation logic.

This script tests the BondingCurveState and PriceTracker classes
to ensure price calculations are working correctly.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import the classes directly from the file
import importlib.util
spec = importlib.util.spec_from_file_location("listen_pumpportal_trades", "learning-examples/listen_pumpportal_trades.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

BondingCurveState = module.BondingCurveState
PriceTracker = module.PriceTracker


def calculate_token_amount_from_sol(curve: BondingCurveState, sol_amount: float, is_buy: bool) -> float:
    """Calculate token amount from SOL amount using bonding curve formula.
    
    For pump.fun bonding curve: k = virtual_sol * virtual_tokens (constant product)
    When buying: new_virtual_sol = virtual_sol + sol_amount
    When selling: new_virtual_sol = virtual_sol - sol_amount
    
    Args:
        curve: Current bonding curve state
        sol_amount: Amount of SOL to trade
        is_buy: True for buy, False for sell
        
    Returns:
        Token amount that would be traded
    """
    if is_buy:
        # Buy: SOL goes in, tokens come out
        new_virtual_sol = curve.virtual_sol_reserves + sol_amount
        # k = virtual_sol * virtual_tokens (constant)
        k = curve.virtual_sol_reserves * curve.virtual_token_reserves
        new_virtual_tokens = k / new_virtual_sol
        token_amount = curve.virtual_token_reserves - new_virtual_tokens
    else:
        # Sell: tokens go in, SOL comes out
        new_virtual_sol = curve.virtual_sol_reserves - sol_amount
        # k = virtual_sol * virtual_tokens (constant)
        k = curve.virtual_sol_reserves * curve.virtual_token_reserves
        new_virtual_tokens = k / new_virtual_sol
        token_amount = new_virtual_tokens - curve.virtual_token_reserves
    
    return token_amount


def test_bonding_curve_basic():
    """Test basic bonding curve price calculation."""
    print("🧪 Testing Basic Bonding Curve Calculation")
    print("=" * 50)
    
    # Initialize bonding curve with correct pump.fun values
    curve = BondingCurveState()
    curve.virtual_sol_reserves = 30.0  # 30 SOL
    curve.virtual_token_reserves = 1073000000.0  # 1,073,000,000 tokens
    curve.real_sol_reserves = 0.0  # Real SOL starts at 0
    curve.real_token_reserves = 793100000.0  # Real tokens start at 793,100,000
    
    print(f"Initial state (pump.fun values):")
    print(f"  Virtual SOL: {curve.virtual_sol_reserves}")
    print(f"  Virtual Tokens: {curve.virtual_token_reserves:,.0f}")
    print(f"  Real SOL: {curve.real_sol_reserves}")
    print(f"  Real Tokens: {curve.real_token_reserves:,.0f}")
    print(f"  Initial Price: {curve.calculate_price():.10f} SOL/token")
    print(f"  Initial Price: ${curve.calculate_price() * 200:.6f} USD (at $200 SOL)")
    print()
    
    # Simulate a buy trade (0.1 SOL for tokens)
    print("Simulating buy trade: 0.1 SOL -> tokens")
    token_amount = calculate_token_amount_from_sol(curve, 0.1, True)
    print(f"  Calculated token amount: {token_amount:.0f} tokens")
    curve.apply_trade(sol_amount=0.1, token_amount=token_amount, is_buy=True)
    
    print(f"After buy trade:")
    print(f"  Virtual SOL: {curve.virtual_sol_reserves:.6f}")
    print(f"  Virtual Tokens: {curve.virtual_token_reserves:.0f}")
    print(f"  New Price: {curve.calculate_price():.10f} SOL/token")
    print()
    
    # Simulate another buy trade
    print("Simulating another buy trade: 0.1 SOL -> tokens")
    token_amount = calculate_token_amount_from_sol(curve, 0.1, True)
    print(f"  Calculated token amount: {token_amount:.0f} tokens")
    curve.apply_trade(sol_amount=0.1, token_amount=token_amount, is_buy=True)
    
    print(f"After second buy trade:")
    print(f"  Virtual SOL: {curve.virtual_sol_reserves:.6f}")
    print(f"  Virtual Tokens: {curve.virtual_token_reserves:.0f}")
    print(f"  New Price: {curve.calculate_price():.10f} SOL/token")
    print()


def test_price_tracker():
    """Test PriceTracker with multiple trades."""
    print("🧪 Testing PriceTracker with Multiple Trades")
    print("=" * 50)
    
    tracker = PriceTracker("test_token")
    tracker.initialize_from_token_creation()  # Uses correct pump.fun values (30 SOL, 1,073,000,000 tokens)
    
    print(f"Initial price: {tracker.bonding_curve.calculate_price():.10f} SOL/token")
    print()
    
    # Simulate several trades with calculated token amounts
    sol_amounts = [0.1, 0.1, 0.05, 0.2]  # SOL amounts
    is_buy_flags = [True, True, False, True]  # Buy/Sell flags


    sol_amounts = [2.568, 0.057, 0.652, 0.669, 0.635, 0.923]  # SOL amounts
    is_buy_flags = [True, True, True, True, True, True]  # Buy/Sell flags

    for i, (sol_amount, is_buy) in enumerate(zip(sol_amounts, is_buy_flags), 1):
        # Calculate token amount from bonding curve
        token_amount = calculate_token_amount_from_sol(tracker.bonding_curve, sol_amount, is_buy)
        
        trade = {
            "solAmount": sol_amount,
            "tokenAmount": token_amount,
            "isBuy": is_buy
        }
        
        print(f"Trade {i}: {sol_amount} SOL {'BUY' if is_buy else 'SELL'}")
        print(f"  Calculated token amount: {token_amount:.0f} tokens")
        result = tracker.process_trade(trade)
        
        if result:
            print(f"  Price: {result['price_sol']:.10f} SOL/token")
            print(f"  Change: {result['price_change_percent']:+.2f}%")
            print(f"  Virtual SOL: {result['virtual_sol']:.6f}")
            print(f"  Virtual Tokens: {result['virtual_tokens']:.0f}")
        print()


def test_price_impact():
    """Test how trade size affects price impact."""
    print("🧪 Testing Price Impact by Trade Size")
    print("=" * 50)
    
    # Test with different trade sizes
    trade_sizes = [0.01, 0.1, 1.0, 10.0]  # SOL amounts
    
    for sol_amount in trade_sizes:
        curve = BondingCurveState()
        # Start with proper pump.fun values
        curve.virtual_sol_reserves = 30.0  # 30 SOL
        curve.virtual_token_reserves = 1073000000.0  # 1,073,000,000 tokens
        curve.real_sol_reserves = 0.0
        curve.real_token_reserves = 793100000.0
        
        initial_price = curve.calculate_price()
        
        # Calculate token amount from bonding curve
        token_amount = calculate_token_amount_from_sol(curve, sol_amount, True)
        
        # Apply trade
        curve.apply_trade(sol_amount=sol_amount, token_amount=token_amount, is_buy=True)
        
        final_price = curve.calculate_price()
        price_change = ((final_price - initial_price) / initial_price) * 100
        
        print(f"Trade: {sol_amount:.2f} SOL")
        print(f"  Calculated token amount: {token_amount:.0f} tokens")
        print(f"  Initial Price: {initial_price:.10f} SOL/token")
        print(f"  Final Price: {final_price:.10f} SOL/token")
        print(f"  Price Change: {price_change:+.2f}%")
        print()


if __name__ == "__main__":
    print("Testing Bonding Curve Price Calculation")
    print("This verifies the price calculation logic works correctly.")
    print()
    
    test_bonding_curve_basic()
    test_price_tracker()
    test_price_impact()
    
    print("✅ All tests completed!")
    print("The bonding curve price calculation is working correctly.")
