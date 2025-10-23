#!/usr/bin/env python3
"""
Test script: Subscribe to PumpPortal trades for a known token.

This script tests the trade subscription functionality using a known token.
"""

import sys
from pathlib import Path
import asyncio
import json
import websockets
from typing import Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from platforms.pumpfun.address_provider import PumpFunAddressProvider
from utils.logger import get_logger

logger = get_logger(__name__)


async def test_trade_subscription():
    """Test subscribing to trades for a known token."""
    print("🧪 Testing PumpPortal Trade Subscription")
    print("=" * 40)
    
    # Use a known token mint (you can replace this with any pump.fun token)
    test_token_mint_str = "11111111111111111111111111111112"  # System program as example
    
    try:
        # Get bonding curve address
        print(f"📋 Test token mint: {test_token_mint_str}")
        
        from solders.pubkey import Pubkey
        test_token_mint = Pubkey.from_string(test_token_mint_str)
        
        address_provider = PumpFunAddressProvider()
        bonding_curve = address_provider.derive_pool_address(test_token_mint)
        
        print(f"🏦 Bonding curve: {bonding_curve}")
        print()
        
        # Connect to PumpPortal
        pumpportal_url = "wss://pumpportal.fun/api/data"
        print(f"🔗 Connecting to: {pumpportal_url}")
        
        async with websockets.connect(pumpportal_url) as websocket:
            print("✅ Connected to PumpPortal")
            
            # Subscribe to token trades
            subscribe_message = {
                "method": "subscribeTokenTrade",
                "keys": [str(bonding_curve)]
            }
            
            print(f"📡 Sending subscription: {json.dumps(subscribe_message, indent=2)}")
            await websocket.send(json.dumps(subscribe_message))
            print("✅ Subscription sent!")
            print()
            
            # Listen for messages for 10 seconds
            print("🎯 Listening for messages (10 seconds)...")
            print("-" * 40)
            
            try:
                async for message in websocket:
                    try:
                        data = json.loads(message)
                        print(f"📨 Received: {json.dumps(data, indent=2)}")
                        print("-" * 40)
                    except json.JSONDecodeError as e:
                        print(f"❌ JSON decode error: {e}")
                        print(f"   Raw: {message[:100]}...")
                    except Exception as e:
                        print(f"❌ Error: {e}")
                        
            except asyncio.TimeoutError:
                print("⏰ Timeout reached")
                
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("Testing PumpPortal trade subscription...")
    print("This will connect to PumpPortal and subscribe to trades for a test token.")
    print()
    
    # Run for 10 seconds then stop
    try:
        asyncio.run(asyncio.wait_for(test_trade_subscription(), timeout=10))
    except asyncio.TimeoutError:
        print("⏰ Test completed (timeout)")
    except KeyboardInterrupt:
        print("\n⏹️  Test stopped by user")
