#!/usr/bin/env python3
"""
Learning example: Listen to PumpPortal for new pump.fun tokens, then subscribe to trades.

This script:
1. Listens to PumpPortal for new tokens
2. Filters for pump.fun tokens only
3. Stops listening when it finds one
4. Subscribes to that token's bonding curve trades via PumpPortal
5. Displays all trade messages
"""

import sys
from pathlib import Path
import asyncio
import json
import websockets
from typing import Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from monitoring.universal_pumpportal_listener import UniversalPumpPortalListener
from interfaces.core import Platform, TokenInfo
from utils.logger import get_logger

logger = get_logger(__name__)


async def listen_for_pumpfun_tokens_and_subscribe_to_trades():
    """Listen for new pump.fun tokens and subscribe to trades using single WebSocket."""
    print("🎧 Starting PumpPortal listener for pump.fun tokens...")
    print("   (This will stop when the first pump.fun token is found)")
    print()
    
    pumpportal_url = "wss://pumpportal.fun/api/data"
    found_token: Optional[TokenInfo] = None
    
    try:
        async with websockets.connect(pumpportal_url) as websocket:
            print("✅ Connected to PumpPortal")
            
            # Subscribe to new tokens first
            subscribe_new_tokens = {
                "method": "subscribeNewToken",
                "params": []
            }
            
            print("📡 Subscribing to new token events...")
            await websocket.send(json.dumps(subscribe_new_tokens))
            print("✅ New token subscription sent!")
            print()
            
            # Listen for new tokens
            print("🎯 Listening for new pump.fun tokens...")
            while found_token is None:
                try:
                    message = await websocket.recv()
                    data = json.loads(message)
                    
                    # Check if this is a new token event
                    if "txType" in data and data["txType"] == "create":
                        
                        # Check if it's a pump.fun token
                        if data.get("pool") == "pump":
                            # Create TokenInfo object
                            from solders.pubkey import Pubkey
                            token_info = TokenInfo(
                                mint=Pubkey.from_string(data["mint"]),
                                platform=Platform.PUMP_FUN,
                                uri=data.get("uri", ""),
                                name=data.get("name", ""),
                                symbol=data.get("symbol", ""),
                                creator=Pubkey.from_string(data["traderPublicKey"]) if data.get("traderPublicKey") else None,
                                bonding_curve=None,  # Will be derived later
                                associated_bonding_curve=None,
                                creator_vault=None
                            )
                            
                            found_token = token_info
                            print(f"🎯 Found pump.fun token: {token_info.mint}")
                            print(f"   Platform: {token_info.platform.value}")
                            print(f"   Creator: {token_info.creator}")
                            print()
                            break
                    
                except json.JSONDecodeError as e:
                    print(f"❌ Failed to parse JSON: {e}")
                    continue
                except Exception as e:
                    print(f"❌ Error processing message: {e}")
                    continue
            
            if found_token:
                # Now subscribe to trades for this token
                await subscribe_to_trades_for_token(websocket, found_token)
        
    except KeyboardInterrupt:
        print("\n⏹️  Stopped by user")
        return None
    except Exception as e:
        print(f"❌ Listener error: {e}")
        return None
    
    return found_token


async def subscribe_to_trades_for_token(websocket, token_info: TokenInfo):
    """Subscribe to trades for a specific token using the existing WebSocket connection."""
    print(f"🔗 Subscribing to trades for token: {token_info.mint}")
    

    # Subscribe to trades using the same WebSocket connection
    subscribe_trades = {
        "method": "subscribeTokenTrade",
        "keys": [str(token_info.mint)]
    }
    
    print(f"📡 Sending trade subscription: {json.dumps(subscribe_trades, indent=2)}")
    await websocket.send(json.dumps(subscribe_trades))
    print("✅ Trade subscription sent!")
    print()
    
    # Listen for trade events
    print("🎯 Listening for trade events... (Press Ctrl+C to stop)")
    print("=" * 60)
    
    try:
        async for message in websocket:
            try:
                data = json.loads(message)
                await handle_trade_message(data)
            except json.JSONDecodeError as e:
                print(f"❌ Failed to parse JSON: {e}")
                print(f"   Raw message: {message[:200]}...")
            except Exception as e:
                print(f"❌ Error handling message: {e}")
                
    except websockets.exceptions.ConnectionClosed:
        print("🔌 Connection closed by server")
    except KeyboardInterrupt:
        print("\n⏹️  Stopped by user")


async def handle_trade_message(data: dict):
    """Handle incoming trade messages."""
    if "result" in data:
        # Subscription confirmation
        print(f"✅ Subscription confirmed: {data['result']}")
        return
    
    if "method" in data and "params" in data:
        # Trade event data
        print(f"📊 Trade Event:")
        print(f"   Method: {data.get('method', 'unknown')}")
        print(f"   Data: {json.dumps(data.get('params', {}), indent=4)}")
        print("-" * 40)
    else:
        # Other messages
        print(f"📨 Message: {json.dumps(data, indent=2)}")


async def get_bonding_curve_address(token_mint: str) -> Optional[str]:
    """Get the bonding curve address for a token mint.
    
    Args:
        token_mint: The token mint address
        
    Returns:
        The bonding curve address or None if not found
    """
    try:
        from platforms.pumpfun.address_provider import PumpFunAddressProvider
        from solders.pubkey import Pubkey
        
        address_provider = PumpFunAddressProvider()
        token_mint_pubkey = Pubkey.from_string(token_mint)
        bonding_curve = address_provider.derive_pool_address(token_mint_pubkey)
        
        print(f"🏦 Bonding curve address: {bonding_curve}")
        return bonding_curve
        
    except Exception as e:
        print(f"❌ Failed to get bonding curve address: {e}")
        return None


async def main():
    """Main function to orchestrate the listening and subscription."""
    print("🚀 PumpPortal Token Listener & Trade Subscriber")
    print("=" * 50)
    print("This example will:")
    print("1. Listen for new pump.fun tokens on PumpPortal")
    print("2. Stop when the first one is found")
    print("3. Subscribe to that token's bonding curve trades")
    print("4. Display all trade events")
    print("⚠️  Uses SINGLE WebSocket connection (PumpPortal requirement)")
    print()
    
    try:
        # Use single WebSocket connection for both listening and trading
        await listen_for_pumpfun_tokens_and_subscribe_to_trades()
        
    except KeyboardInterrupt:
        print("\n⏹️  Stopped by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("Starting PumpPortal token listener and trade subscriber...")
    print("Make sure you have a stable internet connection.")
    print("The script will wait for new pump.fun tokens to appear.")
    print()
    
    asyncio.run(main())
