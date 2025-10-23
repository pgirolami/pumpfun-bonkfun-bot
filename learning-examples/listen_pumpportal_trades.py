#!/usr/bin/env python3
"""
Learning example: Listen to PumpPortal for new pump.fun tokens, then subscribe to trades with price tracking.

This script:
1. Listens to PumpPortal for new tokens
2. Filters for pump.fun tokens only
3. Stops listening when it finds one
4. Subscribes to that token's bonding curve trades via PumpPortal
5. Calculates and displays price changes after each trade using bonding curve formula
6. Tracks virtual and real SOL/token reserves
"""

import sys
from pathlib import Path
import asyncio
import json
import websockets
from typing import Optional
from dataclasses import dataclass
import math

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from monitoring.universal_pumpportal_listener import UniversalPumpPortalListener
from interfaces.core import Platform, TokenInfo
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class BondingCurveState:
    """Tracks the current state of a bonding curve for price calculation."""
    virtual_sol_reserves: float = 0.0
    virtual_token_reserves: float = 0.0
    real_sol_reserves: float = 0.0
    real_token_reserves: float = 0.0
    
    def calculate_price(self) -> float:
        """Calculate current price using bonding curve formula.
        
        Returns:
            Price in SOL per token
        """
        if self.virtual_token_reserves == 0:
            return 0.0
        
        # Price = virtual_sol_reserves / virtual_token_reserves
        return self.virtual_sol_reserves / self.virtual_token_reserves
    
    def apply_trade(self, sol_amount: float, token_amount: float, is_buy: bool):
        """Apply a trade to the bonding curve state.
        
        Args:
            sol_amount: Amount of SOL in the trade
            token_amount: Amount of tokens in the trade
            is_buy: True if this is a buy (token -> SOL), False if sell (SOL -> token)
        """
        if is_buy:
            # Buy: tokens go out, SOL comes in
            self.virtual_sol_reserves += sol_amount
            self.virtual_token_reserves -= token_amount
            self.real_sol_reserves += sol_amount
            self.real_token_reserves -= token_amount
        else:
            # Sell: SOL goes out, tokens come in
            self.virtual_sol_reserves -= sol_amount
            self.virtual_token_reserves += token_amount
            self.real_sol_reserves -= sol_amount
            self.real_token_reserves += token_amount


class PriceTracker:
    """Tracks price changes for a specific token."""
    
    def __init__(self, token_mint: str):
        self.token_mint = token_mint
        self.bonding_curve = BondingCurveState()
        self.initialized = False
        
    def initialize_from_token_creation(self, initial_sol: float = 30.0, initial_tokens: float = 1073000000.0):
        """Initialize bonding curve state from token creation.
        
        Args:
            initial_sol: Initial SOL in the curve (30 SOL for pump.fun)
            initial_tokens: Initial virtual token supply (1,073,000,000 for pump.fun)
        """
        # Pump.fun bonding curve starts with virtual reserves
        # Virtual SOL reserves start at 30 SOL, virtual token reserves at 1,073,000,000
        self.bonding_curve.virtual_sol_reserves = initial_sol
        self.bonding_curve.virtual_token_reserves = initial_tokens
        self.bonding_curve.real_sol_reserves = 0.0  # Real reserves start at 0
        self.bonding_curve.real_token_reserves = 793100000.0  # Real token reserves start at 793,100,000
        self.initialized = True
        
        logger.info(f"Initialized bonding curve for {self.token_mint}")
        logger.info(f"  Virtual SOL: {self.bonding_curve.virtual_sol_reserves}")
        logger.info(f"  Virtual Tokens: {self.bonding_curve.virtual_token_reserves}")
        logger.info(f"  Real SOL: {self.bonding_curve.real_sol_reserves}")
        logger.info(f"  Real Tokens: {self.bonding_curve.real_token_reserves}")
        logger.info(f"  Starting Price: {self.bonding_curve.calculate_price():.10f} SOL/token")
    
    def process_trade(self, trade_data: dict) -> dict:
        """Process a trade and update price.
        
        Args:
            trade_data: Trade data from PumpPortal
            
        Returns:
            Dictionary with price information
        """
        if not self.initialized:
            logger.warning("Bonding curve not initialized, skipping trade")
            return {}
        
        # Extract trade information
        sol_amount = float(trade_data.get("solAmount", 0))
        token_amount = float(trade_data.get("tokenAmount", 0))
        is_buy = trade_data.get("isBuy", True)  # Default to buy if not specified

        old_price = self.bonding_curve.virtual_sol_reserves / max(self.bonding_curve.virtual_token_reserves, 1)

        # Apply trade to bonding curve
        self.bonding_curve.apply_trade(sol_amount, token_amount, is_buy)
        
        # Calculate new price
        new_price = self.bonding_curve.calculate_price()
        
        # Calculate price change
        price_change = ((new_price - old_price) / max(old_price, 1e-10)) * 100 if old_price > 0 else 0
        
        result = {
            "price_sol": new_price,
            "price_change_percent": price_change,
            "virtual_sol": self.bonding_curve.virtual_sol_reserves,
            "virtual_tokens": self.bonding_curve.virtual_token_reserves,
            "real_sol": self.bonding_curve.real_sol_reserves,
            "real_tokens": self.bonding_curve.real_token_reserves,
            "trade_type": "BUY" if is_buy else "SELL",
            "sol_amount": sol_amount,
            "token_amount": token_amount
        }
        
        logger.info(f"Trade processed for {self.token_mint}:")
        logger.info(f"  Type: {'BUY' if is_buy else 'SELL'}")
        logger.info(f"  SOL: {sol_amount:.6f}, Tokens: {token_amount:.0f}")
        logger.info(f"  New Price: {new_price:.10f} SOL/token")
        logger.info(f"  Price Change: {price_change:+.2f}%")
        
        return result


async def listen_for_pumpfun_tokens_and_subscribe_to_trades():
    """Listen for new pump.fun tokens and subscribe to trades using single WebSocket."""
    print("🎧 Starting PumpPortal listener for pump.fun tokens...")
    print("   (This will stop when the first pump.fun token is found)")
    print()
    
    pumpportal_url = "wss://pumpportal.fun/api/data"
    found_token: Optional[TokenInfo] = None
    price_tracker: Optional[PriceTracker] = None
    
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
                            
                            # Initialize price tracker for this token
                            price_tracker = PriceTracker(str(token_info.mint))
                            price_tracker.initialize_from_token_creation()
                            print("💰 Price tracker initialized for bonding curve")
                            print()
                            break
                    
                except json.JSONDecodeError as e:
                    print(f"❌ Failed to parse JSON: {e}")
                    continue
                except Exception as e:
                    print(f"❌ Error processing message: {e}")
                    continue
            
            if found_token and price_tracker:
                # Now subscribe to trades for this token
                await subscribe_to_trades_for_token(websocket, found_token, price_tracker)
        
    except KeyboardInterrupt:
        print("\n⏹️  Stopped by user")
        return None
    except Exception as e:
        print(f"❌ Listener error: {e}")
        return None
    
    return found_token


async def subscribe_to_trades_for_token(websocket, token_info: TokenInfo, price_tracker: PriceTracker):
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
                await handle_trade_message_with_pricing(data, price_tracker)
            except json.JSONDecodeError as e:
                print(f"❌ Failed to parse JSON: {e}")
                print(f"   Raw message: {message[:200]}...")
            except Exception as e:
                print(f"❌ Error handling message: {e}")
                
    except websockets.exceptions.ConnectionClosed:
        print("🔌 Connection closed by server")
    except KeyboardInterrupt:
        print("\n⏹️  Stopped by user")


async def handle_trade_message_with_pricing(data: dict, price_tracker: PriceTracker):
    """Handle incoming trade messages with price calculation."""
    if "result" in data:
        # Subscription confirmation
        print(f"✅ Subscription confirmed: {data['result']}")
        return
    
    if "method" in data and "params" in data:
        # Trade event data
        trade_data = data.get("params", {})
        
        print(f"📊 Trade Event:")
        print(f"   Method: {data.get('method', 'unknown')}")
        print(f"   Raw Data: {json.dumps(trade_data, indent=4)}")
        
        # Try to extract trade information and calculate price
        try:
            # Look for trade information in the data
            if "solAmount" in trade_data or "tokenAmount" in trade_data:
                price_info = price_tracker.process_trade(trade_data)
                
                if price_info:
                    print(f"💰 Price Analysis:")
                    print(f"   Current Price: {price_info['price_sol']:.10f} SOL/token")
                    print(f"   Price Change: {price_info['price_change_percent']:+.2f}%")
                    print(f"   Trade Type: {price_info['trade_type']}")
                    print(f"   SOL Amount: {price_info['sol_amount']:.6f}")
                    print(f"   Token Amount: {price_info['token_amount']:.0f}")
                    print(f"   Virtual SOL: {price_info['virtual_sol']:.6f}")
                    print(f"   Virtual Tokens: {price_info['virtual_tokens']:.0f}")
                    print(f"   Real SOL: {price_info['real_sol']:.6f}")
                    print(f"   Real Tokens: {price_info['real_tokens']:.0f}")
                    print("-" * 60)
                else:
                    print("⚠️  Could not process trade for pricing")
                    print("-" * 40)
            else:
                print("⚠️  No trade amounts found in data")
                print("-" * 40)
                
        except Exception as e:
            print(f"❌ Error processing trade for pricing: {e}")
            print(f"   Raw Data: {json.dumps(trade_data, indent=2)}")
            print("-" * 40)
    else:
        # Other messages
        print(f"📨 Message: {json.dumps(data, indent=2)}")


async def handle_trade_message(data: dict):
    """Handle incoming trade messages (legacy function)."""
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
    print("🚀 PumpPortal Token Listener & Trade Subscriber with Price Tracking")
    print("=" * 70)
    print("This example will:")
    print("1. Listen for new pump.fun tokens on PumpPortal")
    print("2. Stop when the first one is found")
    print("3. Subscribe to that token's bonding curve trades")
    print("4. Calculate and display price changes after each trade")
    print("5. Track virtual and real reserves using bonding curve formula")
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
