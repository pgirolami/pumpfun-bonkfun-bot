"""
Test script for UniversalPumpPortalListener to debug connection issues.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from monitoring.universal_pumpportal_listener import UniversalPumpPortalListener
from interfaces.core import Platform, TokenInfo
from utils.logger import get_logger

# Set up console logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = get_logger(__name__)


async def test_token_callback(token_info: TokenInfo) -> None:
    """Simple callback to handle new tokens."""
    logger.info(f"🎯 NEW TOKEN DETECTED: {token_info.name} ({token_info.symbol})")
    logger.info(f"   Platform: {token_info.platform.value}")
    logger.info(f"   Mint: {token_info.mint}")
    logger.info(f"   Creator: {token_info.creator}")
    logger.info(f"   Bonding Curve: {token_info.bonding_curve}")
    logger.info("=" * 60)


async def test_trade_subscription():
    """Test trade subscription functionality."""
    logger.info("🧪 Testing trade subscription...")
    
    # Create listener
    listener = UniversalPumpPortalListener(
        platforms=[Platform.PUMP_FUN]
    )
    
    # Test subscription (this will fail because WebSocket isn't connected yet)
    test_mint = "4vcfXAiEyXFD4u8hoAL6h7bUPyZrSyeg1jHkVsuNpump"
    
    try:
        await listener.subscribe_token_trades(mint=test_mint)
        logger.info(f"✅ Successfully subscribed to {test_mint}")
        
        # Wait a bit
        await asyncio.sleep(2)
        
        # Test unsubscription
        await listener.unsubscribe_token_trades(mint=test_mint)
        logger.info(f"✅ Successfully unsubscribed from {test_mint}")
        
    except Exception as e:
        logger.exception(f"❌ Trade subscription test failed: {e}")
    
    logger.info("ℹ️  Note: Trade subscription requires active WebSocket connection")
    logger.info("ℹ️  This test shows the expected behavior when WebSocket is not connected")


async def test_listener():
    """Test the full listener functionality."""
    logger.info("🚀 Starting UniversalPumpPortalListener test...")
    
    # Create listener
    listener = UniversalPumpPortalListener(
        platforms=[Platform.PUMP_FUN]
    )
    
    # Test trade subscription within the listener context
    test_mint = "4vcfXAiEyXFD4u8hoAL6h7bUPyZrSyeg1jHkVsuNpump"
    
    async def enhanced_token_callback(token_info: TokenInfo) -> None:
        """Enhanced callback that tests trade subscription."""
        await test_token_callback(token_info)
        
        # Test trade subscription after token is detected
        logger.info("🧪 Testing trade subscription within listener context...")
        try:
            await listener.subscribe_token_trades(mint=str(token_info.mint))
            logger.info(f"✅ Successfully subscribed to {token_info.mint}")
            
            # Wait a bit
            await asyncio.sleep(3)
            
            # Test unsubscription
            await listener.unsubscribe_token_trades(mint=str(token_info.mint))
            logger.info(f"✅ Successfully unsubscribed from {token_info.mint}")
            
        except Exception as e:
            logger.exception(f"❌ Trade subscription failed: {e}")
    
    try:
        # Start listening (this will run indefinitely)
        await listener.listen_for_messages(
            token_callback=enhanced_token_callback,
            match_string=None,  # No filtering
            creator_address=None  # No creator filtering
        )
    except KeyboardInterrupt:
        logger.info("🛑 Test interrupted by user")
    except Exception as e:
        logger.exception(f"❌ Listener test failed: {e}")


async def main():
    """Main test function."""
    logger.info("=" * 60)
    logger.info("🧪 UNIVERSAL PUMPPORTAL LISTENER TEST")
    logger.info("=" * 60)
    
    try:
        # Test 1: Trade subscription (quick test)
        logger.info("\n📋 Test 1: Trade Subscription")
        await test_trade_subscription()
        logger.info("✅ Test 1 completed")
        
        # Test 2: Full listener (will run until interrupted)
        logger.info("\n📋 Test 2: Full Listener (Press Ctrl+C to stop)")
        logger.info("🔄 Starting WebSocket connection...")
        await test_listener()
        
    except KeyboardInterrupt:
        logger.info("🛑 Test interrupted by user")
    except Exception as e:
        logger.exception(f"❌ Test failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())
