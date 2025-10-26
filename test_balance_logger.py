#!/usr/bin/env python3
"""
Test script for wallet balance logging functionality.
"""

import asyncio
import os
from pathlib import Path

# Add src to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent / "src"))

from utils.wallet_balance_logger import WalletBalanceLogger
from core.wallet import Wallet
from solders.pubkey import Pubkey


async def test_balance_logger():
    """Test the wallet balance logger functionality."""
    print("Testing Wallet Balance Logger...")
    
    # Get RPC endpoint from environment
    rpc_endpoint = os.getenv("SOLANA_RPC_HTTP")
    if not rpc_endpoint:
        print("Error: SOLANA_RPC_HTTP environment variable not set")
        return
    
    # Get private key from environment
    private_key = os.getenv("SOLANA_PRIVATE_KEY")
    if not private_key:
        print("Error: SOLANA_PRIVATE_KEY environment variable not set")
        return
    
    # Create wallet
    wallet = Wallet(private_key)
    print(f"Testing with wallet: {wallet.pubkey}")
    
    # Create balance logger
    balance_logger = WalletBalanceLogger(rpc_endpoint)
    
    try:
        # Register wallet
        balance_logger.register_wallet(wallet.pubkey, "test-bot")
        
        # Test single balance log
        print("\nTesting single balance log...")
        await balance_logger.log_wallet_balances()
        
        # Test monitoring for a short period
        print("\nTesting balance monitoring for 10 seconds...")
        await balance_logger.start_balance_monitoring(interval_minutes=0.1)  # Every 6 seconds
        
        # Wait for a few logs
        await asyncio.sleep(15)
        
        # Stop monitoring
        await balance_logger.stop_balance_monitoring()
        
        print("\nTest completed successfully!")
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        await balance_logger.close()


if __name__ == "__main__":
    asyncio.run(test_balance_logger())
