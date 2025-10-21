#!/usr/bin/env python3
"""
Demonstration of blockhash caching optimization.

This example shows how the SolanaClient caches blockhashes to reduce RPC calls,
compared to making direct getLatestBlockhash() calls.
"""

import asyncio
import time
from solana.rpc.async_api import AsyncClient
from solana.rpc.commitment import Processed

# Import our optimized client
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from core.client import SolanaClient
from utils.logger import get_logger

logger = get_logger(__name__)

async def demo_direct_rpc_calls(rpc_endpoint: str, num_calls: int = 5):
    """Demonstrate direct RPC calls (inefficient)."""
    print(f"\n=== Direct RPC Calls (Inefficient) ===")
    start_time = time.time()
    
    async with AsyncClient(rpc_endpoint) as client:
        for i in range(num_calls):
            start_call = time.time()
            response = await client.get_latest_blockhash(commitment=Processed)
            call_time = time.time() - start_call
            print(f"Call {i+1}: {call_time:.3f}s - Blockhash: {response.value.blockhash[:8]}...")
    
    total_time = time.time() - start_time
    print(f"Total time for {num_calls} calls: {total_time:.3f}s")
    return total_time

async def demo_cached_blockhash(rpc_endpoint: str, num_calls: int = 5, update_interval: float = 10.0):
    """Demonstrate cached blockhash usage (efficient)."""
    print(f"\n=== Cached Blockhash (Efficient) ===")
    start_time = time.time()
    
    # Create SolanaClient with caching
    client = SolanaClient(rpc_endpoint, blockhash_update_interval=update_interval)
    
    try:
        # Wait a moment for initial blockhash to be cached
        await asyncio.sleep(1)
        
        for i in range(num_calls):
            start_call = time.time()
            blockhash = await client.get_cached_blockhash()
            call_time = time.time() - start_call
            print(f"Call {i+1}: {call_time:.3f}s - Blockhash: {blockhash[:8]}...")
        
        total_time = time.time() - start_time
        print(f"Total time for {num_calls} calls: {total_time:.3f}s")
        return total_time
    
    finally:
        await client.close()

async def main():
    """Run the blockhash optimization demonstration."""
    # Get RPC endpoint from environment
    import os
    rpc_endpoint = os.getenv("SOLANA_NODE_RPC_ENDPOINT")
    if not rpc_endpoint:
        print("Please set SOLANA_NODE_RPC_ENDPOINT environment variable")
        return
    
    print("Blockhash Caching Optimization Demo")
    print("=" * 50)
    
    num_calls = 5
    
    # Test direct RPC calls
    direct_time = await demo_direct_rpc_calls(rpc_endpoint, num_calls)
    
    # Test cached approach
    cached_time = await demo_cached_blockhash(rpc_endpoint, num_calls, update_interval=5.0)
    
    # Calculate improvement
    improvement = ((direct_time - cached_time) / direct_time) * 100
    print(f"\n=== Results ===")
    print(f"Direct RPC calls: {direct_time:.3f}s")
    print(f"Cached calls: {cached_time:.3f}s")
    print(f"Improvement: {improvement:.1f}% faster")
    print(f"RPC calls saved: {num_calls - 1} (only 1 actual RPC call vs {num_calls})")
    
    print(f"\n=== Configuration Tips ===")
    print(f"• Default update interval: 10 seconds")
    print(f"• Solana blockhashes are valid for ~150 slots (~60-90 seconds)")
    print(f"• Higher intervals = fewer RPC calls but slightly older blockhashes")
    print(f"• Recommended: 15-30 seconds for most use cases")

if __name__ == "__main__":
    asyncio.run(main())

