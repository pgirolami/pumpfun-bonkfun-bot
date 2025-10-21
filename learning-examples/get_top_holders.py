"""
Learning example: Get top token holders with percentage ownership analysis.

This script demonstrates how to fetch the top 10 token holders for any given mint address
and calculate what percentage of the total supply each holder owns.

Usage:
    uv run learning-examples/get_top_holders.py [MINT_ADDRESS]
    
Example:
    uv run learning-examples/get_top_holders.py 7GCihgDB8fe6KNjn2MYtkzZcRjQy3t9GHdC8uHYmW2hr
"""

import asyncio
import os
import sys
import time
from typing import Final

import uvloop
from dotenv import load_dotenv
from httpx import HTTPStatusError
from solana.rpc.async_api import AsyncClient
from solana.exceptions import SolanaRpcException
from solders.pubkey import Pubkey

load_dotenv()

# Constants
RPC_ENDPOINT: Final[str] = os.environ.get("SOLANA_NODE_RPC_ENDPOINT")
DEFAULT_MINT: Final[str] = "7GCihgDB8fe6KNjn2MYtkzZcRjQy3t9GHdC8uHYmW2hr"  # Example pump.fun token

# Alternative RPC endpoints (if you have access)
ALTERNATIVE_RPC_ENDPOINTS = [
    "https://api.mainnet-beta.solana.com",  # Default (rate limited)
    "https://solana-api.projectserum.com",  # Alternative
    "https://rpc.ankr.com/solana",  # Ankr
    "https://solana-mainnet.g.alchemy.com/v2/demo",  # Alchemy (demo)
]


async def get_token_largest_accounts_with_retry(client: AsyncClient, mint: Pubkey, max_retries: int = 5):
    """
    Get token largest accounts with retry logic for rate limiting.
    
    Args:
        client: AsyncClient instance
        mint: Token mint address
        max_retries: Maximum number of retry attempts
        
    Returns:
        RPC response with largest accounts
    """
    for attempt in range(max_retries + 1):
        try:
            return await client.get_token_largest_accounts(mint)
        except Exception as e:
            error_str = str(e)
            is_rate_limited = (
                "429" in error_str or 
                "Too Many Requests" in error_str or 
                "HTTPStatusError" in error_str or
                "Client error '429" in error_str
            )
            
            if is_rate_limited and attempt < max_retries:
                wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s, 8s, 16s
                print(f"  Rate limited, waiting {wait_time}s before retry {attempt + 1}/{max_retries}...")
                await asyncio.sleep(wait_time)
                continue
            
            # If it's not a rate limit error, or we've exhausted retries, re-raise
            raise e
    
    raise Exception(f"Failed after {max_retries} retries")


async def get_top_holders_with_percentage(
    mint_address: str, client: AsyncClient, top_n: int = 10
) -> dict:
    """
    Get top token holders with percentage ownership analysis.
    
    Args:
        mint_address: The token mint address to analyze
        client: AsyncClient instance
        top_n: Number of top holders to return (default: 10)
        
    Returns:
        Dictionary containing holder data and total supply info
    """
    try:
        mint = Pubkey.from_string(mint_address)
    except Exception as e:
        raise ValueError(f"Invalid mint address '{mint_address}': {e}")
    
    # Get token supply information
    print(f"  Fetching token supply for {mint_address}...")
    supply_response = await client.get_token_supply(mint)
    if not supply_response.value:
        raise ValueError(f"Could not fetch token supply for {mint_address}")
    
    total_supply = int(supply_response.value.amount)
    decimals = supply_response.value.decimals
    print(f"  Total supply: {total_supply}, decimals: {decimals}")
    
    # Add a delay between calls to avoid rate limiting
    print(f"  Waiting 3s before fetching largest accounts to avoid rate limits...")
    await asyncio.sleep(3)
    
    # Get largest token accounts
    print(f"  Fetching largest token accounts...")
    largest_response = await get_token_largest_accounts_with_retry(client, mint, max_retries=5)
    if not largest_response.value:
        raise ValueError(f"No token accounts found for {mint_address}")
    
    print(f"  Found {len(largest_response.value)} token accounts")
    
    # Process top holders
    holders = []
    cumulative_percentage = 0.0
    
    for i, account in enumerate(largest_response.value[:top_n]):
        # Handle both string and UiTokenAmount types
        if hasattr(account, 'amount'):
            if hasattr(account.amount, 'amount'):
                # UiTokenAmount object
                balance = int(account.amount.amount)
            else:
                # Direct amount value
                balance = int(account.amount)
        else:
            # Fallback
            balance = int(account)
            
        percentage = (balance / total_supply) * 100 if total_supply > 0 else 0
        cumulative_percentage += percentage
        
        holders.append({
            "rank": i + 1,
            "token_account": str(account.address),
            "raw_balance": balance,
            "formatted_balance": balance / (10**decimals),
            "percentage": percentage,
            "cumulative_percentage": cumulative_percentage,
        })
    
    return {
        "mint_address": mint_address,
        "total_supply": total_supply,
        "formatted_total_supply": total_supply / (10**decimals),
        "decimals": decimals,
        "holders": holders,
    }


def format_output(data: dict) -> None:
    """
    Format and display the token holder analysis results.
    
    Args:
        data: Dictionary containing holder analysis data
    """
    print(f"\nTop {len(data['holders'])} Token Holders for {data['mint_address']}")
    print(f"Total Supply: {data['formatted_total_supply']:,.0f} tokens")
    print("=" * 80)
    print(f"{'Rank':<4} | {'Token Account':<44} | {'Balance':<15} | {'% Supply':<8} | {'Cumulative %':<12}")
    print("=" * 80)
    
    for holder in data["holders"]:
        print(
            f"{holder['rank']:<4} | "
            f"{holder['token_account']:<44} | "
            f"{holder['formatted_balance']:>14,.0f} | "
            f"{holder['percentage']:>7.2f}% | "
            f"{holder['cumulative_percentage']:>11.2f}%"
        )
    
    print("=" * 80)
    print(f"Top {len(data['holders'])} holders control {data['holders'][-1]['cumulative_percentage']:.2f}% of total supply")


async def main() -> None:
    """Main entry point for the token holder analysis script."""
    # Parse command line arguments
    mint_address = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MINT
    
    if not RPC_ENDPOINT:
        print("Error: SOLANA_NODE_RPC_ENDPOINT environment variable not set")
        print("Please set your RPC endpoint in .env file or environment")
        print("\nRecommended RPC endpoints:")
        for endpoint in ALTERNATIVE_RPC_ENDPOINTS:
            print(f"  - {endpoint}")
        sys.exit(1)
    
    print(f"Analyzing token holders for: {mint_address}")
    print(f"Using RPC endpoint: {RPC_ENDPOINT}")
    
    if "api.mainnet-beta.solana.com" in RPC_ENDPOINT:
        print("⚠️  Warning: Using public RPC endpoint which has rate limits.")
        print("   Consider using a paid RPC provider for better performance.")
    
    # Try multiple RPC endpoints if the first one fails
    rpc_endpoints_to_try = [RPC_ENDPOINT]
    if "api.mainnet-beta.solana.com" in RPC_ENDPOINT:
        # Add fallback endpoints for public RPC
        rpc_endpoints_to_try.extend([
            "https://rpc.ankr.com/solana",
            "https://solana-mainnet.g.alchemy.com/v2/demo"
        ])
    
    last_error = None
    for i, endpoint in enumerate(rpc_endpoints_to_try):
        try:
            print(f"Trying RPC endpoint {i+1}/{len(rpc_endpoints_to_try)}: {endpoint}")
            async with AsyncClient(endpoint, commitment="confirmed", timeout=60) as client:
                await client.is_connected()
                print("Connected to Solana RPC")
                
                print("Fetching token supply...")
                # Get top holders with percentage analysis
                data = await get_top_holders_with_percentage(mint_address, client)
                
                print("Displaying results...")
                # Display results
                format_output(data)
                return  # Success, exit the function
                
        except Exception as e:
            last_error = e
            print(f"Failed with endpoint {endpoint}: {e}")
            if i < len(rpc_endpoints_to_try) - 1:
                print("Trying next endpoint...")
                await asyncio.sleep(2)  # Brief delay before trying next endpoint
            continue
    
    # If we get here, all endpoints failed
    print(f"All RPC endpoints failed. Last error: {last_error}")
    sys.exit(1)


if __name__ == "__main__":
    # Set uvloop policy for better async performance
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    asyncio.run(main())
