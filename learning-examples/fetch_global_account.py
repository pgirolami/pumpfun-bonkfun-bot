#!/usr/bin/env python3
"""
Fetch pump.fun global account data to see the initial reserves constants.

This script demonstrates how to fetch and parse the pump.fun global account
which contains the initial virtual reserves and other constants used for
all bonding curves.
"""

import asyncio
import base64
import struct
from solders.pubkey import Pubkey
from solana.rpc.async_api import AsyncClient

# Pump.fun constants
PUMP_GLOBAL = Pubkey.from_string("4wTV1YmiEkRvAtNtsSGPtUrqRYQMe5SKy2uB4Jjaxnjf")
LAMPORTS_PER_SOL = 1_000_000_000
TOKEN_DECIMALS = 6

# Expected discriminator for Global account (from IDL)
GLOBAL_DISCRIMINATOR = bytes([167, 232, 232, 177, 200, 108, 114, 127])


def parse_global_account_data(data: bytes) -> dict:
    """Parse the pump.fun global account data.
    
    Args:
        data: Raw account data from RPC
        
    Returns:
        Dictionary containing parsed global account fields
    """
    if len(data) < 8:
        raise ValueError("Account data too short")
    
    # Check discriminator
    discriminator = data[:8]
    if discriminator != GLOBAL_DISCRIMINATOR:
        raise ValueError(f"Invalid discriminator: {discriminator.hex()}")
    
    # Parse fields according to IDL structure
    offset = 8
    fields = {}
    
    # initialized (bool)
    fields["initialized"] = bool(data[offset])
    offset += 1
    
    # authority (pubkey)
    fields["authority"] = Pubkey.from_bytes(data[offset:offset + 32])
    offset += 32
    
    # fee_recipient (pubkey)
    fields["fee_recipient"] = Pubkey.from_bytes(data[offset:offset + 32])
    offset += 32
    
    # initial_virtual_token_reserves (u64)
    fields["initial_virtual_token_reserves"] = struct.unpack("<Q", data[offset:offset + 8])[0]
    offset += 8
    
    # initial_virtual_sol_reserves (u64)
    fields["initial_virtual_sol_reserves"] = struct.unpack("<Q", data[offset:offset + 8])[0]
    offset += 8
    
    # initial_real_token_reserves (u64)
    fields["initial_real_token_reserves"] = struct.unpack("<Q", data[offset:offset + 8])[0]
    offset += 8
    
    # token_total_supply (u64)
    fields["token_total_supply"] = struct.unpack("<Q", data[offset:offset + 8])[0]
    offset += 8
    
    # fee_basis_points (u64)
    fields["fee_basis_points"] = struct.unpack("<Q", data[offset:offset + 8])[0]
    offset += 8
    
    # withdraw_authority (pubkey)
    fields["withdraw_authority"] = Pubkey.from_bytes(data[offset:offset + 32])
    offset += 32
    
    # enable_migrate (bool)
    fields["enable_migrate"] = bool(data[offset])
    offset += 1
    
    return fields


async def main():
    """Fetch and display pump.fun global account data."""
    # Use environment variable or default RPC
    import os
    rpc_endpoint = os.getenv("SOLANA_NODE_RPC_ENDPOINT", "https://api.mainnet-beta.solana.com")
    
    print("Fetching pump.fun global account data...")
    print(f"Global account: {PUMP_GLOBAL}")
    print(f"RPC endpoint: {rpc_endpoint}")
    print()
    
    async with AsyncClient(rpc_endpoint) as client:
        try:
            # Fetch account info
            response = await client.get_account_info(PUMP_GLOBAL, encoding="base64")
            
            if not response.value:
                print("❌ Global account not found!")
                return
            
            if not response.value.data:
                print("❌ No data in global account!")
                return
            
            # Data is already decoded when using base64 encoding
            raw_data = response.value.data
            
            # Parse the data
            global_data = parse_global_account_data(raw_data)
            
            print("✅ Global account data parsed successfully!")
            print()
            print("=" * 60)
            print("PUMP.FUN GLOBAL ACCOUNT DATA")
            print("=" * 60)
            print(f"Initialized: {global_data['initialized']}")
            print(f"Authority: {global_data['authority']}")
            print(f"Fee Recipient: {global_data['fee_recipient']}")
            print(f"Withdraw Authority: {global_data['withdraw_authority']}")
            print(f"Enable Migrate: {global_data['enable_migrate']}")
            print()
            print("INITIAL RESERVES (Constants for all tokens):")
            print("-" * 40)
            
            # Convert to human-readable format
            initial_virtual_tokens = global_data["initial_virtual_token_reserves"] / 10**TOKEN_DECIMALS
            initial_virtual_sol = global_data["initial_virtual_sol_reserves"] / LAMPORTS_PER_SOL
            initial_real_tokens = global_data["initial_real_token_reserves"] / 10**TOKEN_DECIMALS
            total_supply = global_data["token_total_supply"] / 10**TOKEN_DECIMALS
            
            print(f"Initial Virtual Token Reserves: {initial_virtual_tokens:,.0f} tokens")
            print(f"Initial Virtual SOL Reserves: {initial_virtual_sol:,.0f} SOL")
            print(f"Initial Real Token Reserves: {initial_real_tokens:,.0f} tokens")
            print(f"Token Total Supply: {total_supply:,.0f} tokens")
            print()
            
            # Calculate starting price
            starting_price = initial_virtual_sol / initial_virtual_tokens
            print("STARTING PRICE CALCULATION:")
            print("-" * 30)
            print(f"Starting Price = {initial_virtual_sol} SOL / {initial_virtual_tokens:,.0f} tokens")
            print(f"Starting Price = {starting_price:.8f} SOL per token")
            print(f"Starting Price = ${starting_price * 200:.6f} USD (at $200 SOL)")
            print()
            
            # Fee information
            fee_bps = global_data["fee_basis_points"]
            fee_percentage = fee_bps / 100
            print("FEE INFORMATION:")
            print("-" * 15)
            print(f"Fee Basis Points: {fee_bps}")
            print(f"Fee Percentage: {fee_percentage:.2f}%")
            print()
            
            print("=" * 60)
            print("✅ All pump.fun tokens start with these exact same values!")
            print("=" * 60)
            
        except Exception as e:
            print(f"❌ Error fetching global account: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
