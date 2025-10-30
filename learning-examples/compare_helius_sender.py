"""
Compare Helius Sender vs Standard Solana RPC transaction submission.

This script buys 0.01 SOL worth of a token, waits 5 seconds, then sells all tokens
using both send_method="solana" and send_method="helius_sender" to compare latency.
"""

import asyncio
import os
import struct
import time
import logging

import base58
from construct import Bytes, Flag, Int64ul, Struct
from dotenv import load_dotenv
from solana.rpc.async_api import AsyncClient
from solana.rpc.commitment import Confirmed
from solana.rpc.types import TxOpts
from solders.compute_budget import set_compute_unit_limit, set_compute_unit_price
from solders.instruction import AccountMeta, Instruction
from solders.keypair import Keypair
from solders.message import Message
from solders.pubkey import Pubkey
from solders.transaction import Transaction
from spl.token.instructions import (
    create_idempotent_associated_token_account,
    get_associated_token_address,
)

from core.client import SolanaClient
from interfaces.core import Platform, TokenInfo
from platforms import get_platform_implementations

load_dotenv()

# Configure basic console logging for this script
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)
# Enable HTTPX/HTTPCore INFO logs to console
logging.getLogger("httpx").setLevel(logging.INFO)
logging.getLogger("httpcore").setLevel(logging.INFO)

# Constants
TOKEN_MINT = Pubkey.from_string("9r4Hkiqc7TmbEMVSUir5VeXsvX4S2vfdwCqPMwLZpump")
TOKEN_DECIMALS = 6
BUY_AMOUNT_SOL = 0.01

PUMP_PROGRAM = Pubkey.from_string("6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P")
PUMP_GLOBAL = Pubkey.from_string("4wTV1YmiEkRvAtNtsSGPtUrqRYQMe5SKy2uB4Jjaxnjf")
PUMP_FEE = Pubkey.from_string("CebN5WGQ4jvEPvsVU4EoHEpgzq1VV7AbicfhtW4xC9iM")
SYSTEM_PROGRAM = Pubkey.from_string("11111111111111111111111111111111")
SYSTEM_TOKEN_PROGRAM = Pubkey.from_string("TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA")
SYSTEM_ASSOCIATED_TOKEN_ACCOUNT_PROGRAM = Pubkey.from_string(
    "ATokenGPvbdGVxr1b2hvZbsiqW5xWH25efTNsLJA8knL"
)
LAMPORTS_PER_SOL = 1_000_000_000
EXPECTED_DISCRIMINATOR = struct.pack("<Q", 6966180631402821399)

RPC_ENDPOINT = os.environ.get("SOLANA_NODE_RPC_ENDPOINT")
WSS_ENDPOINT = os.environ.get("SOLANA_NODE_WSS_ENDPOINT")
PRIVATE_KEY = base58.b58decode(os.environ.get("SOLANA_PRIVATE_KEY"))
PAYER = Keypair.from_bytes(PRIVATE_KEY)

TOKEN_BALANCE_RAW = 35766666666

class BondingCurveState:
    """Represents the state of a bonding curve account."""
    
    _STRUCT = Struct(
        "virtual_token_reserves" / Int64ul,
        "virtual_sol_reserves" / Int64ul,
        "real_token_reserves" / Int64ul,
        "real_sol_reserves" / Int64ul,
        "token_total_supply" / Int64ul,
        "complete" / Flag,
        "creator" / Bytes(32),  # Creator field - 32 bytes for Pubkey
    )

    def __init__(self, data: bytes) -> None:
        """Parse bonding curve data."""
        if data[:8] != EXPECTED_DISCRIMINATOR:
            raise ValueError("Invalid curve state discriminator")

        parsed = self._STRUCT.parse(data[8:])
        self.__dict__.update(parsed)
        
        # Convert raw bytes to Pubkey for creator field
        if hasattr(self, "creator") and isinstance(self.creator, bytes):
            self.creator = Pubkey.from_bytes(self.creator)


async def get_bonding_curve_state(
    conn: AsyncClient, curve_address: Pubkey
) -> BondingCurveState:
    """Fetch and parse bonding curve state."""
    response = await conn.get_account_info(curve_address, encoding="base64")
    if not response.value or not response.value.data:
        raise ValueError("Invalid curve state: No data")

    data = response.value.data
    if data[:8] != EXPECTED_DISCRIMINATOR:
        raise ValueError("Invalid curve state discriminator")

    return BondingCurveState(data)


def get_bonding_curve_address(mint: Pubkey) -> tuple[Pubkey, int]:
    return Pubkey.find_program_address([b"bonding-curve", bytes(mint)], PUMP_PROGRAM)


def find_associated_bonding_curve(mint: Pubkey, bonding_curve: Pubkey) -> Pubkey:
    derived_address, _ = Pubkey.find_program_address(
        [bytes(bonding_curve), bytes(SYSTEM_TOKEN_PROGRAM), bytes(mint)],
        SYSTEM_ASSOCIATED_TOKEN_ACCOUNT_PROGRAM,
    )
    return derived_address


def find_creator_vault(creator: Pubkey) -> Pubkey:
    derived_address, _ = Pubkey.find_program_address(
        [b"creator-vault", bytes(creator)], PUMP_PROGRAM
    )
    return derived_address


async def buy_token(client: SolanaClient, mint: Pubkey, amount_sol: float) -> tuple[str, int, float]:
    """Buy tokens and return transaction signature and token amount received."""
    async with AsyncClient(RPC_ENDPOINT) as async_client:
        bonding_curve, _ = get_bonding_curve_address(mint)
        associated_bonding_curve = find_associated_bonding_curve(mint, bonding_curve)
        
        # Fetch bonding curve state to get creator address
        curve_state = await get_bonding_curve_state(async_client, bonding_curve)
        creator = curve_state.creator
        
        # Derive creator vault from creator
        creator_vault = find_creator_vault(creator)
        
        # Get platform implementations
        platform_impls = get_platform_implementations(Platform.PUMP_FUN, client)
        address_provider = platform_impls.address_provider
        instruction_builder = platform_impls.instruction_builder
        
        # Create TokenInfo for instruction builder
        token_info = TokenInfo(
            name="",
            symbol="",
            uri="",
            mint=mint,
            platform=Platform.PUMP_FUN,
            bonding_curve=bonding_curve,
            associated_bonding_curve=associated_bonding_curve,
            user=PAYER.pubkey(),
            creator=creator,
            creator_vault=creator_vault,
        )
        
        amount_lamports = int(amount_sol * LAMPORTS_PER_SOL)
                
        # Build buy instruction using platform implementation
        instructions = await instruction_builder.build_buy_instruction(
            token_info=token_info,
            user=PAYER.pubkey(),
            amount_in=int(0.01*LAMPORTS_PER_SOL),
            minimum_amount_out=TOKEN_BALANCE_RAW,
            address_provider=address_provider,
        )
        
        start_time = time.time()
        send_start = time.time()
        signature = await client.build_and_send_transaction(
            instructions,
            PAYER,
            skip_preflight=True,
            max_retries=3,
            priority_fee=200_000,
            compute_unit_limit=100_000,
        )
        send_elapsed = time.time() - send_start
        
        # Wait for confirmation
        result = await client.confirm_transaction(signature)
        elapsed = time.time() - start_time
        
        if not result.success:
            raise Exception(f"Transaction failed: {result.error_message}")
        
        # Get token balance
        user_ata = get_associated_token_address(PAYER.pubkey(), mint)
        balance = await client.get_token_account_balance(user_ata)
        
        return str(signature), balance, send_elapsed


async def sell_all_tokens(
    client: SolanaClient,
    mint: Pubkey,
) -> tuple[str, float]:
    """Sell all tokens with minimum SOL out set to 0 lamports."""
    async with AsyncClient(RPC_ENDPOINT) as async_client:
        bonding_curve, _ = get_bonding_curve_address(mint)
        associated_bonding_curve = find_associated_bonding_curve(mint, bonding_curve)
        
        # Fetch bonding curve state to get creator address
        curve_state = await get_bonding_curve_state(async_client, bonding_curve)
        creator = curve_state.creator
        
        # Derive creator vault from creator
        creator_vault = find_creator_vault(creator)
        
        # Get platform implementations
        platform_impls = get_platform_implementations(Platform.PUMP_FUN, client)
        address_provider = platform_impls.address_provider
        instruction_builder = platform_impls.instruction_builder
        
        # Create TokenInfo for instruction builder
        token_info = TokenInfo(
            name="",
            symbol="",
            uri="",
            mint=mint,
            platform=Platform.PUMP_FUN,
            bonding_curve=bonding_curve,
            associated_bonding_curve=associated_bonding_curve,
            user=PAYER.pubkey(),
            creator=creator,
            creator_vault=creator_vault,
        )
                        
        instructions = await instruction_builder.build_sell_instruction(
            token_info=token_info,
            user=PAYER.pubkey(),
            amount_in=TOKEN_BALANCE_RAW,
            minimum_amount_out=0,
            address_provider=address_provider,
        )
        
        start_time = time.time()
        send_start = time.time()
        signature = await client.build_and_send_transaction(
            instructions,
            PAYER,
            skip_preflight=True,
            max_retries=3,
            priority_fee=200_000,
            compute_unit_limit=60_000,
        )
        send_elapsed = time.time() - send_start
        
        result = await client.confirm_transaction(signature)
        elapsed = time.time() - start_time
        
        if not result.success:
            raise Exception(f"Transaction failed: {result.error_message}")
        
        return str(signature), send_elapsed


async def run_comparison():
    """Run comparison between standard Solana RPC and Helius Sender."""
    logger.info("%s", "=" * 80)
    logger.info("Helius Sender vs Standard Solana RPC Comparison")
    logger.info("%s", "=" * 80)
    logger.info("Token: %s", TOKEN_MINT)
    logger.info("Buy Amount: %s SOL", BUY_AMOUNT_SOL)
    logger.info("")
    
    # Setup Standard Solana RPC client
    rpc_config_chainstack = {
        "rpc_endpoint": RPC_ENDPOINT,
        "wss_endpoint": WSS_ENDPOINT,
        "send_method": "solana",
    }

    rpc_config_helius = {
        "rpc_endpoint": "https://mainnet.helius-rpc.com/?api-key=5b5d0eec-c92c-4ae9-a7f3-c02fdc504290",
        "wss_endpoint": "wss://mainnet.helius-rpc.com/?api-key=5b5d0eec-c92c-4ae9-a7f3-c02fdc504290",
        "send_method": "solana",
    }

    rpc_config_helius_sender = {
        "rpc_endpoint": "https://mainnet.helius-rpc.com/?api-key=5b5d0eec-c92c-4ae9-a7f3-c02fdc504290",
        "wss_endpoint": "wss://mainnet.helius-rpc.com/?api-key=5b5d0eec-c92c-4ae9-a7f3-c02fdc504290",
        "send_method": "helius_sender",
        "helius_sender": {
            "routing": "swqos_only",
            "tip_amount_sol": 0.000005,
        },
    }

    # Setup Helius Sender client
    rpc_config_helius_sender_chainstack = {
        "rpc_endpoint": RPC_ENDPOINT,
        "wss_endpoint": WSS_ENDPOINT,
        "send_method": "helius_sender",
        "helius_sender": {
            "routing": "swqos_only",
            "tip_amount_sol": 0.000005,
        },
    }
    
    results = {}
    
    for method_name, rpc_config in [("Chainstack Free RPC", rpc_config_chainstack), ("Helius Free Solana RPC", rpc_config_helius),("Helius Sender + Chainstack Free RPC", rpc_config_helius_sender_chainstack),("Helius Sender + Helius Free RPC", rpc_config_helius_sender)]:
        logger.info("\n%s:", method_name)
        logger.info("%s", "-" * 40)
        
        try:
            client = SolanaClient(rpc_config)

            logger.info("  Waiting 5 seconds...")
            await asyncio.sleep(5)

            logger.info("  Warming blockhash cache")
            await client.get_cached_blockhash()

            # Buy
            logger.info("  Buying...")
            buy_start = time.time()
            buy_sig, token_amount, buy_send_time = await buy_token(
                client, TOKEN_MINT, BUY_AMOUNT_SOL
            )
            buy_elapsed = time.time() - buy_start
            logger.info("  Buy: %s... (send: %.3fs, total: %.3fs)", buy_sig[:16], buy_send_time, buy_elapsed)
                        
            # Sell
            logger.info("  Selling...")
            sell_start = time.time()
            sell_sig, sell_send_time = await sell_all_tokens(
                client,
                TOKEN_MINT,
            )
            sell_elapsed = time.time() - sell_start
            logger.info("  Sell: %s... (send: %.3fs, total: %.3fs)", sell_sig[:16], sell_send_time, sell_elapsed)
            
            total_time = buy_elapsed + sell_elapsed
            results[method_name] = {
                "buy_time": buy_elapsed,
                "buy_send_time": buy_send_time,
                "sell_time": sell_elapsed,
                "sell_send_time": sell_send_time,
                "total_time": total_time,
            }
            logger.info("  Total time: %.3fs", total_time)

            await client.close()
            
            
        except Exception as e:
            logger.exception("%s: ERROR during run", method_name)
            results[method_name] = {"error": str(e)}
        finally:
            await client.close()
    
    # Summary
    logger.info("\n%s", "=" * 80)
    logger.info("Summary:")
    logger.info("%s", "=" * 80)
    for method, metrics in results.items():
        if "error" in metrics:
            logger.info("%s: FAILED - %s", method, metrics["error"])
        else:
            logger.info(
                "%s: Buy: send=%.3fs, total=%.3fs | Sell: send=%.3fs, total=%.3fs | Total=%.3fs",
                method,
                metrics["buy_send_time"],
                metrics["buy_time"],
                metrics["sell_send_time"],
                metrics["sell_time"],
                metrics["total_time"],
            )

if __name__ == "__main__":
    asyncio.run(run_comparison())

