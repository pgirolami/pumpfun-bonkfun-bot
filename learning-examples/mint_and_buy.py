import asyncio
import os
import struct
from typing import Final

import base58
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

# Configuration for the token to be created
TOKEN_NAME = "Test Token"
TOKEN_SYMBOL = "TEST"
TOKEN_URI = "https://example.com/token.json"
BUY_AMOUNT_SOL = 0.001  # Amount of SOL to spend on buying
MAX_SLIPPAGE = 0.3  # 30% slippage
PRIORITY_FEE_MICROLAMPORTS = 37_037  # Priority fee in microlamports
COMPUTE_UNIT_LIMIT = 250_000  # Compute unit limit for the transaction

load_dotenv()

# Global constants from existing codebase
PUMP_PROGRAM: Final[Pubkey] = Pubkey.from_string(
    "6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P"
)
PUMP_GLOBAL: Final[Pubkey] = Pubkey.from_string(
    "4wTV1YmiEkRvAtNtsSGPtUrqRYQMe5SKy2uB4Jjaxnjf"
)
PUMP_EVENT_AUTHORITY: Final[Pubkey] = Pubkey.from_string(
    "Ce6TQqeHC9p8KetsN6JsjHK7UTZk7nasjjnr7XxXp9F1"
)
PUMP_FEE: Final[Pubkey] = Pubkey.from_string(
    "CebN5WGQ4jvEPvsVU4EoHEpgzq1VV7AbicfhtW4xC9iM"
)
PUMP_FEE_PROGRAM: Final[Pubkey] = Pubkey.from_string(
    "pfeeUxB6jkeY1Hxd7CsFCAjcbHA9rWtchMGdZ6VojVZ"
)
PUMP_MINT_AUTHORITY: Final[Pubkey] = Pubkey.from_string(
    "TSLvdd1pWpHVjahSpsvCXUbgwsL3JAcvokwaKt1eokM"
)

SYSTEM_PROGRAM: Final[Pubkey] = Pubkey.from_string("11111111111111111111111111111111")
SYSTEM_TOKEN_PROGRAM: Final[Pubkey] = Pubkey.from_string(
    "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"
)
SYSTEM_ASSOCIATED_TOKEN_ACCOUNT_PROGRAM: Final[Pubkey] = Pubkey.from_string(
    "ATokenGPvbdGVxr1b2hvZbsiqW5xWH25efTNsLJA8knL"
)
SYSTEM_RENT: Final[Pubkey] = Pubkey.from_string(
    "SysvarRent111111111111111111111111111111111"
)
METAPLEX_TOKEN_METADATA: Final[Pubkey] = Pubkey.from_string(
    "metaqbxxUerdq28cj1RbAWkYQm3ybzjb6a8bt518x1s"
)

LAMPORTS_PER_SOL: Final[int] = 1_000_000_000
TOKEN_DECIMALS: Final[int] = 6

# Discriminators
CREATE_DISCRIMINATOR: Final[bytes] = struct.pack("<Q", 8576854823835016728)
BUY_DISCRIMINATOR: Final[bytes] = struct.pack("<Q", 16927863322537952870)
EXTEND_ACCOUNT_DISCRIMINATOR: Final[bytes] = bytes(
    [234, 102, 194, 203, 150, 72, 62, 229]
)

# From environment
RPC_ENDPOINT = os.environ.get("SOLANA_NODE_RPC_ENDPOINT")
PRIVATE_KEY = os.environ.get("SOLANA_PRIVATE_KEY")


def find_bonding_curve_address(mint: Pubkey) -> tuple[Pubkey, int]:
    """Find the bonding curve PDA for a mint."""
    return Pubkey.find_program_address([b"bonding-curve", bytes(mint)], PUMP_PROGRAM)


def find_associated_bonding_curve(mint: Pubkey, bonding_curve: Pubkey) -> Pubkey:
    """Find the associated bonding curve token account."""
    derived_address, _ = Pubkey.find_program_address(
        [
            bytes(bonding_curve),
            bytes(SYSTEM_TOKEN_PROGRAM),
            bytes(mint),
        ],
        SYSTEM_ASSOCIATED_TOKEN_ACCOUNT_PROGRAM,
    )
    return derived_address


def find_metadata_address(mint: Pubkey) -> Pubkey:
    """Find the metadata PDA for a mint."""
    derived_address, _ = Pubkey.find_program_address(
        [
            b"metadata",
            bytes(METAPLEX_TOKEN_METADATA),
            bytes(mint),
        ],
        METAPLEX_TOKEN_METADATA,
    )
    return derived_address


def find_creator_vault(creator: Pubkey) -> Pubkey:
    """Find the creator vault PDA."""
    derived_address, _ = Pubkey.find_program_address(
        [b"creator-vault", bytes(creator)],
        PUMP_PROGRAM,
    )
    return derived_address


def _find_global_volume_accumulator() -> Pubkey:
    derived_address, _ = Pubkey.find_program_address(
        [b"global_volume_accumulator"],
        PUMP_PROGRAM,
    )
    return derived_address


def _find_user_volume_accumulator(user: Pubkey) -> Pubkey:
    derived_address, _ = Pubkey.find_program_address(
        [b"user_volume_accumulator", bytes(user)],
        PUMP_PROGRAM,
    )
    return derived_address


def _find_fee_config() -> Pubkey:
    derived_address, _ = Pubkey.find_program_address(
        [b"fee_config", bytes(PUMP_PROGRAM)],
        PUMP_FEE_PROGRAM,
    )
    return derived_address


def create_pump_create_instruction(
    mint: Pubkey,
    mint_authority: Pubkey,
    bonding_curve: Pubkey,
    associated_bonding_curve: Pubkey,
    global_state: Pubkey,
    metadata: Pubkey,
    user: Pubkey,
    creator: Pubkey,
    name: str,
    symbol: str,
    uri: str,
) -> Instruction:
    """Create the pump.fun create instruction."""
    accounts = [
        AccountMeta(pubkey=mint, is_signer=True, is_writable=True),
        AccountMeta(pubkey=mint_authority, is_signer=False, is_writable=False),
        AccountMeta(pubkey=bonding_curve, is_signer=False, is_writable=True),
        AccountMeta(pubkey=associated_bonding_curve, is_signer=False, is_writable=True),
        AccountMeta(pubkey=global_state, is_signer=False, is_writable=False),
        AccountMeta(pubkey=METAPLEX_TOKEN_METADATA, is_signer=False, is_writable=False),
        AccountMeta(pubkey=metadata, is_signer=False, is_writable=True),
        AccountMeta(pubkey=user, is_signer=True, is_writable=True),
        AccountMeta(pubkey=SYSTEM_PROGRAM, is_signer=False, is_writable=False),
        AccountMeta(pubkey=SYSTEM_TOKEN_PROGRAM, is_signer=False, is_writable=False),
        AccountMeta(
            pubkey=SYSTEM_ASSOCIATED_TOKEN_ACCOUNT_PROGRAM,
            is_signer=False,
            is_writable=False,
        ),
        AccountMeta(pubkey=SYSTEM_RENT, is_signer=False, is_writable=False),
        AccountMeta(pubkey=PUMP_EVENT_AUTHORITY, is_signer=False, is_writable=False),
        AccountMeta(pubkey=PUMP_PROGRAM, is_signer=False, is_writable=False),
    ]

    # Encode string as length-prefixed
    def encode_string(s: str) -> bytes:
        encoded = s.encode("utf-8")
        return struct.pack("<I", len(encoded)) + encoded

    def encode_pubkey(pubkey: Pubkey) -> bytes:
        return bytes(pubkey)

    data = (
        CREATE_DISCRIMINATOR
        + encode_string(name)
        + encode_string(symbol)
        + encode_string(uri)
        + encode_pubkey(creator)
    )

    return Instruction(PUMP_PROGRAM, data, accounts)


def create_extend_account_instruction(
    bonding_curve: Pubkey,
    user: Pubkey,
) -> Instruction:
    """Create the extend_account instruction to expand bonding curve account size."""
    accounts = [
        AccountMeta(pubkey=bonding_curve, is_signer=False, is_writable=True),
        AccountMeta(pubkey=user, is_signer=True, is_writable=True),
        AccountMeta(pubkey=SYSTEM_PROGRAM, is_signer=False, is_writable=False),
        AccountMeta(pubkey=PUMP_EVENT_AUTHORITY, is_signer=False, is_writable=False),
        AccountMeta(pubkey=PUMP_PROGRAM, is_signer=False, is_writable=False),
    ]

    # No arguments for extend_account instruction
    data = EXTEND_ACCOUNT_DISCRIMINATOR

    return Instruction(PUMP_PROGRAM, data, accounts)


def create_buy_instruction(
    global_state: Pubkey,
    fee_recipient: Pubkey,
    mint: Pubkey,
    bonding_curve: Pubkey,
    associated_bonding_curve: Pubkey,
    associated_user: Pubkey,
    user: Pubkey,
    creator_vault: Pubkey,
    token_amount: int,
    max_sol_cost: int,
    track_volume: bool = True,
) -> Instruction:
    """Create the buy instruction."""
    accounts = [
        AccountMeta(pubkey=global_state, is_signer=False, is_writable=False),
        AccountMeta(pubkey=fee_recipient, is_signer=False, is_writable=True),
        AccountMeta(pubkey=mint, is_signer=False, is_writable=False),
        AccountMeta(pubkey=bonding_curve, is_signer=False, is_writable=True),
        AccountMeta(pubkey=associated_bonding_curve, is_signer=False, is_writable=True),
        AccountMeta(pubkey=associated_user, is_signer=False, is_writable=True),
        AccountMeta(pubkey=user, is_signer=True, is_writable=True),
        AccountMeta(pubkey=SYSTEM_PROGRAM, is_signer=False, is_writable=False),
        AccountMeta(pubkey=SYSTEM_TOKEN_PROGRAM, is_signer=False, is_writable=False),
        AccountMeta(pubkey=creator_vault, is_signer=False, is_writable=True),
        AccountMeta(pubkey=PUMP_EVENT_AUTHORITY, is_signer=False, is_writable=False),
        AccountMeta(pubkey=PUMP_PROGRAM, is_signer=False, is_writable=False),
        AccountMeta(
            pubkey=_find_global_volume_accumulator(), is_signer=False, is_writable=True
        ),
        AccountMeta(
            pubkey=_find_user_volume_accumulator(user),
            is_signer=False,
            is_writable=True,
        ),
        # Index 14: fee_config (readonly)
        AccountMeta(
            pubkey=_find_fee_config(),
            is_signer=False,
            is_writable=False,
        ),
        # Index 15: fee_program (readonly)
        AccountMeta(
            pubkey=PUMP_FEE_PROGRAM,
            is_signer=False,
            is_writable=False,
        ),
    ]

    # Encode OptionBool for track_volume
    # OptionBool: [0] = None, [1, 0] = Some(false), [1, 1] = Some(true)
    track_volume_bytes = bytes([1, 1 if track_volume else 0])

    data = (
        BUY_DISCRIMINATOR
        + struct.pack("<Q", token_amount)
        + struct.pack("<Q", max_sol_cost)
        + track_volume_bytes
    )

    return Instruction(PUMP_PROGRAM, data, accounts)


async def main():
    """Create and buy pump.fun token in a single transaction."""
    private_key_bytes = base58.b58decode(PRIVATE_KEY)
    payer = Keypair.from_bytes(private_key_bytes)
    mint_keypair = Keypair()

    print("Creating token with:")
    print(f"  Name: {TOKEN_NAME}")
    print(f"  Symbol: {TOKEN_SYMBOL}")
    print(f"  Mint: {mint_keypair.pubkey()}")
    print(f"  Creator: {payer.pubkey()}")

    # Derive PDAs
    bonding_curve, _ = find_bonding_curve_address(mint_keypair.pubkey())
    associated_bonding_curve = find_associated_bonding_curve(
        mint_keypair.pubkey(), bonding_curve
    )
    metadata = find_metadata_address(mint_keypair.pubkey())
    user_ata = get_associated_token_address(payer.pubkey(), mint_keypair.pubkey())
    creator_vault = find_creator_vault(payer.pubkey())

    print("\nDerived addresses:")
    print(f"  Bonding curve: {bonding_curve}")
    print(f"  Associated bonding curve: {associated_bonding_curve}")
    print(f"  Metadata: {metadata}")
    print(f"  User ATA: {user_ata}")
    print(f"  Creator vault: {creator_vault}")

    # Calculate buy parameters
    # For pump.fun, we need to calculate expected tokens based on initial curve state
    # Initial virtual reserves (from pump.fun constants)
    initial_virtual_token_reserves = 1_073_000_000 * 10**TOKEN_DECIMALS
    initial_virtual_sol_reserves = 30 * LAMPORTS_PER_SOL
    initial_real_token_reserves = 793_100_000 * 10**TOKEN_DECIMALS

    initial_price = initial_virtual_sol_reserves / initial_virtual_token_reserves

    buy_amount_lamports = int(BUY_AMOUNT_SOL * LAMPORTS_PER_SOL)
    expected_tokens = int(
        (buy_amount_lamports * 0.99) / initial_price
    )  # 1% buffer for fees
    max_sol_cost = int(buy_amount_lamports * (1 + MAX_SLIPPAGE))

    print("\nBuy parameters:")
    print(f"  Buy amount: {BUY_AMOUNT_SOL} SOL")
    print(f"  Expected tokens: {expected_tokens / 10**TOKEN_DECIMALS:.6f}")
    print(f"  Max SOL cost: {max_sol_cost / LAMPORTS_PER_SOL:.6f} SOL")

    instructions = [
        # Priority fee instructions
        set_compute_unit_limit(COMPUTE_UNIT_LIMIT),
        set_compute_unit_price(PRIORITY_FEE_MICROLAMPORTS),
        # Create token with pump.fun (this will handle mint account, metadata, etc.)
        create_pump_create_instruction(
            mint=mint_keypair.pubkey(),
            mint_authority=PUMP_MINT_AUTHORITY,
            bonding_curve=bonding_curve,
            associated_bonding_curve=associated_bonding_curve,
            global_state=PUMP_GLOBAL,
            metadata=metadata,
            user=payer.pubkey(),
            creator=payer.pubkey(),
            name=TOKEN_NAME,
            symbol=TOKEN_SYMBOL,
            uri=TOKEN_URI,
        ),
        # Extend bonding curve account (required for frontend visibility)
        create_extend_account_instruction(
            bonding_curve=bonding_curve,
            user=payer.pubkey(),
        ),
        # Create user ATA
        create_idempotent_associated_token_account(
            payer.pubkey(),
            payer.pubkey(),
            mint_keypair.pubkey(),
            SYSTEM_TOKEN_PROGRAM,
        ),
        # Buy tokens
        create_buy_instruction(
            global_state=PUMP_GLOBAL,
            fee_recipient=PUMP_FEE,
            mint=mint_keypair.pubkey(),
            bonding_curve=bonding_curve,
            associated_bonding_curve=associated_bonding_curve,
            associated_user=user_ata,
            user=payer.pubkey(),
            creator_vault=creator_vault,
            token_amount=expected_tokens,
            max_sol_cost=max_sol_cost,
            track_volume=True,
        ),
    ]

    # Send transaction
    async with AsyncClient(RPC_ENDPOINT) as client:
        recent_blockhash = await client.get_latest_blockhash()
        message = Message(instructions, payer.pubkey())
        transaction = Transaction(
            [payer, mint_keypair], message, recent_blockhash.value.blockhash
        )

        print("\nSending transaction...")
        opts = TxOpts(skip_preflight=True, preflight_commitment=Confirmed)

        try:
            response = await client.send_transaction(transaction, opts)
            tx_hash = response.value

            print(f"Transaction sent: https://solscan.io/tx/{tx_hash}")

            print("Waiting for confirmation...")
            await client.confirm_transaction(tx_hash, commitment="confirmed")
            print("Transaction confirmed!")

            return tx_hash

        except Exception as e:
            print(f"Transaction failed: {e}")
            raise


if __name__ == "__main__":
    asyncio.run(main())
