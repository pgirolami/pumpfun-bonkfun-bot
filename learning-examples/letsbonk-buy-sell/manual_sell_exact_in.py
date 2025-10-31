"""
Manual Sell Exact In Example for Raydium LaunchLab

This script demonstrates how to sell tokens using the sell_exact_in instruction
from the Raydium LaunchLab program. It follows the IDL structure.

Key features:
- Uses sell_exact_in instruction
- Implements proper account ordering as per IDL
- Includes slippage protection with minimum_amount_out
- Handles WSOL wrapping/unwrapping automatically
- Follows the exact transaction structure from the buy_exact_in example
- User configurable token amount and slippage
- Uses idempotent ATA creation
"""

import asyncio
import os
import struct
import sys

import base58
from dotenv import load_dotenv
from idl_parser import load_idl_parser
from solana.rpc.async_api import AsyncClient
from solana.rpc.commitment import Confirmed
from solana.rpc.types import TxOpts
from solders.compute_budget import set_compute_unit_limit, set_compute_unit_price
from solders.instruction import AccountMeta, Instruction
from solders.keypair import Keypair
from solders.message import Message
from solders.pubkey import Pubkey
from solders.system_program import CreateAccountWithSeedParams, create_account_with_seed
from solders.transaction import VersionedTransaction

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Initialize IDL parser for Raydium LaunchLab with verbose mode for debugging
IDL_PARSER = load_idl_parser("idl/raydium_launchlab_idl.json", verbose=True)

load_dotenv()

TOKEN_MINT_ADDRESS = Pubkey.from_string(
    "YOUR_TOKEN_MINT_ADDRESS_HERE"
)  # Replace with actual token mint address

# Configuration constants
RPC_ENDPOINT = os.environ.get("SOLANA_NODE_RPC_ENDPOINT")
PRIVATE_KEY = base58.b58decode(os.environ.get("SOLANA_PRIVATE_KEY"))
PAYER = Keypair.from_bytes(PRIVATE_KEY)

# User configurable parameters
TOKEN_AMOUNT_TO_SELL = int(
    os.environ.get("TOKEN_AMOUNT", "1000000")
)  # Amount of tokens to sell (in base units)
SLIPPAGE_TOLERANCE = float(os.environ.get("SLIPPAGE", "0.25"))

# Transaction parameters
SHARE_FEE_RATE = 0

# Program IDs and addresses from Raydium LaunchLab
RAYDIUM_LAUNCHLAB_PROGRAM_ID = Pubkey.from_string(
    "LanMV9sAd7wArD4vJFi2qDdfnVhFxYSUg6eADduJ3uj"
)
GLOBAL_CONFIG = Pubkey.from_string("6s1xP3hpbAfFoNtUNF8mfHsjr2Bd97JxFJRWLbL6aHuX")
LETSBONK_PLATFORM_CONFIG = Pubkey.from_string(
    "5thqcDwKp5QQ8US4XRMoseGeGbmLKMmoKZmS6zHrQAsA"
)

# Token program and system addresses
TOKEN_PROGRAM_ID = Pubkey.from_string("TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA")
SYSTEM_PROGRAM_ID = Pubkey.from_string("11111111111111111111111111111111")
WSOL_MINT = Pubkey.from_string("So11111111111111111111111111111111111111112")
COMPUTE_BUDGET_PROGRAM_ID = Pubkey.from_string(
    "ComputeBudget111111111111111111111111111111"
)
ASSOCIATED_TOKEN_PROGRAM_ID = Pubkey.from_string(
    "ATokenGPvbdGVxr1b2hvZbsiqW5xWH25efTNsLJA8knL"
)
SYSTEM_RENT_PROGRAM_ID = Pubkey.from_string(
    "SysvarRent111111111111111111111111111111111"
)

# Instruction discriminator for sell_exact_in (from IDL)
SELL_EXACT_IN_DISCRIMINATOR = bytes([149, 39, 222, 155, 211, 124, 152, 26])

# Compute budget settings
COMPUTE_UNIT_LIMIT = 150_000
COMPUTE_UNIT_PRICE = 1_000

LAMPORTS_PER_SOL = 1_000_000_000


def derive_authority_pda() -> Pubkey:
    """
    Derive the authority PDA for the Raydium LaunchLab program.

    This PDA acts as the authority for pool vault operations and is generated
    using the AUTH_SEED as specified in the IDL.

    Returns:
        Pubkey: The derived authority PDA
    """
    AUTH_SEED = b"vault_auth_seed"
    authority_pda, _ = Pubkey.find_program_address(
        [AUTH_SEED], RAYDIUM_LAUNCHLAB_PROGRAM_ID
    )
    return authority_pda


def derive_event_authority_pda() -> Pubkey:
    """
    Derive the event authority PDA for the Raydium LaunchLab program.

    This PDA is used for emitting program events during swaps.

    Returns:
        Pubkey: The derived event authority PDA
    """
    EVENT_AUTHORITY_SEED = b"__event_authority"
    event_authority_pda, _ = Pubkey.find_program_address(
        [EVENT_AUTHORITY_SEED], RAYDIUM_LAUNCHLAB_PROGRAM_ID
    )
    return event_authority_pda


def derive_pool_state_for_token(base_token_mint: Pubkey) -> Pubkey | None:
    """
    Derive the pool state account for a given base token mint.

    Args:
        base_token_mint: The token mint address to search for

    Returns:
        Pubkey of the pool state account, or None if not found
    """
    seeds = [b"pool", bytes(base_token_mint), bytes(WSOL_MINT)]
    pool_state_pda, _ = Pubkey.find_program_address(seeds, RAYDIUM_LAUNCHLAB_PROGRAM_ID)
    return pool_state_pda


def derive_creator_fee_vault(creator: Pubkey, quote_mint: Pubkey) -> Pubkey:
    """
    Derive the creator fee vault PDA.

    This vault accumulates creator fees from trades.

    Args:
        creator: The pool creator's pubkey
        quote_mint: The quote token mint (WSOL)

    Returns:
        Pubkey of the creator fee vault
    """
    seeds = [bytes(creator), bytes(quote_mint)]
    creator_fee_vault_pda, _ = Pubkey.find_program_address(
        seeds, RAYDIUM_LAUNCHLAB_PROGRAM_ID
    )
    return creator_fee_vault_pda


def derive_platform_fee_vault(platform_config: Pubkey, quote_mint: Pubkey) -> Pubkey:
    """
    Derive the platform fee vault PDA.

    This vault accumulates platform fees from trades.

    Args:
        platform_config: The platform config account
        quote_mint: The quote token mint (WSOL)

    Returns:
        Pubkey of the platform fee vault
    """
    seeds = [bytes(platform_config), bytes(quote_mint)]
    platform_fee_vault_pda, _ = Pubkey.find_program_address(
        seeds, RAYDIUM_LAUNCHLAB_PROGRAM_ID
    )
    return platform_fee_vault_pda


def decode_pool_state(account_data: bytes) -> dict | None:
    """
    Decode pool state account data using the IDL parser.

    Args:
        account_data: Raw account data from the pool state account

    Returns:
        Dictionary containing decoded pool state data, or None if decoding fails
    """
    try:
        result = IDL_PARSER.decode_account_data(
            account_data, "PoolState", skip_discriminator=True
        )
        if result:
            return result

        return None

    except Exception as e:
        print(f"Error decoding pool state: {e}")
        import traceback

        traceback.print_exc()
        return None


async def get_pool_state_data(client: AsyncClient, pool_state: Pubkey) -> dict | None:
    """
    Get and decode the pool state account data.

    Args:
        client: Solana RPC client
        pool_state: The pool state account address

    Returns:
        Dictionary containing decoded pool state data, or None if error
    """
    try:
        account_info = await client.get_account_info(pool_state)
        if not account_info.value:
            print("Pool state account not found")
            return None

        return decode_pool_state(account_info.value.data)

    except Exception as e:
        print(f"Error getting pool state data: {e}")
        return None


def get_associated_token_address(owner: Pubkey, mint: Pubkey) -> Pubkey:
    """
    Calculate the associated token account address for a given owner and mint.

    This manually implements the ATA derivation without requiring the spl-token package.

    Args:
        owner: The wallet that owns the token account
        mint: The token mint address

    Returns:
        Pubkey of the associated token account
    """
    ata_address, _ = Pubkey.find_program_address(
        [bytes(owner), bytes(TOKEN_PROGRAM_ID), bytes(mint)],
        ASSOCIATED_TOKEN_PROGRAM_ID,
    )
    return ata_address


def create_associated_token_account_idempotent_instruction(
    payer: Pubkey, owner: Pubkey, mint: Pubkey
) -> Instruction:
    """
    Create an idempotent instruction to create an Associated Token Account.

    This uses the CreateIdempotent instruction which doesn't fail if the ATA already exists.

    Args:
        payer: The account that will pay for the creation
        owner: The owner of the new token account
        mint: The token mint

    Returns:
        Instruction for creating the ATA idempotently
    """
    ata_address = get_associated_token_address(owner, mint)

    accounts = [
        AccountMeta(pubkey=payer, is_signer=True, is_writable=True),  # Funding account
        AccountMeta(
            pubkey=ata_address, is_signer=False, is_writable=True
        ),  # Associated token account
        AccountMeta(pubkey=owner, is_signer=False, is_writable=False),  # Wallet address
        AccountMeta(pubkey=mint, is_signer=False, is_writable=False),  # Token mint
        AccountMeta(
            pubkey=SYSTEM_PROGRAM_ID, is_signer=False, is_writable=False
        ),  # System program
        AccountMeta(
            pubkey=TOKEN_PROGRAM_ID, is_signer=False, is_writable=False
        ),  # Token program
    ]

    data = bytes([1])

    return Instruction(
        program_id=ASSOCIATED_TOKEN_PROGRAM_ID, data=data, accounts=accounts
    )


def create_initialize_account_instruction(
    account: Pubkey, mint: Pubkey, owner: Pubkey
) -> Instruction:
    """
    Create an InitializeAccount instruction for the Token Program.

    Args:
        account: The account to initialize
        mint: The token mint
        owner: The account owner

    Returns:
        Instruction for initializing the account
    """
    accounts = [
        AccountMeta(pubkey=account, is_signer=False, is_writable=True),
        AccountMeta(pubkey=mint, is_signer=False, is_writable=False),
        AccountMeta(pubkey=owner, is_signer=False, is_writable=False),
        AccountMeta(pubkey=SYSTEM_RENT_PROGRAM_ID, is_signer=False, is_writable=False),
    ]

    # InitializeAccount instruction discriminator (instruction 1 in Token Program)
    data = bytes([1])

    return Instruction(program_id=TOKEN_PROGRAM_ID, data=data, accounts=accounts)


def create_close_account_instruction(
    account: Pubkey, destination: Pubkey, owner: Pubkey
) -> Instruction:
    """
    Create a CloseAccount instruction for the Token Program.

    Args:
        account: The account to close
        destination: Where to send the remaining lamports
        owner: The account owner (must sign)

    Returns:
        Instruction for closing the account
    """
    accounts = [
        AccountMeta(pubkey=account, is_signer=False, is_writable=True),
        AccountMeta(pubkey=destination, is_signer=False, is_writable=True),
        AccountMeta(pubkey=owner, is_signer=True, is_writable=False),
    ]

    data = bytes([9])

    return Instruction(program_id=TOKEN_PROGRAM_ID, data=data, accounts=accounts)


def create_wsol_account_with_seed(
    payer: Pubkey, seed: str, lamports: int
) -> tuple[Pubkey, Instruction, Instruction]:
    """
    Create a WSOL account using createAccountWithSeed and initialize it.

    This replicates the exact pattern from the Solscan example where a new account
    is created with a seed and then initialized as a token account.

    Args:
        payer: The account that will pay for and own the new account
        seed: String seed for deterministic account generation
        lamports: Amount of lamports to transfer to the new account

    Returns:
        Tuple of (new_account_pubkey, create_instruction, initialize_instruction)
    """
    new_account = Pubkey.create_with_seed(payer, seed, TOKEN_PROGRAM_ID)

    create_ix = create_account_with_seed(
        CreateAccountWithSeedParams(
            from_pubkey=payer,
            to_pubkey=new_account,
            base=payer,
            seed=seed,
            lamports=lamports,
            space=165,  # Size of a token account
            owner=TOKEN_PROGRAM_ID,
        )
    )

    initialize_ix = create_initialize_account_instruction(new_account, WSOL_MINT, payer)

    return new_account, create_ix, initialize_ix


def get_user_base_token_account(payer: Pubkey, base_mint: Pubkey) -> Pubkey:
    """
    Get the user's associated token account for the base token.

    In a real implementation, this should check if the account exists and create it if needed.
    For this example, we'll derive the standard ATA address.

    Args:
        payer: The user's wallet address
        base_mint: The base token mint address

    Returns:
        Pubkey of the user's base token account
    """
    return get_associated_token_address(payer, base_mint)


def calculate_minimum_amount_out_from_pool_state(
    pool_state_data: dict, amount_in: int, slippage_tolerance: float
) -> int:
    """
    Calculate the minimum amount out based on pool state data and slippage tolerance.

    Uses the actual pool reserves to calculate expected output using constant product formula.
    This is for selling base tokens to get quote tokens (WSOL).

    Args:
        pool_state_data: Decoded pool state data containing reserves
        amount_in: Amount of base tokens being sold
        slippage_tolerance: Slippage tolerance as a decimal (0.25 = 25%)

    Returns:
        Minimum amount of quote tokens (WSOL) to receive
    """
    try:
        # Extract pool reserves from decoded state
        virtual_base = pool_state_data["virtual_base"]
        virtual_quote = pool_state_data["virtual_quote"]
        real_base = pool_state_data["real_base"]
        real_quote = pool_state_data["real_quote"]

        print("Pool State:")
        print(f"  Virtual Base: {virtual_base:,}")
        print(f"  Virtual Quote: {virtual_quote:,}")
        print(f"  Real Base: {real_base:,}")
        print(f"  Real Quote: {real_quote:,}")

        # Use virtual reserves for bonding curve calculation
        # For selling base tokens: amount_out = (amount_in * virtual_quote) / (virtual_base + amount_in)

        # Calculate expected output using constant product formula
        # Note: The program deducts fees before calculating output, so we need to account for that
        # Trade fee is typically around 0.25% - 1%
        # For safety, we'll calculate without fee adjustment and let slippage handle it
        numerator = amount_in * virtual_quote
        denominator = virtual_base + amount_in
        expected_output = numerator // denominator

        # Apply slippage tolerance (be more conservative for small amounts)
        minimum_with_slippage = int(expected_output * (1 - slippage_tolerance))

        print(f"Amount in: {amount_in:,} tokens")
        print(
            f"Expected output: {expected_output:,} lamports ({expected_output / LAMPORTS_PER_SOL:.6f} SOL)"
        )
        print(
            f"Minimum with {slippage_tolerance * 100}% slippage: {minimum_with_slippage:,} lamports ({minimum_with_slippage / LAMPORTS_PER_SOL:.6f} SOL)"
        )

        return minimum_with_slippage

    except Exception as e:
        print(f"Error calculating minimum amount out from pool state: {e}")
        return None


async def sell_exact_in(
    client: AsyncClient,
    base_token_mint: Pubkey,
    amount_in_tokens: int,
    slippage_tolerance: float,
) -> str | None:
    """
    Execute a sell_exact_in transaction on Raydium LaunchLab.

    This function implements the exact transaction flow similar to buy_exact_in:
    1. SetComputeUnitPrice
    2. SetComputeUnitLimit
    3. Create WSOL account with seed
    4. Initialize WSOL account
    5. Execute sell_exact_in instruction
    6. Close WSOL account
    7. Optional: Transfer remaining SOL (as seen in the example)

    Args:
        client: Solana RPC client
        base_token_mint: Address of the token to sell
        amount_in_tokens: Amount of tokens to sell
        slippage_tolerance: Slippage tolerance as decimal

    Returns:
        Transaction signature if successful, None otherwise
    """
    try:
        print(f"Finding pool state for token: {base_token_mint}")
        pool_state = derive_pool_state_for_token(base_token_mint)
        if not pool_state:
            print("Pool state not found for this token")
            return None

        # Get and decode pool state data using IDL parser
        pool_state_data = await get_pool_state_data(client, pool_state)
        if not pool_state_data:
            print("Failed to decode pool state data")
            return None

        # Extract vault addresses and creator from decoded pool state (convert from base58 strings to Pubkey objects)
        base_vault = Pubkey.from_string(pool_state_data["base_vault"])
        quote_vault = Pubkey.from_string(pool_state_data["quote_vault"])
        creator = Pubkey.from_string(pool_state_data["creator"])

        print(f"Found pool state: {pool_state}")
        print(f"Base vault: {base_vault}")
        print(f"Quote vault: {quote_vault}")
        print(f"Creator: {creator}")
        print(f"Pool status: {pool_state_data['status']}")

        # Derive necessary PDAs
        authority = derive_authority_pda()
        event_authority = derive_event_authority_pda()
        creator_fee_vault = derive_creator_fee_vault(creator, WSOL_MINT)
        platform_fee_vault = derive_platform_fee_vault(
            LETSBONK_PLATFORM_CONFIG, WSOL_MINT
        )

        print(f"Creator fee vault: {creator_fee_vault}")
        print(f"Platform fee vault: {platform_fee_vault}")

        # Calculate amounts using pool state data
        minimum_amount_out = calculate_minimum_amount_out_from_pool_state(
            pool_state_data, amount_in_tokens, slippage_tolerance
        )

        if minimum_amount_out is None or minimum_amount_out == 0:
            print("Failed to calculate minimum amount out or amount is too small")
            return None

        print(f"Amount in: {amount_in_tokens:,} tokens")
        print(
            f"Minimum amount out: {minimum_amount_out:,} lamports ({minimum_amount_out / LAMPORTS_PER_SOL:.6f} SOL)"
        )

        # Get user's base token account (where tokens will be debited from)
        user_base_token = get_associated_token_address(PAYER.pubkey(), base_token_mint)

        # Step 1: Create WSOL account with seed (where WSOL will be received)
        import hashlib
        import time

        # Generate a unique seed based on timestamp and user pubkey
        seed_data = f"{int(time.time())}{PAYER.pubkey()!s}"
        wsol_seed = hashlib.sha256(seed_data.encode()).hexdigest()[:32]

        # Calculate required lamports (minimal amount for account creation)
        account_creation_lamports = 2_039_280  # Standard account creation cost

        user_quote_token, create_wsol_ix, init_wsol_ix = create_wsol_account_with_seed(
            PAYER.pubkey(), wsol_seed, account_creation_lamports
        )

        print(f"User base token account: {user_base_token}")
        print(f"User quote token account: {user_quote_token}")

        # Step 2: Build the sell_exact_in instruction
        accounts = [
            AccountMeta(
                pubkey=PAYER.pubkey(), is_signer=True, is_writable=False
            ),  # payer
            AccountMeta(
                pubkey=authority, is_signer=False, is_writable=False
            ),  # authority
            AccountMeta(
                pubkey=GLOBAL_CONFIG, is_signer=False, is_writable=False
            ),  # global_config
            AccountMeta(
                pubkey=LETSBONK_PLATFORM_CONFIG, is_signer=False, is_writable=False
            ),  # platform_config
            AccountMeta(
                pubkey=pool_state, is_signer=False, is_writable=True
            ),  # pool_state
            AccountMeta(
                pubkey=user_base_token, is_signer=False, is_writable=True
            ),  # user_base_token (tokens being sold)
            AccountMeta(
                pubkey=user_quote_token, is_signer=False, is_writable=True
            ),  # user_quote_token (WSOL received)
            AccountMeta(
                pubkey=base_vault, is_signer=False, is_writable=True
            ),  # base_vault (receives tokens)
            AccountMeta(
                pubkey=quote_vault, is_signer=False, is_writable=True
            ),  # quote_vault (sends WSOL)
            AccountMeta(
                pubkey=base_token_mint, is_signer=False, is_writable=False
            ),  # base_token_mint
            AccountMeta(
                pubkey=WSOL_MINT, is_signer=False, is_writable=False
            ),  # quote_token_mint
            AccountMeta(
                pubkey=TOKEN_PROGRAM_ID, is_signer=False, is_writable=False
            ),  # base_token_program
            AccountMeta(
                pubkey=TOKEN_PROGRAM_ID, is_signer=False, is_writable=False
            ),  # quote_token_program
            AccountMeta(
                pubkey=event_authority, is_signer=False, is_writable=False
            ),  # event_authority
            AccountMeta(
                pubkey=RAYDIUM_LAUNCHLAB_PROGRAM_ID, is_signer=False, is_writable=False
            ),  # program
        ]

        # Add remaining accounts (not explicitly listed in IDL but required by the program)
        # These accounts are used for fee collection during swaps
        accounts.append(
            AccountMeta(pubkey=SYSTEM_PROGRAM_ID, is_signer=False, is_writable=False)
        )  # #16: System Program
        accounts.append(
            AccountMeta(pubkey=platform_fee_vault, is_signer=False, is_writable=True)
        )  # #17: Platform fee vault
        accounts.append(
            AccountMeta(pubkey=creator_fee_vault, is_signer=False, is_writable=True)
        )  # #18: Creator fee vault

        # Instruction data: discriminator + amount_in + minimum_amount_out + share_fee_rate
        instruction_data = (
            SELL_EXACT_IN_DISCRIMINATOR
            + struct.pack("<Q", amount_in_tokens)  # amount_in (u64)
            + struct.pack("<Q", minimum_amount_out)  # minimum_amount_out (u64)
            + struct.pack("<Q", SHARE_FEE_RATE)  # share_fee_rate (u64): 0
        )

        sell_exact_in_ix = Instruction(
            program_id=RAYDIUM_LAUNCHLAB_PROGRAM_ID,
            data=instruction_data,
            accounts=accounts,
        )

        # Step 3: Create close WSOL account instruction
        close_wsol_ix = create_close_account_instruction(
            user_quote_token, PAYER.pubkey(), PAYER.pubkey()
        )

        # Step 4: Build complete transaction
        instructions = [
            set_compute_unit_price(COMPUTE_UNIT_PRICE),
            set_compute_unit_limit(COMPUTE_UNIT_LIMIT),
            # Instruction #3: Create WSOL account with seed
            create_wsol_ix,
            # Instruction #4: Initialize WSOL account
            init_wsol_ix,
            # Instruction #5: Execute sell_exact_in
            sell_exact_in_ix,
            # Instruction #6: Close WSOL account
            close_wsol_ix,
        ]

        blockhash_resp = await client.get_latest_blockhash()
        recent_blockhash = blockhash_resp.value.blockhash

        message = Message.new_with_blockhash(
            instructions, PAYER.pubkey(), recent_blockhash
        )

        transaction = VersionedTransaction(message, [PAYER])

        print("Simulating transaction...")
        simulation = await client.simulate_transaction(transaction)

        if simulation.value.err:
            print(f"Simulation failed: {simulation.value.err}")
            return None

        print(
            f"Simulation successful. Compute units consumed: {simulation.value.units_consumed}"
        )

        print("Sending transaction...")
        result = await client.send_transaction(
            transaction,
            opts=TxOpts(skip_preflight=True, preflight_commitment=Confirmed),
        )

        tx_signature = result.value
        print(f"Transaction sent: https://solscan.io/tx/{tx_signature}")

        print("Waiting for confirmation...")
        await client.confirm_transaction(tx_signature, commitment="confirmed")
        print("Transaction confirmed!")

        return tx_signature

    except Exception as e:
        print(f"Error executing sell_exact_in: {e}")
        import traceback

        traceback.print_exc()
        return None


async def main():
    """
    Main function to execute the sell_exact_in example.

    Takes configuration from environment variables or uses defaults.
    """
    try:
        print(f"Starting sell_exact_in for token: {TOKEN_MINT_ADDRESS}")
        print(f"Amount to sell: {TOKEN_AMOUNT_TO_SELL:,} tokens")
        print(f"Slippage tolerance: {SLIPPAGE_TOLERANCE * 100}%")
        print(f"Using RPC endpoint: {RPC_ENDPOINT}")
        print()

        async with AsyncClient(RPC_ENDPOINT) as client:
            balance_resp = await client.get_balance(PAYER.pubkey())
            balance_sol = balance_resp.value / LAMPORTS_PER_SOL
            print(f"Wallet balance: {balance_sol:.6f} SOL")

            # Check if user has the base token account and sufficient balance
            user_base_token = get_associated_token_address(
                PAYER.pubkey(), TOKEN_MINT_ADDRESS
            )
            try:
                token_account_info = await client.get_token_account_balance(
                    user_base_token
                )
                if token_account_info.value:
                    token_balance = int(token_account_info.value.amount)
                    print(f"Token balance: {token_balance:,} tokens")

                    if token_balance < TOKEN_AMOUNT_TO_SELL:
                        print(
                            f"Insufficient token balance! You have {token_balance:,} tokens but want to sell {TOKEN_AMOUNT_TO_SELL:,}"
                        )
                        return
                else:
                    print("Token account not found or has no balance!")
                    return
            except Exception as e:
                print(f"Error checking token balance: {e}")
                print("Continuing anyway...")

            tx_signature = await sell_exact_in(
                client, TOKEN_MINT_ADDRESS, TOKEN_AMOUNT_TO_SELL, SLIPPAGE_TOLERANCE
            )

            if tx_signature:
                print(f"\n✅ Success! Transaction: {tx_signature}")
                print(f"🔗 View on Solscan: https://solscan.io/tx/{tx_signature}")
            else:
                print("\n❌ Transaction failed!")

    except ValueError as e:
        print(f"Invalid token mint address: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
