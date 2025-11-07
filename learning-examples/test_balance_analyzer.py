"""
Learning example to test PumpFunBalanceAnalyzer on a specific transaction.

This script demonstrates how to:
1. Fetch a transaction from Solana
2. Extract instruction accounts from the transaction
3. Create a TokenInfo object
4. Call the balance analyzer to get balance changes
"""

import asyncio
import base64
import os
import struct
import sys
from pathlib import Path

import base58
from construct import Bytes, Flag, Int64ul, Struct
from solders.pubkey import Pubkey

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.client import SolanaClient
from interfaces.core import Platform, TokenInfo
from platforms.pumpfun.address_provider import PumpFunAddressProvider
from platforms.pumpfun.balance_analyzer import PumpFunBalanceAnalyzer
from utils.idl_parser import IDLParser

# Bonding curve state parser
EXPECTED_DISCRIMINATOR = struct.pack("<Q", 6966180631402821399)


class BondingCurveState:
    """Parser for bonding curve account data."""

    _STRUCT_1 = Struct(
        "virtual_token_reserves" / Int64ul,
        "virtual_sol_reserves" / Int64ul,
        "real_token_reserves" / Int64ul,
        "real_sol_reserves" / Int64ul,
        "token_total_supply" / Int64ul,
        "complete" / Flag,
    )

    # Struct after creator fee update has been introduced
    _STRUCT_2 = Struct(
        "virtual_token_reserves" / Int64ul,
        "virtual_sol_reserves" / Int64ul,
        "real_token_reserves" / Int64ul,
        "real_sol_reserves" / Int64ul,
        "token_total_supply" / Int64ul,
        "complete" / Flag,
        "creator" / Bytes(32),  # Added new creator field - 32 bytes for Pubkey
    )

    def __init__(self, data: bytes) -> None:
        """Parse bonding curve data."""
        if data[:8] != EXPECTED_DISCRIMINATOR:
            raise ValueError("Invalid curve state discriminator")

        if len(data) < 150:
            parsed = self._STRUCT_1.parse(data[8:])
            self.__dict__.update(parsed)
        else:
            parsed = self._STRUCT_2.parse(data[8:])
            self.__dict__.update(parsed)
            # Convert raw bytes to Pubkey for creator field
            if hasattr(self, "creator") and isinstance(self.creator, bytes):
                self.creator = Pubkey.from_bytes(self.creator)


async def extract_instruction_accounts(
    tx, idl_parser: IDLParser, program_id: Pubkey
) -> dict[str, Pubkey] | None:
    """Extract instruction accounts from a transaction.

    Args:
        tx: Transaction object
        idl_parser: IDL parser instance
        program_id: Program ID to match

    Returns:
        Dictionary of instruction accounts or None if not found
    """
    if not tx or not tx.transaction:
        return None

    account_keys = tx.transaction.transaction.message.account_keys
    instructions = tx.transaction.transaction.message.instructions

    for ix in instructions:
        # Handle both regular Instruction and UiPartiallyDecodedInstruction
        instruction_program_id = None
        
        # Check if this is a UiPartiallyDecodedInstruction (from jsonParsed encoding)
        if hasattr(ix, "program_id"):
            # UiPartiallyDecodedInstruction has program_id directly
            instruction_program_id = ix.program_id
            if isinstance(instruction_program_id, str):
                instruction_program_id = Pubkey.from_string(instruction_program_id)
            elif hasattr(instruction_program_id, "pubkey"):
                instruction_program_id = instruction_program_id.pubkey
        elif hasattr(ix, "program_id_index"):
            # Regular Instruction has program_id_index
            program_id_index = ix.program_id_index
            if program_id_index >= len(account_keys):
                continue
            instruction_program_id = account_keys[program_id_index].pubkey
        else:
            continue

        # Check if this instruction is from the pump.fun program
        if instruction_program_id != program_id:
            continue

        # Get instruction data and accounts
        ix_data = None
        account_indices = None

        # Handle UiPartiallyDecodedInstruction
        if hasattr(ix, "program_id"):
            # For UiPartiallyDecodedInstruction, we need to get data and accounts differently
            if hasattr(ix, "data") and ix.data:
                # Data might be base58 encoded string
                if isinstance(ix.data, str):
                    try:
                        ix_data = base58.b58decode(ix.data)
                    except Exception:
                        continue
                else:
                    ix_data = bytes(ix.data)
            
            if hasattr(ix, "accounts") and ix.accounts:
                # Accounts are already Pubkey objects or strings in UiPartiallyDecodedInstruction
                # We need to find their indices in account_keys
                account_indices = []
                for acc in ix.accounts:
                    acc_pubkey = acc
                    if isinstance(acc, str):
                        acc_pubkey = Pubkey.from_string(acc)
                    elif hasattr(acc, "pubkey"):
                        acc_pubkey = acc.pubkey
                    
                    # Find index in account_keys
                    for i, key in enumerate(account_keys):
                        if key.pubkey == acc_pubkey:
                            account_indices.append(i)
                            break
        else:
            # Regular Instruction
            if hasattr(ix, "data"):
                ix_data = bytes(ix.data)
            if hasattr(ix, "accounts"):
                account_indices = list(ix.accounts)

        if not ix_data or not account_indices:
            continue

        # Convert account keys to bytes for IDL parser
        account_keys_bytes = [bytes(key.pubkey) for key in account_keys]

        decoded = idl_parser.decode_instruction(
            ix_data, account_keys_bytes, account_indices
        )

        if decoded and decoded.get("instruction_name") in ["buy", "sell"]:
            # Extract accounts from decoded instruction
            accounts_dict = decoded.get("accounts", {})
            instruction_accounts = {}

            # Convert string addresses to Pubkey objects
            for key, value in accounts_dict.items():
                if value:
                    try:
                        instruction_accounts[key] = Pubkey.from_string(value)
                    except Exception as e:
                        print(f"Warning: Could not parse account {key}={value}: {e}")

            # Store instruction type for later use
            instruction_accounts["_instruction_type"] = decoded.get("instruction_name")
            return instruction_accounts

    return None


async def main():
    """Main function to test balance analyzer."""
    # Configuration

    #buy
    # token_mint = "CxtvoNDCGNKm3XdfTTqSEd2XfDFiRroy93EvjrxKpump"
    # tx_signature = "4egaySCTrsepC83PwS9Z182baKSZSBg3vXtyEXKiMjzAvLKxcv2fMNdJnK13BPJK6SdMesYb8Ktqcapj4WSsXBQt"

    #sell
    token_mint = "CxtvoNDCGNKm3XdfTTqSEd2XfDFiRroy93EvjrxKpump"
    tx_signature = "4kjAn1m8AAgGu4PzMmU84uCbjThAaydkCPubCvc7SNo6RpWogmRKTBxw16BkrNNdjRdnLu9LCwJzxLF5WuFZXcvq"


    # Get RPC endpoint from environment
    rpc_endpoint = os.getenv("SOLANA_RPC_HTTP")
    if not rpc_endpoint:
        print("Error: SOLANA_RPC_HTTP environment variable not set")
        sys.exit(1)

    # Initialize components
    rpc_config = {
        "rpc_endpoint": rpc_endpoint,
        "wss_endpoint": os.getenv("SOLANA_RPC_WEBSOCKET", ""),
        "send_method": "solana",
    }

    client = SolanaClient(rpc_config)
    balance_analyzer = PumpFunBalanceAnalyzer()
    address_provider = PumpFunAddressProvider()

    # Load IDL parser
    idl_path = Path(__file__).parent.parent / "idl" / "pump_fun_idl.json"
    if not idl_path.exists():
        print(f"Error: IDL file not found at {idl_path}")
        sys.exit(1)

    idl_parser = IDLParser(str(idl_path), verbose=False)

    try:
        print(f"Fetching transaction: {tx_signature}")
        print(f"Token mint: {token_mint}")
        print()

        # Fetch transaction (client will convert string to Signature object)
        tx = await client.get_transaction(tx_signature)
        if not tx:
            print("Error: Transaction not found")
            sys.exit(1)

        # Get program ID
        program_id = address_provider.program_id

        # Get wallet pubkey (fee payer is first account)
        if not tx.transaction or not tx.transaction.transaction:
            print("Error: Invalid transaction structure")
            sys.exit(1)

        wallet_pubkey = tx.transaction.transaction.message.account_keys[0].pubkey
        print(f"Wallet (fee payer): {wallet_pubkey}")
        print()

        # Extract instruction accounts from transaction
        print("Extracting instruction accounts from transaction...")
        extracted_accounts = await extract_instruction_accounts(
            tx, idl_parser, program_id
        )

        # Create TokenInfo
        mint_pubkey = Pubkey.from_string(token_mint)

        # Get user from extracted accounts or use wallet as fallback
        user = extracted_accounts.get("user") if extracted_accounts else wallet_pubkey

        # Derive bonding curve from mint
        bonding_curve = address_provider.derive_pool_address(mint_pubkey)
        associated_bonding_curve = address_provider.derive_associated_bonding_curve(
            mint_pubkey, bonding_curve
        )

        # Override with extracted accounts if available
        if extracted_accounts:
            if "bonding_curve" in extracted_accounts:
                bonding_curve = extracted_accounts["bonding_curve"]
            if "associated_bonding_curve" in extracted_accounts:
                associated_bonding_curve = extracted_accounts["associated_bonding_curve"]

        # Get creator from bonding curve account data and derive creator vault
        print("Fetching bonding curve account to get creator...")
        creator = None
        creator_vault = None
        
        try:
            # Fetch bonding curve account data
            curve_account = await client.get_account_info(bonding_curve)
            if curve_account and curve_account.data:
                # Parse account data
                if isinstance(curve_account.data, list):
                    # Data is base64 encoded
                    curve_data = base64.b64decode(curve_account.data[0])
                else:
                    curve_data = curve_account.data
                
                # Parse bonding curve state
                curve_state = BondingCurveState(curve_data)
                
                # Get creator from curve state
                if hasattr(curve_state, "creator") and curve_state.creator:
                    creator = curve_state.creator
                    creator_vault = address_provider.derive_creator_vault(creator)
                    print(f"  Creator from bonding curve: {creator}")
                    print(f"  Creator vault: {creator_vault}")
                else:
                    print("  Warning: Bonding curve does not have creator field (old format)")
        except Exception as e:
            print(f"  Warning: Could not fetch bonding curve account: {e}")
        
        # Fallback: try to get creator from extracted accounts
        if not creator:
            if extracted_accounts and "creator" in extracted_accounts:
                creator = extracted_accounts["creator"]
            elif extracted_accounts and "user" in extracted_accounts:
                # For pump.fun, creator might be the same as user in some cases
                creator = extracted_accounts["user"]
            
            if creator and not creator_vault:
                creator_vault = address_provider.derive_creator_vault(creator)
                print(f"  Using creator from extracted accounts: {creator}")
        
        # Final fallback: use extracted creator_vault if available
        if not creator_vault and extracted_accounts:
            creator_vault = extracted_accounts.get("creator_vault")

        token_info = TokenInfo(
            name="",
            symbol="",
            uri="",
            mint=mint_pubkey,
            platform=Platform.PUMP_FUN,
            bonding_curve=bonding_curve,
            associated_bonding_curve=associated_bonding_curve,
            creator_vault=creator_vault,
            user=user,
            creator=creator,
        )

        print("TokenInfo created:")
        print(f"  Mint: {token_info.mint}")
        print(f"  Bonding curve: {token_info.bonding_curve}")
        print(f"  Associated bonding curve: {token_info.associated_bonding_curve}")
        print(f"  Creator vault: {token_info.creator_vault}")
        print(f"  User: {token_info.user}")
        if token_info.creator:
            print(f"  Creator: {token_info.creator}")
        print()

        # Build instruction accounts using address provider
        # This ensures we have all required accounts in the correct format
        print("Building instruction accounts using address provider...")
        
        # Determine if this is a buy or sell transaction
        instruction_type = (
            extracted_accounts.get("_instruction_type") if extracted_accounts else "buy"
        )
        
        if instruction_type == "sell":
            instruction_accounts = address_provider.get_sell_instruction_accounts(
                token_info, user
            )
        else:
            instruction_accounts = address_provider.get_buy_instruction_accounts(
                token_info, user
            )

        # Override with any accounts we extracted from the transaction if available
        if extracted_accounts:
            for key in ["user_token_account", "fee", "creator_vault"]:
                if key in extracted_accounts:
                    instruction_accounts[key] = extracted_accounts[key]
        
        print(f"Transaction type: {instruction_type}")

        print("Instruction accounts:")
        for key, value in instruction_accounts.items():
            print(f"  {key}: {value}")
        print()

        # Analyze balance changes
        print("Analyzing balance changes...")
        result = balance_analyzer.analyze_balance_changes(
            tx, token_info, wallet_pubkey, instruction_accounts
        )

        print()
        print("=" * 80)
        print("BALANCE ANALYSIS RESULT")
        print("=" * 80)
        print(result)
        print("=" * 80)

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())

