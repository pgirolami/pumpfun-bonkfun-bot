"""
LetsBonk implementation of CurveManager interface.

This module handles LetsBonk (Raydium LaunchLab) specific pool operations
by implementing the CurveManager interface using IDL-based decoding.
"""

from typing import Any

from solders.pubkey import Pubkey

from core.client import SolanaClient
from core.pubkeys import LAMPORTS_PER_SOL, TOKEN_DECIMALS
from interfaces.core import CurveManager, Platform
from platforms.letsbonk.address_provider import LetsBonkAddressProvider
from utils.idl_parser import IDLParser
from utils.logger import get_logger

logger = get_logger(__name__)


class LetsBonkCurveManager(CurveManager):
    """LetsBonk (Raydium LaunchLab) implementation of CurveManager interface."""

    def __init__(self, client: SolanaClient, idl_parser: IDLParser):
        """Initialize LetsBonk curve manager with injected IDL parser.

        Args:
            client: Solana RPC client
            idl_parser: Pre-loaded IDL parser for LetsBonk platform
        """
        self.client = client
        self.address_provider = LetsBonkAddressProvider()
        self._idl_parser = idl_parser
        self.constants: dict[str, Any] = {}
        self._constants_loaded = False

        logger.info("LetsBonk curve manager initialized with injected IDL parser")

    @property
    def platform(self) -> Platform:
        """Get the platform this manager serves."""
        return Platform.LETS_BONK

    async def get_pool_state(self, pool_address: Pubkey) -> dict[str, Any]:
        """Get the current state of a LetsBonk pool.

        Args:
            pool_address: Address of the pool state account

        Returns:
            Dictionary containing pool state data
        """
        try:
            account = await self.client.get_account_info(pool_address)
            if not account.data:
                raise ValueError(f"No data in pool state account {pool_address}")

            # Decode pool state using injected IDL parser
            pool_state_data = self._decode_pool_state_with_idl(account.data)

            return pool_state_data

        except Exception as e:
            logger.exception("Failed to get pool state")
            raise ValueError(f"Invalid pool state: {e!s}")

    async def calculate_price(self, pool_address: Pubkey) -> float:
        """Calculate current token price from pool state.

        Args:
            pool_address: Address of the pool state

        Returns:
            Current token price in SOL
        """
        pool_state = await self.get_pool_state(pool_address)

        # Use virtual reserves for price calculation
        virtual_base = pool_state["virtual_base"]
        virtual_quote = pool_state["virtual_quote"]

        if virtual_base <= 0 or virtual_quote <= 0:
            raise ValueError("Invalid reserve state")

        # Price = quote_reserves / base_reserves (how much SOL per token)
        price_lamports = virtual_quote / virtual_base
        price_sol = price_lamports * (10**TOKEN_DECIMALS) / LAMPORTS_PER_SOL

        return price_sol


    async def calculate_buy_amount_out(
        self, pool_address: Pubkey, amount_in: int
    ) -> int:
        """Calculate expected tokens received for a buy operation.

        Uses the constant product AMM formula.

        Args:
            pool_address: Address of the pool state
            amount_in: Amount of SOL to spend (in lamports)

        Returns:
            Expected amount of tokens to receive (in raw token units)
        """
        pool_state = await self.get_pool_state(pool_address)

        virtual_base = pool_state["virtual_base"]
        virtual_quote = pool_state["virtual_quote"]

        # Constant product formula: tokens_out = (amount_in * virtual_base) / (virtual_quote + amount_in)
        numerator = amount_in * virtual_base
        denominator = virtual_quote + amount_in

        if denominator == 0:
            return 0

        tokens_out = numerator // denominator
        return tokens_out

    async def calculate_sell_amount_out(
        self, mint:Pubkey, pool_address: Pubkey, amount_in: int
    ) -> int:
        """Calculate expected SOL received for a sell operation.

        Uses the constant product AMM formula.

        Args:
            pool_address: Address of the pool state
            amount_in: Amount of tokens to sell (in raw token units)

        Returns:
            Expected amount of SOL to receive (in lamports)
        """
        pool_state = await self.get_pool_state(pool_address)

        virtual_base = pool_state["virtual_base"]
        virtual_quote = pool_state["virtual_quote"]

        # Constant product formula: sol_out = (amount_in * virtual_quote) / (virtual_base + amount_in)
        numerator = amount_in * virtual_quote
        denominator = virtual_base + amount_in

        if denominator == 0:
            return 0

        sol_out = numerator // denominator
        return sol_out

    async def get_reserves(self, mint:Pubkey, pool_address: Pubkey) -> tuple[int, int]:
        """Get current pool reserves.

        Args:
            pool_address: Address of the pool state

        Returns:
            Tuple of (base_reserves, quote_reserves) in raw units
        """
        pool_state = await self.get_pool_state(pool_address)
        return (pool_state["virtual_base"], pool_state["virtual_quote"])

    def _decode_pool_state_with_idl(self, data: bytes) -> dict[str, Any]:
        """Decode pool state data using injected IDL parser.

        Args:
            data: Raw account data

        Returns:
            Dictionary with decoded pool state

        Raises:
            ValueError: If IDL parsing fails
        """
        # Use injected IDL parser to decode PoolState account data
        decoded_pool_state = self._idl_parser.decode_account_data(
            data, "PoolState", skip_discriminator=True
        )

        if not decoded_pool_state:
            raise ValueError("Failed to decode pool state with IDL parser")

        # Extract the fields we need for trading calculations
        # Based on the PoolState structure from the IDL
        pool_data = {
            "virtual_base": decoded_pool_state.get("virtual_base", 0),
            "virtual_quote": decoded_pool_state.get("virtual_quote", 0),
            "real_base": decoded_pool_state.get("real_base", 0),
            "real_quote": decoded_pool_state.get("real_quote", 0),
            "status": decoded_pool_state.get("status", 0),
            "supply": decoded_pool_state.get("supply", 0),
        }

        # Calculate additional metrics
        if pool_data["virtual_base"] > 0:
            pool_data["price_per_token"] = (
                (pool_data["virtual_quote"] / pool_data["virtual_base"])
                * (10**TOKEN_DECIMALS)
                / LAMPORTS_PER_SOL
            )
        else:
            pool_data["price_per_token"] = 0

        logger.debug(
            f"Decoded pool state: virtual_base={pool_data['virtual_base']}, "
            f"virtual_quote={pool_data['virtual_quote']}, "
            f"price={pool_data['price_per_token']:.8f} SOL"
        )

        return pool_data

    async def validate_pool_state_structure(self, pool_address: Pubkey) -> bool:
        """Validate that the pool state structure matches IDL expectations.

        Args:
            pool_address: Address of the pool state

        Returns:
            True if structure is valid, False otherwise
        """
        try:
            # This would be used during development/testing to ensure
            # the IDL parsing is working correctly
            pool_state = await self.get_pool_state(pool_address)

            required_fields = [
                "virtual_base",
                "virtual_quote",
                "real_base",
                "real_quote",
            ]

            for field in required_fields:
                if field not in pool_state:
                    logger.error(f"Missing required field: {field}")
                    return False

                if not isinstance(pool_state[field], int):
                    logger.error(
                        f"Field {field} is not an integer: {type(pool_state[field])}"
                    )
                    return False

            return True

        except Exception:
            logger.exception("Pool state validation failed")
            return False

    async def get_platform_constants(self) -> dict[str, Any]:
        """Get LetsBonk platform constants loaded from the global config account.

        Returns:
            Dictionary containing LetsBonk constants (fees, limits, etc.)
        """
        # Return cached constants if already loaded
        if self._constants_loaded:
            return self.constants
            
        try:
            # Fetch global config account data
            global_config = await self.client.get_account_info(self.address_provider.get_system_addresses()["global_config"])
            if not global_config.data:
                raise ValueError("No data in LetsBonk global config account")
            
            # Parse global config account data
            global_data = self._parse_global_config_data(global_config.data)
            
            # Convert to human-readable format
            constants = {
                "epoch": global_data["epoch"],
                "curve_type": global_data["curve_type"],
                "index": global_data["index"],
                "migrate_fee": global_data["migrate_fee"],
                "trade_fee_rate": global_data["trade_fee_rate"],
                "max_share_fee_rate": global_data["max_share_fee_rate"],
                "min_base_supply": global_data["min_base_supply"],
                "max_lock_rate": global_data["max_lock_rate"],
                "min_base_sell_rate": global_data["min_base_sell_rate"],
                "min_base_migrate_rate": global_data["min_base_migrate_rate"],
                "min_quote_fund_raising": global_data["min_quote_fund_raising"],
                "quote_mint": str(global_data["quote_mint"]),
                "protocol_fee_owner": str(global_data["protocol_fee_owner"]),
                # Human-readable versions
                "trade_fee_percentage": global_data["trade_fee_rate"] / 1_000_000,  # Convert from hundredths of a bip
                "max_share_fee_percentage": global_data["max_share_fee_rate"] / 1_000_000,
                "max_lock_percentage": global_data["max_lock_rate"] / 1_000_000,
                "min_base_sell_percentage": global_data["min_base_sell_rate"] / 1_000_000,
                "min_base_migrate_percentage": global_data["min_base_migrate_rate"] / 1_000_000,
                "min_quote_fund_raising_sol": global_data["min_quote_fund_raising"] / LAMPORTS_PER_SOL,
                "min_base_supply_decimal": global_data["min_base_supply"] / 10**TOKEN_DECIMALS,
                "migrate_fee_sol": global_data["migrate_fee"] / LAMPORTS_PER_SOL,
            }
            
            logger.info(f"Loaded LetsBonk constants: trade_fee={constants['trade_fee_percentage']:.4f}%, min_base_supply={constants['min_base_supply_decimal']:,.0f} tokens")
            
            # Store constants for use by other methods and mark as loaded
            self.constants = constants
            self._constants_loaded = True
            
            return constants
            
        except Exception as e:
            logger.exception("Failed to load LetsBonk platform constants")
            raise ValueError(f"Failed to load platform constants: {e!s}")

    def _parse_global_config_data(self, data: bytes) -> dict[str, Any]:
        """Parse the LetsBonk global config account data.
        
        Args:
            data: Raw account data from RPC
            
        Returns:
            Dictionary containing parsed global config fields
        """
        import struct
        
        if len(data) < 8:
            raise ValueError("Account data too short")
        
        # Expected discriminator for GlobalConfig account (from IDL)
        expected_discriminator = bytes([149, 8, 156, 202, 160, 252, 176, 217])
        discriminator = data[:8]
        if discriminator != expected_discriminator:
            raise ValueError(f"Invalid discriminator: {discriminator.hex()}")
        
        # Parse fields according to IDL structure
        offset = 8
        fields = {}
        
        # epoch (u64)
        fields["epoch"] = struct.unpack("<Q", data[offset:offset + 8])[0]
        offset += 8
        
        # curve_type (u8)
        fields["curve_type"] = data[offset]
        offset += 1
        
        # index (u16)
        fields["index"] = struct.unpack("<H", data[offset:offset + 2])[0]
        offset += 2
        
        # migrate_fee (u64)
        fields["migrate_fee"] = struct.unpack("<Q", data[offset:offset + 8])[0]
        offset += 8
        
        # trade_fee_rate (u64)
        fields["trade_fee_rate"] = struct.unpack("<Q", data[offset:offset + 8])[0]
        offset += 8
        
        # max_share_fee_rate (u64)
        fields["max_share_fee_rate"] = struct.unpack("<Q", data[offset:offset + 8])[0]
        offset += 8
        
        # min_base_supply (u64)
        fields["min_base_supply"] = struct.unpack("<Q", data[offset:offset + 8])[0]
        offset += 8
        
        # max_lock_rate (u64)
        fields["max_lock_rate"] = struct.unpack("<Q", data[offset:offset + 8])[0]
        offset += 8
        
        # min_base_sell_rate (u64)
        fields["min_base_sell_rate"] = struct.unpack("<Q", data[offset:offset + 8])[0]
        offset += 8
        
        # min_base_migrate_rate (u64)
        fields["min_base_migrate_rate"] = struct.unpack("<Q", data[offset:offset + 8])[0]
        offset += 8
        
        # min_quote_fund_raising (u64)
        fields["min_quote_fund_raising"] = struct.unpack("<Q", data[offset:offset + 8])[0]
        offset += 8
        
        # quote_mint (pubkey)
        fields["quote_mint"] = Pubkey.from_bytes(data[offset:offset + 32])
        offset += 32
        
        # protocol_fee_owner (pubkey)
        fields["protocol_fee_owner"] = Pubkey.from_bytes(data[offset:offset + 32])
        offset += 32
        
        return fields
