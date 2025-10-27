"""
Pump.Fun implementation of CurveManager interface.

This module handles pump.fun-specific bonding curve operations
by implementing the CurveManager interface using IDL-based decoding.
"""

from typing import Any

from solders.pubkey import Pubkey

from core.client import SolanaClient
from core.pubkeys import LAMPORTS_PER_SOL, TOKEN_DECIMALS
from interfaces.core import CurveManager, Platform
from monitoring.base_listener import BaseTokenListener
from utils.idl_parser import IDLParser
from utils.logger import get_logger

logger = get_logger(__name__)


class PumpFunCurveManager(CurveManager):
    """Pump.Fun implementation of CurveManager interface using IDL-based decoding."""

    def __init__(
        self, 
        client: SolanaClient, 
        idl_parser: IDLParser,
        listener: BaseTokenListener | None = None,
        trade_staleness_threshold: float = 30.0
    ):
        """Initialize pump.fun curve manager with injected IDL parser.

        Args:
            client: Solana RPC client
            idl_parser: Pre-loaded IDL parser for pump.fun platform
            listener: Optional listener for trade tracking
            trade_staleness_threshold: Seconds before trade data is considered stale
        """
        self.client = client
        self._idl_parser = idl_parser
        self.listener = listener
        self.trade_staleness_threshold = trade_staleness_threshold
        self.constants: dict[str, Any] = {}
        self._constants_loaded = False

        logger.info("Pump.Fun curve manager initialized with injected IDL parser")

    @property
    def platform(self) -> Platform:
        """Get the platform this manager serves."""
        return Platform.PUMP_FUN

    async def get_pool_state(self, pool_address: Pubkey) -> dict[str, Any]:
        """Get the current state of a pump.fun bonding curve.

        Args:
            pool_address: Address of the bonding curve

        Returns:
            Dictionary containing bonding curve state data
        """
        try:
            account = await self.client.get_account_info(pool_address)
            if not account.data:
                raise ValueError(f"No data in bonding curve account {pool_address}")

            # Decode bonding curve state using injected IDL parser
            curve_state_data = self._decode_curve_state_with_idl(account.data)

            return curve_state_data

        except Exception as e:
            # logger.exception("Failed to get curve state")
            raise ValueError(f"Invalid bonding curve state: {e!s}")

    async def calculate_price(self, mint:Pubkey, pool_address: Pubkey) -> float:
        """Calculate current token price from bonding curve state.

        Args:
            pool_address: Address of the bonding curve

        Returns:
            Current token price in SOL
        """
        # Check if we have trade tracking available
        if self.listener:
            tracker = self.listener.get_trade_tracker_by_mint(str(mint))
            if tracker and not tracker.is_stale(self.trade_staleness_threshold):
                try:
                    price = tracker.calculate_price()
                    logger.debug(f"Using trade tracker price: {price:.10f} SOL")
                    return price
                except RuntimeError:
                    logger.debug("Trade tracker not initialized, falling back to RPC")
            else:
                logger.info("Trade tracker is stale, using RPC")

        
        # Fallback to RPC-based calculation
        logger.info("Trade tracker not available or stale, using RPC")
        pool_state = await self.get_pool_state(pool_address)

        # Use virtual reserves for price calculation
        virtual_token_reserves = pool_state["virtual_token_reserves"]
        virtual_sol_reserves = pool_state["virtual_sol_reserves"]

        logger.info(f"[{str(mint)[:8]}] (RPC) Virtual token reserves: {virtual_token_reserves}, Virtual sol reserves: {virtual_sol_reserves}")
        if virtual_token_reserves <= 0:
            return 0.0

        # Price = sol_reserves / token_reserves
        price_lamports = virtual_sol_reserves / virtual_token_reserves
        return price_lamports * (10**TOKEN_DECIMALS) / LAMPORTS_PER_SOL

    async def calculate_buy_amount_out(
        self, mint:Pubkey, pool_address: Pubkey, amount_in: int
    ) -> int:
        """Calculate expected SOL used for a buy operation in PumpFun.

        Uses the pump.fun bonding curve formula to calculate token output.

        Args:
            pool_address: Address of the bonding curve
            amount_in: Amount of token to buy (in raw units)

        Returns:
            Expected amount of lamports to spend
        """
        virtual_token_reserves, virtual_sol_reserves = await self.get_reserves(mint, pool_address)

        # k = virtual_token_reserves * virtual_sol_reserves
        # and
        # k = (virtual_token_reserves - token_swap_amount) * (virtual_sol_reserves + sol_swap_amount)
        # 
        # So virtual_token_reserves * virtual_sol_reserves = (virtual_token_reserves - token_swap_amount) * (virtual_sol_reserves + sol_swap_amount)
        # => virtual_sol_reserves + sol_swap_amount = (virtual_token_reserves * virtual_sol_reserves) / (virtual_token_reserves - token_swap_amount)
        # => sol_swap_amount = (virtual_token_reserves * virtual_sol_reserves) / (virtual_token_reserves - token_swap_amount) - virtual_sol_reserves
        # => sol_swap_amount = (virtual_token_reserves * virtual_sol_reserves) / (virtual_token_reserves - token_swap_amount) - virtual_sol_reserves * (virtual_token_reserves - token_swap_amount) / (virtual_token_reserves - token_swap_amount)
        # => sol_swap_amount = (virtual_token_reserves * virtual_sol_reserves - virtual_sol_reserves * virtual_token_reserves + virtual_sol_reserves * token_swap_amount) / (virtual_token_reserves - token_swap_amount)
        # => sol_swap_amount = (virtual_sol_reserves * token_swap_amount) / (virtual_token_reserves - token_swap_amount)

        numerator = virtual_sol_reserves * amount_in
        denominator = virtual_token_reserves - amount_in

        tokens_out = numerator // denominator

        return tokens_out

    async def calculate_sell_amount_out(
        self, mint:Pubkey, pool_address: Pubkey, amount_in: int
    ) -> int:
        """Calculate expected SOL received for a sell operation.

        Uses the pump.fun bonding curve formula to calculate SOL output.

        Args:
            pool_address: Address of the bonding curve
            amount_in: Amount of tokens to sell (in raw token units)

        Returns:
            Expected amount of SOL to receive (in lamports)
        """
        virtual_token_reserves, virtual_sol_reserves = await self.get_reserves(mint, pool_address)


        # k = virtual_token_reserves * virtual_sol_reserves
        # and
        # k = (virtual_token_reserves - token_swap_amount) * (virtual_sol_reserves + sol_swap_amount)
        # 
        # So virtual_token_reserves * virtual_sol_reserves = (virtual_token_reserves + token_swap_amount) * (virtual_sol_reserves - sol_swap_amount)
        # => virtual_sol_reserves - sol_swap_amount = (virtual_token_reserves * virtual_sol_reserves) / (virtual_token_reserves + token_swap_amount)
        # => sol_swap_amount = virtual_sol_reserves - (virtual_token_reserves * virtual_sol_reserves) / (virtual_token_reserves + token_swap_amount)
        # => sol_swap_amount = virtual_sol_reserves * (virtual_token_reserves + token_swap_amount) / (virtual_token_reserves + token_swap_amount) - (virtual_token_reserves * virtual_sol_reserves) / (virtual_token_reserves + token_swap_amount)
        # => sol_swap_amount = (virtual_sol_reserves * token_swap_amount + virtual_token_reserves * virtual_sol_reserves - virtual_token_reserves * virtual_sol_reserves) / (virtual_token_reserves + token_swap_amount)
        # => sol_swap_amount = (virtual_sol_reserves * token_swap_amount) / (virtual_token_reserves + token_swap_amount)

        numerator = virtual_sol_reserves * amount_in
        denominator = virtual_token_reserves + amount_in

        sol_out = numerator // denominator
        return sol_out

    async def get_reserves(self, mint: Pubkey, pool_address: Pubkey) -> tuple[int, int]:
        """Get current bonding curve reserves.

        Args:
            pool_address: Address of the bonding curve

        Returns:
            Tuple of (token_reserves, sol_reserves) in raw units
        """
        virtual_sol_reserves, virtual_token_reserves = (None,None)
        if self.listener:
            tracker = self.listener.get_trade_tracker_by_mint(str(mint))
            if tracker and not tracker.is_stale(self.trade_staleness_threshold):
                try:
                    return tracker.get_reserves()
                except RuntimeError:
                    logger.debug("Trade tracker not initialized, falling back to RPC")
            else:
                logger.info("Trade tracker is stale, using RPC")

        if virtual_sol_reserves is None or virtual_token_reserves is None:
            pool_state = await self.get_pool_state(pool_address)
            return (
                pool_state["virtual_token_reserves"],
                pool_state["virtual_sol_reserves"],
            )

    def _decode_curve_state_with_idl(self, data: bytes) -> dict[str, Any]:
        """Decode bonding curve state data using injected IDL parser.

        Args:
            data: Raw account data

        Returns:
            Dictionary with decoded bonding curve state

        Raises:
            ValueError: If IDL parsing fails
        """
        # Use injected IDL parser to decode BondingCurve account data
        decoded_curve_state = self._idl_parser.decode_account_data(
            data, "BondingCurve", skip_discriminator=True
        )

        if not decoded_curve_state:
            raise ValueError("Failed to decode bonding curve state with IDL parser")

        # Extract the fields we need for trading calculations
        # Based on the BondingCurve structure from the IDL
        curve_data = {
            "virtual_token_reserves": decoded_curve_state.get(
                "virtual_token_reserves", 0
            ),
            "virtual_sol_reserves": decoded_curve_state.get("virtual_sol_reserves", 0),
            "real_token_reserves": decoded_curve_state.get("real_token_reserves", 0),
            "real_sol_reserves": decoded_curve_state.get("real_sol_reserves", 0),
            "token_total_supply": decoded_curve_state.get("token_total_supply", 0),
            "complete": decoded_curve_state.get("complete", False),
            "creator": decoded_curve_state.get("creator", ""),
        }

        # Calculate additional metrics
        if curve_data["virtual_token_reserves"] > 0:
            curve_data["price_per_token"] = (
                (
                    curve_data["virtual_sol_reserves"]
                    / curve_data["virtual_token_reserves"]
                )
                * (10**TOKEN_DECIMALS)
                / LAMPORTS_PER_SOL
            )
        else:
            curve_data["price_per_token"] = 0

        # Add convenience decimal fields
        curve_data["token_reserves_decimal"] = (
            curve_data["virtual_token_reserves"] / 10**TOKEN_DECIMALS
        )
        curve_data["sol_reserves_decimal"] = (
            curve_data["virtual_sol_reserves"] / LAMPORTS_PER_SOL
        )

        logger.debug(
            f"Decoded curve state: virtual_token_reserves={curve_data['virtual_token_reserves']}, "
            f"virtual_sol_reserves={curve_data['virtual_sol_reserves']}, "
            f"price={curve_data['price_per_token']:.8f} SOL"
        )

        return curve_data

    # Additional convenience methods for pump.fun specific operations
    async def calculate_expected_tokens(
        self, pool_address: Pubkey, sol_amount: float
    ) -> float:
        """Calculate the expected token amount for a given SOL input.

        This is a convenience method that converts between decimal and raw units.

        Args:
            pool_address: Address of the bonding curve
            sol_amount: Amount of SOL to spend (in decimal SOL)

        Returns:
            Expected token amount (in decimal tokens)
        """
        sol_lamports = int(sol_amount * LAMPORTS_PER_SOL)
        tokens_raw = await self.calculate_buy_amount_out(pool_address, sol_lamports)
        return tokens_raw / 10**TOKEN_DECIMALS

    async def calculate_expected_sol(
        self, pool_address: Pubkey, token_amount: float
    ) -> float:
        """Calculate the expected SOL amount for a given token input.

        This is a convenience method that converts between decimal and raw units.

        Args:
            pool_address: Address of the bonding curve
            token_amount: Amount of tokens to sell (in decimal tokens)

        Returns:
            Expected SOL amount (in decimal SOL)
        """
        tokens_raw = int(token_amount * 10**TOKEN_DECIMALS)
        sol_lamports = await self.calculate_sell_amount_out(pool_address, tokens_raw)
        return sol_lamports / LAMPORTS_PER_SOL

    async def is_curve_complete(self, pool_address: Pubkey) -> bool:
        """Check if the bonding curve is complete (migrated to Raydium).

        Args:
            pool_address: Address of the bonding curve

        Returns:
            True if curve is complete, False otherwise
        """
        pool_state = await self.get_pool_state(pool_address)
        return pool_state.get("complete", False)

    async def get_curve_progress(self, pool_address: Pubkey) -> dict[str, Any]:
        """Get bonding curve completion progress information.

        Args:
            pool_address: Address of the bonding curve

        Returns:
            Dictionary with progress information
        """
        pool_state = await self.get_pool_state(pool_address)

        # Calculate progress based on SOL raised vs target
        # This is approximate since the exact target isn't stored in the curve state
        sol_raised = pool_state["real_sol_reserves"] / LAMPORTS_PER_SOL

        # Estimate progress based on typical pump.fun graduation requirements
        # (This could be made more accurate with additional on-chain data)
        estimated_target_sol = 85.0  # Typical pump.fun graduation target
        progress_percentage = min((sol_raised / estimated_target_sol) * 100, 100.0)

        return {
            "complete": pool_state.get("complete", False),
            "sol_raised": sol_raised,
            "estimated_target_sol": estimated_target_sol,
            "progress_percentage": progress_percentage,
            "tokens_available": pool_state["virtual_token_reserves"]
            / 10**TOKEN_DECIMALS,
            "market_cap_sol": sol_raised,  # Approximate market cap
        }

    def validate_curve_state_structure(self, pool_address: Pubkey) -> bool:
        """Validate that the curve state structure matches IDL expectations.

        Args:
            pool_address: Address of the bonding curve

        Returns:
            True if structure is valid, False otherwise
        """
        try:
            # This would be used during development/testing to ensure
            # the IDL parsing is working correctly
            pool_state = self.get_pool_state(pool_address)

            required_fields = [
                "virtual_token_reserves",
                "virtual_sol_reserves",
                "real_token_reserves",
                "real_sol_reserves",
                "token_total_supply",
                "complete",
            ]

            for field in required_fields:
                if field not in pool_state:
                    logger.error(f"Missing required field: {field}")
                    return False

                if field != "complete" and not isinstance(pool_state[field], int):
                    logger.error(
                        f"Field {field} is not an integer: {type(pool_state[field])}"
                    )
                    return False

            return True

        except Exception:
            logger.exception("Curve state validation failed")
            return False

    async def get_platform_constants(self) -> dict[str, Any]:
        """Get pump.fun platform constants loaded from the global account.

        Returns:
            Dictionary containing pump.fun constants (initial reserves, fees, etc.)
        """
        # Return cached constants if already loaded
        if self._constants_loaded:
            return self.constants
            
        try:
            # Fetch global account data
            from platforms.pumpfun.address_provider import PumpFunAddresses
            
            global_account = await self.client.get_account_info(PumpFunAddresses.GLOBAL)
            if not global_account.data:
                raise ValueError("No data in pump.fun global account")
            
            # Parse global account data
            global_data = self._parse_global_account_data(global_account.data)
            
            # Convert to human-readable format
            constants = {
                "initialized": global_data["initialized"],
                "authority": str(global_data["authority"]),
                "fee_recipient": str(global_data["fee_recipient"]),
                "withdraw_authority": str(global_data["withdraw_authority"]),
                "enable_migrate": global_data["enable_migrate"],
                "initial_virtual_token_reserves": global_data["initial_virtual_token_reserves"],
                "initial_virtual_sol_reserves": global_data["initial_virtual_sol_reserves"],
                "initial_real_token_reserves": global_data["initial_real_token_reserves"],
                "token_total_supply": global_data["token_total_supply"],
                "fee_basis_points": global_data["fee_basis_points"],
                # Human-readable versions
                "initial_virtual_token_reserves_decimal": global_data["initial_virtual_token_reserves"] / 10**TOKEN_DECIMALS,
                "initial_virtual_sol_reserves_decimal": global_data["initial_virtual_sol_reserves"] / LAMPORTS_PER_SOL,
                "initial_real_token_reserves_decimal": global_data["initial_real_token_reserves"] / 10**TOKEN_DECIMALS,
                "token_total_supply_decimal": global_data["token_total_supply"] / 10**TOKEN_DECIMALS,
                "fee_percentage": global_data["fee_basis_points"] / 100,
                # Calculated values
                "starting_price_sol": (global_data["initial_virtual_sol_reserves"] / LAMPORTS_PER_SOL) / (global_data["initial_virtual_token_reserves"] / 10**TOKEN_DECIMALS),
                "starting_price_lamports": global_data["initial_virtual_sol_reserves"] / global_data["initial_virtual_token_reserves"],
            }
            
            logger.info(f"Loaded pump.fun constants: {constants['initial_virtual_token_reserves_decimal']:,.0f} tokens, {constants['initial_virtual_sol_reserves_decimal']:,.0f} SOL, {constants['fee_percentage']:.2f}% fee")
            
            # Store constants for use by other methods and mark as loaded
            self.constants = constants
            self._constants_loaded = True
            
            return constants
            
        except Exception as e:
            logger.exception("Failed to load pump.fun platform constants")
            raise ValueError(f"Failed to load platform constants: {e!s}")

    def _parse_global_account_data(self, data: bytes) -> dict[str, Any]:
        """Parse the pump.fun global account data.
        
        Args:
            data: Raw account data from RPC
            
        Returns:
            Dictionary containing parsed global account fields
        """
        import struct
        
        if len(data) < 8:
            raise ValueError("Account data too short")
        
        # Expected discriminator for Global account (from IDL)
        expected_discriminator = bytes([167, 232, 232, 177, 200, 108, 114, 127])
        discriminator = data[:8]
        if discriminator != expected_discriminator:
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
