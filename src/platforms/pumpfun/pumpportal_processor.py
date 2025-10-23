"""
PumpFun-specific PumpPortal event processor.
File: src/platforms/pumpfun/pumpportal_processor.py
"""

from solders.pubkey import Pubkey

from core.pubkeys import LAMPORTS_PER_SOL, TOKEN_DECIMALS
from interfaces.core import Platform, TokenInfo
from platforms.pumpfun.address_provider import PumpFunAddressProvider
from utils.logger import get_logger

logger = get_logger(__name__)


class PumpFunPumpPortalProcessor:
    """PumpPortal processor for pump.fun tokens."""

    def __init__(self):
        """Initialize the processor with address provider."""
        self.address_provider = PumpFunAddressProvider()

    @property
    def platform(self) -> Platform:
        """Get the platform this processor handles."""
        return Platform.PUMP_FUN

    @property
    def supported_pool_names(self) -> list[str]:
        """Get the pool names this processor supports from PumpPortal."""
        return ["pump"]  # PumpPortal pool name for pump.fun

    def can_process(self, token_data: dict) -> bool:
        """Check if this processor can handle the given token data.

        Args:
            token_data: Token data from PumpPortal

        Returns:
            True if this processor can handle the token data
        """
        pool = token_data.get("pool", "").lower()
        return pool in self.supported_pool_names

    def process_token_data(self, token_data: dict) -> TokenInfo | None:
        """Process pump.fun token data from PumpPortal.

        Args:
            token_data: Token data from PumpPortal WebSocket

        Returns:
            TokenInfo if token creation found, None otherwise
        """
        try:
            # Extract required fields
            name = token_data.get("name", "")
            symbol = token_data.get("symbol", "")
            mint_str = token_data.get("mint")
            bonding_curve_str = token_data.get("bondingCurveKey")
            creator_str = token_data.get("traderPublicKey")  # Maps to user field
            uri = token_data.get("uri", "")

            # Extract PumpPortal-specific initial buy data
            initial_buy = token_data.get("initialBuy")  # Decimal token amount
            sol_amount = token_data.get("solAmount")  # SOL decimal amount

            # Extract virtual reserves (state after creator's first buy)
            v_sol_in_curve = token_data.get("vSolInBondingCurve")  # Decimal SOL
            v_tokens_in_curve = token_data.get("vTokensInBondingCurve")  # Decimal tokens
            
            # Convert to raw units for trade tracking
            virtual_sol_reserves = int(v_sol_in_curve * LAMPORTS_PER_SOL) if v_sol_in_curve is not None else None
            virtual_token_reserves = int(v_tokens_in_curve * 10**TOKEN_DECIMALS) if v_tokens_in_curve is not None else None

            # Additional fields available from PumpPortal but not currently used:
            # - marketCapSol: Market cap in SOL
            # - signature: Transaction signature

            if not all([name, symbol, mint_str, bonding_curve_str, creator_str]):
                logger.warning("Missing required fields in PumpPortal token data")
                return None

            # Convert string addresses to Pubkey objects
            mint = Pubkey.from_string(mint_str)
            bonding_curve = Pubkey.from_string(bonding_curve_str)
            user = Pubkey.from_string(creator_str)

            # For PumpPortal, we assume the creator is the same as the user
            # since PumpPortal doesn't distinguish between them
            creator = user

            # Derive additional addresses using platform provider
            associated_bonding_curve = (
                self.address_provider.derive_associated_bonding_curve(
                    mint, bonding_curve
                )
            )
            creator_vault = self.address_provider.derive_creator_vault(creator)

            return TokenInfo(
                name=name,
                symbol=symbol,
                uri=uri,
                mint=mint,
                platform=Platform.PUMP_FUN,
                bonding_curve=bonding_curve,
                associated_bonding_curve=associated_bonding_curve,
                user=user,
                creator=creator,
                creator_vault=creator_vault,
                initial_buy_token_amount_decimal=initial_buy,
                initial_buy_sol_amount_decimal=sol_amount,
                virtual_sol_reserves=virtual_sol_reserves,
                virtual_token_reserves=virtual_token_reserves,
            )

        except Exception:
            logger.exception("Failed to process PumpPortal token data")
            return None
