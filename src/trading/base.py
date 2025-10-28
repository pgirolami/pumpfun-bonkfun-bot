"""
Enhanced base interfaces for trading operations with platform support.

This module provides the complete enhanced base classes that replace the existing
trading/base.py while maintaining full backward compatibility. It integrates the
new interface system with the existing trading infrastructure.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from solders.pubkey import Pubkey

# Import from interfaces to avoid duplication
from interfaces.core import Platform, TokenInfo


@dataclass
class TradeResult:
    """Enhanced result of a trading operation with platform support."""

    success: bool
    platform: Platform = Platform.PUMP_FUN  # Add platform tracking
    tx_signature: str | None = None
    error_message: str | None = None
    block_time: int | None = None  # Unix epoch milliseconds from block time
    # Actual amounts from transaction analysis
    token_swap_amount_raw: int | None = None  # Raw token amount from balance change
    sol_swap_amount_raw: int | None = None  # Raw SOL amount from balance change (including transaction fee)
    net_sol_swap_amount_raw: int | None = None
    transaction_fee_raw: int | None = None  # Base + priority transaction fee in lamports (from meta.fee)
    platform_fee_raw: int | None = None  # Platform fee in lamports (includes creator + platform fees)
    trade_duration_ms: int | None = None  # Trade execution duration in milliseconds

    def token_swap_amount_decimal(self) -> float | None:
        """Get token amount in decimal form.
        
        Returns:
            Token amount in decimal form, or None if not available
        """
        if self.token_swap_amount_raw is None:
            return None
        from core.pubkeys import TOKEN_DECIMALS
        return self.token_swap_amount_raw / (10 ** TOKEN_DECIMALS)

    def sol_swap_amount_decimal(self) -> float | None:
        """Get SOL amount in decimal form.
        
        Returns:
            SOL amount in decimal form, or None if not available
        """
        if self.sol_swap_amount_raw is None:
            return None
        from core.pubkeys import LAMPORTS_PER_SOL
        return self.sol_swap_amount_raw / LAMPORTS_PER_SOL

    def price_sol_decimal(self) -> float | None:
        """Get price per token in SOL.
        
        Returns:
            Price per token in SOL, or None if not available
        """
        if self.token_swap_amount_raw is None or self.sol_swap_amount_raw is None or self.token_swap_amount_raw == 0:
            return None
        from core.pubkeys import LAMPORTS_PER_SOL, TOKEN_DECIMALS
        return (abs(self.sol_swap_amount_raw / self.token_swap_amount_raw)) * (10 ** TOKEN_DECIMALS) / LAMPORTS_PER_SOL


    def net_price_sol_decimal(self) -> float | None:
        """Get net price per token in SOL (excluding platform and transaction fees).
        
        Returns:
            Net price per token in SOL, or None if not available
        """
        if self.token_swap_amount_raw is None or self.token_swap_amount_raw == 0:
            return None
        
        if self.net_sol_swap_amount_raw is None:
            return None
        
        from core.pubkeys import LAMPORTS_PER_SOL, TOKEN_DECIMALS
        
        result = abs((self.net_sol_swap_amount_raw / LAMPORTS_PER_SOL) / (self.token_swap_amount_raw / (10 ** TOKEN_DECIMALS)))
        
        return result

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging/serialization.

        Returns:
            Dictionary representation of the trade result
        """
        return {
            "success": self.success,
            "platform": self.platform.value,
            "tx_signature": self.tx_signature,
            "error_message": self.error_message,
            "transaction_fee_raw": self.transaction_fee_raw,
            "token_swap_amount_raw": self.token_swap_amount_raw,
            "sol_swap_amount_raw": self.sol_swap_amount_raw,
            "platform_fee_raw": self.platform_fee_raw,
            # Computed values
            "token_swap_amount_decimal": self.token_swap_amount_decimal(),
            "net_sol_swap_amount_raw": self.net_sol_swap_amount_raw,
            "sol_swap_amount_decimal": self.sol_swap_amount_decimal(),
            "price_decimal": self.price_sol_decimal(),
            "net_price_decimal": self.net_price_sol_decimal(),
        }

    def __str__(self) -> str:
        """String representation of trade result."""
        from core.pubkeys import LAMPORTS_PER_SOL, TOKEN_DECIMALS
        
        result = f"TradeResult(success={self.success}, platform={self.platform.value}"
        
        if self.tx_signature:
            result += f", tx_signature='{self.tx_signature}'"
        
        if self.error_message:
            result += f", error_message='{self.error_message}'"
        
        if self.token_swap_amount_raw is not None:
            token_decimal = self.token_swap_amount_decimal()
            result += f", token_swap_amount_raw={self.token_swap_amount_raw} ({token_decimal:.6f} tokens)" if token_decimal is not None else f", token_swap_amount_raw={self.token_swap_amount_raw}"
        
        if self.sol_swap_amount_raw is not None:
            sol_decimal = self.sol_swap_amount_raw / LAMPORTS_PER_SOL
            result += f", sol_swap_amount_raw={self.sol_swap_amount_raw} ({sol_decimal:.6f} SOL)"

        if self.sol_swap_amount_raw is not None:
            sol_decimal = self.net_sol_swap_amount_raw / LAMPORTS_PER_SOL
            result += f", net_sol_swap_amount_raw={self.net_sol_swap_amount_raw} ({sol_decimal:.6f} SOL)"

        if self.transaction_fee_raw is not None:
            fee_decimal = self.transaction_fee_raw / LAMPORTS_PER_SOL
            result += f", transaction_fee_raw={self.transaction_fee_raw} ({fee_decimal:.6f} SOL)"
        
        if self.platform_fee_raw is not None:
            platform_fee_decimal = self.platform_fee_raw / LAMPORTS_PER_SOL
            result += f", platform_fee_raw={self.platform_fee_raw} ({platform_fee_decimal:.6f} SOL)"
        
        price_decimal = self.price_sol_decimal()
        if price_decimal is not None:
            result += f", price_decimal={price_decimal:.8f} SOL"
        
        net_price_sol_decimal = self.net_price_sol_decimal()
        if net_price_sol_decimal is not None:
            result += f", net_price_sol_decimal={net_price_sol_decimal:.8f} SOL"
        
        result += ")"
        return result


class Trader(ABC):
    """Enhanced base interface for trading operations with platform support."""

    @abstractmethod
    async def execute(self, token_info: TokenInfo, *args, **kwargs) -> TradeResult:
        """Execute trading operation.

        Args:
            token_info: Enhanced token information with platform support

        Returns:
            TradeResult with operation outcome including platform info
        """
        pass

    def _get_relevant_accounts(self, token_info: TokenInfo) -> list[Pubkey]:
        """
        Get the list of accounts relevant for calculating the priority fee.

        This is now platform-agnostic and should be overridden by platform-specific traders.

        Args:
            token_info: Enhanced token information

        Returns:
            List of relevant accounts (basic implementation)
        """
        # Basic implementation - platform-specific traders should override this
        accounts = [token_info.mint]

        if token_info.bonding_curve:
            accounts.append(token_info.bonding_curve)

        if token_info.pool_state:  # For other platforms
            accounts.append(token_info.pool_state)

        return accounts


# Legacy TokenInfo for backward compatibility (keep pump.fun specific)
@dataclass
class TokenInfo_Legacy:
    """Legacy token information structure for backward compatibility."""

    name: str
    symbol: str
    uri: str
    mint: Pubkey
    bonding_curve: Pubkey
    associated_bonding_curve: Pubkey
    user: Pubkey
    creator: Pubkey
    creator_vault: Pubkey

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TokenInfo_Legacy":
        """Create TokenInfo from dictionary.

        Args:
            data: Dictionary with token data

        Returns:
            TokenInfo_Legacy instance
        """
        return cls(
            name=data["name"],
            symbol=data["symbol"],
            uri=data["uri"],
            mint=Pubkey.from_string(data["mint"]),
            bonding_curve=Pubkey.from_string(data["bondingCurve"]),
            associated_bonding_curve=Pubkey.from_string(data["associatedBondingCurve"]),
            user=Pubkey.from_string(data["user"]),
            creator=Pubkey.from_string(data["creator"]),
            creator_vault=Pubkey.from_string(data["creator_vault"]),
        )

    def to_dict(self) -> dict[str, str]:
        """Convert to dictionary.

        Returns:
            Dictionary representation
        """
        return {
            "name": self.name,
            "symbol": self.symbol,
            "uri": self.uri,
            "mint": str(self.mint),
            "bondingCurve": str(self.bonding_curve),
            "associatedBondingCurve": str(self.associated_bonding_curve),
            "user": str(self.user),
            "creator": str(self.creator),
            "creatorVault": str(self.creator_vault),
        }


def upgrade_token_info(legacy_token_info: TokenInfo_Legacy) -> TokenInfo:
    """Convert legacy TokenInfo to enhanced TokenInfo.

    This function allows existing code that creates legacy TokenInfo objects
    to be upgraded to the new enhanced format.

    Args:
        legacy_token_info: Legacy TokenInfo instance

    Returns:
        Enhanced TokenInfo with platform information
    """
    return TokenInfo(
        name=legacy_token_info.name,
        symbol=legacy_token_info.symbol,
        uri=legacy_token_info.uri,
        mint=legacy_token_info.mint,
        platform=Platform.PUMP_FUN,  # Default to pump.fun for legacy tokens
        bonding_curve=legacy_token_info.bonding_curve,
        associated_bonding_curve=legacy_token_info.associated_bonding_curve,
        user=legacy_token_info.user,
        creator=legacy_token_info.creator,
        creator_vault=legacy_token_info.creator_vault,
    )


def create_legacy_token_info(enhanced_token_info: TokenInfo) -> TokenInfo_Legacy:
    """Convert enhanced TokenInfo back to legacy TokenInfo if needed.

    This function allows the enhanced TokenInfo to be used with existing
    code that expects the legacy format.

    Args:
        enhanced_token_info: Enhanced TokenInfo instance

    Returns:
        Legacy TokenInfo instance

    Raises:
        ValueError: If enhanced TokenInfo doesn't have required pump.fun fields
    """
    if enhanced_token_info.platform != Platform.PUMP_FUN:
        raise ValueError("Can only convert pump.fun tokens to legacy format")

    if not all(
        [
            enhanced_token_info.bonding_curve,
            enhanced_token_info.associated_bonding_curve,
            enhanced_token_info.creator_vault,
        ]
    ):
        raise ValueError("Enhanced TokenInfo missing required pump.fun fields")

    return TokenInfo_Legacy(
        name=enhanced_token_info.name,
        symbol=enhanced_token_info.symbol,
        uri=enhanced_token_info.uri,
        mint=enhanced_token_info.mint,
        bonding_curve=enhanced_token_info.bonding_curve,
        associated_bonding_curve=enhanced_token_info.associated_bonding_curve,
        user=enhanced_token_info.user or enhanced_token_info.creator,
        creator=enhanced_token_info.creator or enhanced_token_info.user,
        creator_vault=enhanced_token_info.creator_vault,
    )


def create_pump_fun_token_info(
    name: str,
    symbol: str,
    uri: str,
    mint: Pubkey,
    bonding_curve: Pubkey,
    associated_bonding_curve: Pubkey,
    user: Pubkey,
    creator: Pubkey | None = None,
    creator_vault: Pubkey | None = None,
    **kwargs,
) -> TokenInfo:
    """Convenience function to create pump.fun TokenInfo with proper platform setting.

    Args:
        name: Token name
        symbol: Token symbol
        uri: Token metadata URI
        mint: Token mint address
        bonding_curve: Bonding curve address
        associated_bonding_curve: Associated bonding curve address
        user: User/trader address
        creator: Creator address (defaults to user if not provided)
        creator_vault: Creator vault address (will be derived if not provided)
        **kwargs: Additional fields for TokenInfo

    Returns:
        Enhanced TokenInfo configured for pump.fun
    """
    # Derive creator vault if not provided (import here to avoid circular imports)
    if creator_vault is None and creator:
        # We can't import PumpAddresses here, so this will need to be handled elsewhere
        # For now, leave it as None and let the platform implementation handle it
        pass

    return TokenInfo(
        name=name,
        symbol=symbol,
        uri=uri,
        mint=mint,
        platform=Platform.PUMP_FUN,
        bonding_curve=bonding_curve,
        associated_bonding_curve=associated_bonding_curve,
        user=user,
        creator=creator or user,
        creator_vault=creator_vault,
        **kwargs,
    )


def create_lets_bonk_token_info(
    name: str,
    symbol: str,
    uri: str,
    mint: Pubkey,
    pool_state: Pubkey,
    base_vault: Pubkey,
    quote_vault: Pubkey,
    user: Pubkey,
    creator: Pubkey | None = None,
    **kwargs,
) -> TokenInfo:
    """Convenience function to create LetsBonk TokenInfo with proper platform setting.

    Args:
        name: Token name
        symbol: Token symbol
        uri: Token metadata URI
        mint: Token mint address
        pool_state: Pool state address
        base_vault: Base token vault address
        quote_vault: Quote token vault address
        user: User/trader address
        creator: Creator address (defaults to user if not provided)
        **kwargs: Additional fields for TokenInfo

    Returns:
        Enhanced TokenInfo configured for LetsBonk
    """
    return TokenInfo(
        name=name,
        symbol=symbol,
        uri=uri,
        mint=mint,
        platform=Platform.LETS_BONK,
        pool_state=pool_state,
        base_vault=base_vault,
        quote_vault=quote_vault,
        user=user,
        creator=creator or user,
        **kwargs,
    )


def is_pump_fun_token(token_info: TokenInfo) -> bool:
    """Check if a TokenInfo is for pump.fun platform.

    Args:
        token_info: Token information to check

    Returns:
        True if token is for pump.fun platform
    """
    return token_info.platform == Platform.PUMP_FUN


def is_lets_bonk_token(token_info: TokenInfo) -> bool:
    """Check if a TokenInfo is for LetsBonk platform.

    Args:
        token_info: Token information to check

    Returns:
        True if token is for LetsBonk platform
    """
    return token_info.platform == Platform.LETS_BONK


def get_platform_specific_fields(token_info: TokenInfo) -> dict[str, Any]:
    """Get platform-specific fields from TokenInfo.

    Args:
        token_info: Token information

    Returns:
        Dictionary of platform-specific fields
    """
    if token_info.platform == Platform.PUMP_FUN:
        return {
            "bonding_curve": token_info.bonding_curve,
            "associated_bonding_curve": token_info.associated_bonding_curve,
            "creator_vault": token_info.creator_vault,
        }
    elif token_info.platform == Platform.LETS_BONK:
        return {
            "pool_state": token_info.pool_state,
            "base_vault": token_info.base_vault,
            "quote_vault": token_info.quote_vault,
        }
    else:
        return {}


def validate_token_info(token_info: TokenInfo) -> bool:
    """Validate that TokenInfo has required fields for its platform.

    Args:
        token_info: Token information to validate

    Returns:
        True if TokenInfo is valid for its platform
    """
    # Check common required fields
    if not all(
        [
            token_info.name,
            token_info.symbol,
            token_info.mint,
            token_info.platform,
        ]
    ):
        return False

    # Check platform-specific required fields
    if token_info.platform == Platform.PUMP_FUN:
        return all(
            [
                token_info.bonding_curve,
                token_info.associated_bonding_curve,
            ]
        )
    elif token_info.platform == Platform.LETS_BONK:
        return all(
            [
                token_info.pool_state,
                token_info.base_vault,
                token_info.quote_vault,
            ]
        )

    return True


# Backward compatibility exports
__all__ = [
    "Platform",  # Platform enum
    "TokenInfo",  # Enhanced TokenInfo (main export)
    "TokenInfo_Legacy",  # Legacy TokenInfo for compatibility
    "TradeResult",  # Enhanced TradeResult
    "Trader",  # Enhanced Trader base class
    "create_legacy_token_info",
    "create_lets_bonk_token_info",
    "create_pump_fun_token_info",
    "get_platform_specific_fields",
    "is_lets_bonk_token",
    "is_pump_fun_token",
    "upgrade_token_info",
    "validate_token_info",
]
