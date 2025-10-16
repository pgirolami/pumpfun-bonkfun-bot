"""
Platform-aware trader implementations that use the interface system.
Final cleanup removing all platform-specific hardcoding.
"""

from solders.pubkey import Pubkey

from core.client import SolanaClient
from core.priority_fee.manager import PriorityFeeManager
from core.pubkeys import LAMPORTS_PER_SOL, TOKEN_DECIMALS
from core.wallet import Wallet
from interfaces.core import AddressProvider, Platform, TokenInfo
from platforms import get_platform_implementations
from trading.base import Trader, TradeResult
from trading.position import Position
from utils.logger import get_logger

logger = get_logger(__name__)


class PlatformAwareBuyer(Trader):
    """Platform-aware token buyer that works with any supported platform."""

    def __init__(
        self,
        client: SolanaClient,
        wallet: Wallet,
        priority_fee_manager: PriorityFeeManager,
        amount: float,
        slippage: float = 0.01,
        max_retries: int = 5,
        extreme_fast_token_amount: int = 0,
        extreme_fast_mode: bool = False,
        compute_units: dict | None = None,
    ):
        """Initialize platform-aware token buyer."""
        self.client = client
        self.wallet = wallet
        self.priority_fee_manager = priority_fee_manager
        self.amount = amount
        self.slippage = slippage
        self.max_retries = max_retries
        self.extreme_fast_mode = extreme_fast_mode
        self.extreme_fast_token_amount = extreme_fast_token_amount
        self.compute_units = compute_units or {}

    async def execute(self, token_info: TokenInfo) -> TradeResult:
        """Execute buy operation using platform-specific implementations."""
        try:
            # Get platform-specific implementations
            implementations = get_platform_implementations(
                token_info.platform, self.client
            )
            address_provider = implementations.address_provider
            instruction_builder = implementations.instruction_builder
            curve_manager = implementations.curve_manager

            # Convert amount to lamports
            amount_lamports = int(self.amount * LAMPORTS_PER_SOL)

            if self.extreme_fast_mode:
                # Skip the wait and directly calculate the amount
                token_amount = self.extreme_fast_token_amount
                token_price_sol = self.amount / token_amount if token_amount > 0 else 0
            else:
                # Get pool address based on platform using platform-agnostic method
                pool_address = self._get_pool_address(token_info, address_provider)

                # Regular behavior with RPC call
                token_price_sol = await curve_manager.calculate_price(pool_address)
                token_amount = (
                    self.amount / token_price_sol if token_price_sol > 0 else 0
                )

            logger.info(f"Token price computed on-chain: {token_price_sol:.8f} SOL")

            # Calculate minimum token amount with slippage
            minimum_token_amount = token_amount * (1 - self.slippage)
            minimum_token_swap_amount_raw = int(minimum_token_amount * 10**TOKEN_DECIMALS)

            logger.info(f"Amount to spend: {self.amount:.6f} SOL => expected token amount: {token_amount:.6f} tokens"
            f", slippage: {self.slippage:.2f} so expected minimum token amount: {minimum_token_amount:.6f} tokens")

            # Calculate maximum SOL to spend with slippage
            max_amount_lamports = amount_lamports
            # Build buy instructions using platform-specific builder
            instructions = await instruction_builder.build_buy_instruction(
                token_info,
                self.wallet.pubkey,
                max_amount_lamports,  # amount_in (SOL)
                minimum_token_swap_amount_raw,  # minimum_amount_out (tokens)
                address_provider,
            )

            # Get accounts for priority fee calculation
            priority_accounts = instruction_builder.get_required_accounts_for_buy(
                token_info, self.wallet.pubkey, address_provider
            )


            # Send transaction
            tx_signature = await self.client.build_and_send_transaction(
                instructions,
                self.wallet.keypair,
                skip_preflight=True,
                max_retries=self.max_retries,
                priority_fee=await self.priority_fee_manager.calculate_priority_fee(
                    priority_accounts
                ),
                compute_unit_limit=instruction_builder.get_buy_compute_unit_limit(
                    self._get_cu_override("buy", token_info.platform)
                ),
            )

            confirm_result = await self.client.confirm_transaction(tx_signature)
            logger.info(f"Confirm result is {confirm_result}")    
            balance_changes = None
            try:
                # Get transaction with full metadata for balance analysis
                tx = await self.client.get_transaction(tx_signature)
                if tx:
                    # Get instruction accounts for balance analysis
                    instruction_accounts = address_provider.get_buy_instruction_accounts(token_info, self.wallet.pubkey)
                    balance_changes = implementations.balance_analyzer.analyze_balance_changes(
                        tx, token_info, self.wallet.pubkey, instruction_accounts
                    )
                    logger.info(f"Transaction inspection resulted in {balance_changes}")
                else:
                    logger.info(f"Failed to analyze transaction balances for lack of transaction {tx_signature} : {tx}")
            except Exception as e:
                logger.warning(f"Failed to analyze transaction balances")
                logger.exception(e)

            result = TradeResult(
                success=confirm_result.success,
                platform=token_info.platform,
                tx_signature=str(tx_signature),
                transaction_fee_raw=balance_changes.transaction_fee_raw,
                token_swap_amount_raw=balance_changes.token_swap_amount_raw,
                sol_swap_amount_raw=balance_changes.sol_swap_amount_raw,
                platform_fee_raw=balance_changes.platform_fee_raw,
            )
            if not confirm_result.success:
                result.error_message = confirm_result.error_message or f"Transaction failed to confirm: {tx_signature}"

            return result

        except Exception as e:
            logger.exception("Buy operation failed")
            return TradeResult(
                success=False, platform=token_info.platform, error_message=str(e)
            )

    def _get_pool_address(
        self, token_info: TokenInfo, address_provider: AddressProvider
    ) -> Pubkey:
        """Get the pool/curve address for price calculations using platform-agnostic method."""
        # Try to get the address from token_info first, then derive if needed
        if token_info.platform == Platform.PUMP_FUN:
            if hasattr(token_info, "bonding_curve") and token_info.bonding_curve:
                return token_info.bonding_curve
        elif token_info.platform == Platform.LETS_BONK:
            if hasattr(token_info, "pool_state") and token_info.pool_state:
                return token_info.pool_state

        # Fallback to deriving the address using platform provider
        return address_provider.derive_pool_address(token_info.mint)

    def _get_cu_override(self, operation: str, platform: Platform) -> int | None:
        """Get compute unit override from configuration.
        
        Args:
            operation: "buy" or "sell"
            platform: Trading platform (unused - each config is platform-specific)
            
        Returns:
            CU override value if configured, None otherwise
        """
        if not self.compute_units:
            return None
            
        # Just check for operation override (buy/sell)
        return self.compute_units.get(operation)


class PlatformAwareSeller(Trader):
    """Platform-aware token seller that works with any supported platform."""

    def __init__(
        self,
        client: SolanaClient,
        wallet: Wallet,
        priority_fee_manager: PriorityFeeManager,
        slippage: float = 0.25,
        max_retries: int = 5,
        compute_units: dict | None = None,
    ):
        """Initialize platform-aware token seller."""
        self.client = client
        self.wallet = wallet
        self.priority_fee_manager = priority_fee_manager
        self.slippage = slippage
        self.max_retries = max_retries
        self.compute_units = compute_units or {}

    async def execute(self, token_info: TokenInfo, position: Position) -> TradeResult:
        """Execute sell operation using platform-specific implementations."""
        try:
            # Get platform-specific implementations
            implementations = get_platform_implementations(
                token_info.platform, self.client
            )
            address_provider = implementations.address_provider
            instruction_builder = implementations.instruction_builder
            curve_manager = implementations.curve_manager

            # Use token amount from position instead of RPC call
            token_balance = position.token_amount_raw
            token_balance_decimal = token_balance / 10**TOKEN_DECIMALS

            logger.info(f"Token balance: {token_balance_decimal}")

            # Get pool address and current price using platform-agnostic method
            pool_address = self._get_pool_address(token_info, address_provider)
            token_price_sol = await curve_manager.calculate_price(pool_address)

            logger.info(f"Price per Token: {token_price_sol:.8f} SOL")

            # Calculate expected SOL output
            expected_sol_output = token_balance_decimal * token_price_sol

            # Calculate minimum SOL output with slippage protection
            min_sol_output = int(
                (expected_sol_output * (1 - self.slippage)) * LAMPORTS_PER_SOL
            )

            logger.info(f"Selling {token_balance_decimal} tokens on {token_info.platform.value}")
            logger.info(f"Expected SOL output: {expected_sol_output:.8f} SOL")
            logger.info(
                f"Minimum SOL output (with {self.slippage * 100:.1f}% slippage): {min_sol_output / LAMPORTS_PER_SOL:.8f} SOL"
            )

            # Build sell instructions using platform-specific builder
            instructions = await instruction_builder.build_sell_instruction(
                token_info,
                self.wallet.pubkey,
                token_balance,  # amount_in (tokens)
                min_sol_output,  # minimum_amount_out (SOL)
                address_provider,
            )

            # Get accounts for priority fee calculation
            priority_accounts = instruction_builder.get_required_accounts_for_sell(
                token_info, self.wallet.pubkey, address_provider
            )

            # Send transaction
            tx_signature = await self.client.build_and_send_transaction(
                instructions,
                self.wallet.keypair,
                skip_preflight=True,
                max_retries=self.max_retries,
                priority_fee=await self.priority_fee_manager.calculate_priority_fee(
                    priority_accounts
                ),
                compute_unit_limit=instruction_builder.get_sell_compute_unit_limit(
                    self._get_cu_override("sell", token_info.platform)
                ),
            )

            confirm_result = await self.client.confirm_transaction(tx_signature)

            balance_changes = None
            try:
                # Get transaction with full metadata for balance analysis
                tx_with_meta = await self.client.get_transaction(tx_signature)
                if tx_with_meta:
                    # Get instruction accounts for balance analysis
                    instruction_accounts = address_provider.get_sell_instruction_accounts(token_info, self.wallet.pubkey)
                    balance_changes = implementations.balance_analyzer.analyze_balance_changes(
                        tx_with_meta, token_info, self.wallet.pubkey, instruction_accounts
                    )
                    logger.info(f"Balance changes ={balance_changes}")
                else:
                    logger.info(f"Failed to analyze transaction balances for lack of transaction {tx_signature} : {tx_with_meta}")
            except Exception as e:
                logger.warning(f"Failed to analyze transaction balances: {e}")

            result = TradeResult(
                success=confirm_result.success,
                platform=token_info.platform,
                tx_signature=str(tx_signature),
                transaction_fee_raw=balance_changes.transaction_fee_raw,
                token_swap_amount_raw=balance_changes.token_swap_amount_raw,
                sol_swap_amount_raw=balance_changes.sol_amount_raw,
                platform_fee_raw=balance_changes.platform_fee_raw,
            )
            if not confirm_result.success:
                result.error_message = confirm_result.error_message or f"Transaction failed to confirm: {tx_signature}"

            return result

        except Exception as e:
            logger.exception("Sell operation failed")
            logger.exception(e)
            return TradeResult(
                success=False, platform=token_info.platform, error_message=str(e)
            )

    def _get_pool_address(
        self, token_info: TokenInfo, address_provider: AddressProvider
    ) -> Pubkey:
        """Get the pool/curve address for price calculations using platform-agnostic method."""
        # Try to get the address from token_info first, then derive if needed
        if token_info.platform == Platform.PUMP_FUN:
            if hasattr(token_info, "bonding_curve") and token_info.bonding_curve:
                return token_info.bonding_curve
        elif token_info.platform == Platform.LETS_BONK:
            if hasattr(token_info, "pool_state") and token_info.pool_state:
                return token_info.pool_state

        # Fallback to deriving the address using platform provider
        return address_provider.derive_pool_address(token_info.mint)

    def _get_cu_override(self, operation: str, platform: Platform) -> int | None:
        """Get compute unit override from configuration.
        
        Args:
            operation: "buy" or "sell"
            platform: Trading platform (unused - each config is platform-specific)
            
        Returns:
            CU override value if configured, None otherwise
        """
        if not self.compute_units:
            return None
            
        # Just check for operation override (buy/sell)
        return self.compute_units.get(operation)