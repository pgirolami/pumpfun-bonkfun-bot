"""
Platform-aware trader implementations that use the interface system.
Final cleanup removing all platform-specific hardcoding.
"""

import time
from solders.pubkey import Pubkey

from core.client import SolanaClient
from core.priority_fee.manager import PriorityFeeManager
from core.pubkeys import LAMPORTS_PER_SOL, TOKEN_DECIMALS
from core.wallet import Wallet
from interfaces.core import AddressProvider, BalanceChangeResult, Platform, TokenInfo
from platforms import get_platform_implementations
from platforms.pumpfun.address_provider import PumpFunAddressProvider
from trading.base import Trader, TradeResult
from trading.position import Position
from trading.trade_order import BuyOrder, OrderState, SellOrder
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
        compute_units: dict | None = None,
    ):
        """Initialize platform-aware token buyer."""
        self.client = client
        self.wallet = wallet
        self.priority_fee_manager = priority_fee_manager
        self.amount = amount
        self.slippage = slippage
        self.max_retries = max_retries
        self.compute_units = compute_units or {}

    async def _prepare_buy_order(self, token_info: TokenInfo) -> BuyOrder:
        """Prepare buy order by calculating all parameters (shared logic)."""
        implementations = get_platform_implementations(token_info.platform, self.client)
        address_provider = implementations.address_provider
        curve_manager = implementations.curve_manager
        instruction_builder = implementations.instruction_builder

        # Create order with input
        order = BuyOrder(token_info=token_info, sol_amount_raw=int(self.amount * LAMPORTS_PER_SOL))

        # Calculate price and token amount using platform constants
        platform_constants = await curve_manager.get_platform_constants()
        starting_price_sol = platform_constants["starting_price_sol"]
        
        # Calculate token amount based on buy amount and starting price
        order.token_amount_raw = int((self.amount / starting_price_sol) * 10**TOKEN_DECIMALS)
        order.token_price_sol = starting_price_sol

        # logger.info(f"Token price computed on-chain: {order.token_price_sol} SOL")

        # Calculate minimum with slippage (for Let's Bonk)
        order.minimum_token_swap_amount_raw = int(order.token_amount_raw * (1 - self.slippage))
        
        # Calculate maximum SOL cost with slippage tolerance (for PumpFun)
        order.max_sol_amount_raw = int(order.sol_amount_raw * (1 + self.slippage))

        # logger.info(f"Amount to spend: {self.amount:.6f} SOL => expected token amount: {order.token_amount_raw / 10**TOKEN_DECIMALS:.6f} tokens"
        # f", slippage: {self.slippage:.2f} so expected minimum token amount: {order.minimum_token_swap_amount_raw / 10**TOKEN_DECIMALS:.6f} tokens")

        # Get compute units and priority fee
        order.compute_unit_limit = instruction_builder.get_buy_compute_unit_limit(
            self._get_cu_override("buy", token_info.platform)
        )
        priority_accounts = instruction_builder.get_required_accounts_for_buy(
            token_info, self.wallet.pubkey, address_provider
        )
        order.priority_fee = await self.priority_fee_manager.calculate_priority_fee(priority_accounts)
        order.account_data_size_limit = self._get_cu_override(
             "account_data_size_buy", token_info.platform
        )

        return order

    async def _execute_transaction(self, order: BuyOrder) -> BuyOrder:
        """Execute the transaction (overridden in dry-run)."""
        implementations = get_platform_implementations(order.token_info.platform, self.client)
        instruction_builder = implementations.instruction_builder
        address_provider = implementations.address_provider

        # Build buy instructions using platform-specific builder
        instructions = await instruction_builder.build_buy_instruction(
            order.token_info,
            self.wallet.pubkey,
            order.sol_amount_raw,  # amount_in (SOL)
            order.minimum_token_swap_amount_raw,  # minimum_amount_out (tokens)
            address_provider,
        )

        # Send transaction
        order.tx_signature = await self.client.build_and_send_transaction(
            instructions,
            self.wallet.keypair,
            skip_preflight=True,
            max_retries=self.max_retries,
            priority_fee=order.priority_fee,
            compute_unit_limit=order.compute_unit_limit,
            account_data_size_limit=order.account_data_size_limit,
        )

        return order

    async def _confirm_transaction(self, order: BuyOrder) -> SolanaClient.ConfirmationResult:
        """Confirm transaction (overridden in dry-run)."""
        return await self.client.confirm_transaction(order.tx_signature)

    async def _analyze_balance_changes(self, order: BuyOrder) -> BalanceChangeResult | None:
        """Analyze balance changes (overridden in dry-run)."""
        # Get transaction with full metadata for balance analysis
        tx = await self.client.get_transaction(order.tx_signature)
        if tx:
            # Get instruction accounts for balance analysis
            implementations = get_platform_implementations(order.token_info.platform, self.client)
            instruction_accounts = implementations.address_provider.get_buy_instruction_accounts(order.token_info, self.wallet.pubkey)
            
            # Extract user_token_account from transaction instruction (not from ATA derivation)
            # This works even if the account was created with CreateAccountWithSeed
            if order.token_info.platform == Platform.PUMP_FUN:
                address_provider = PumpFunAddressProvider()
                user_token_account = address_provider.extract_user_token_account_from_transaction(tx)
                if user_token_account:
                    instruction_accounts["user_token_account"] = user_token_account
                else:
                    logger.warning(
                        f"Could not extract user_token_account from transaction {order.tx_signature[:8]}..., "
                        f"token balance analysis may fail"
                    )
            
            return implementations.balance_analyzer.analyze_balance_changes(
                tx, order.token_info, self.wallet.pubkey, instruction_accounts
            )
        else:
            logger.info(f"Failed to analyze transaction balances for lack of transaction {order.tx_signature} : {tx}")
            return None

    async def prepare_and_send_order(self, token_info: TokenInfo) -> BuyOrder:
        """Prepare and send buy order without waiting for confirmation.
        
        Args:
            token_info: Token information for the buy order
            
        Returns:
            BuyOrder with state set to SENT
        """
        # Prepare order (shared with dry-run)
        order = await self._prepare_buy_order(token_info)

        # Record start time before sending transaction
        order.trade_start_time = time.time()

        # Execute transaction (overridden in dry-run)
        order = await self._execute_transaction(order)
        
        # Set state to SENT after transaction is sent
        order.state = OrderState.SENT
        
        return order

    async def process_order(self, buy_order: BuyOrder) -> TradeResult:
        """Confirm and process buy order, returning trade result.
        
        Args:
            buy_order: BuyOrder that was previously sent (state should be SENT)
            
        Returns:
            TradeResult with confirmation status and balance changes
        """
        try:
            # Confirm and analyze
            confirm_result = await self._confirm_transaction(buy_order)
            logger.debug(f"Confirm result is {confirm_result}")

            balance_changes = None
            try:
                balance_changes = await self._analyze_balance_changes(buy_order)
                logger.debug(f"[{str(buy_order.token_info.mint)[:8]}] Balance analysis resulted in {balance_changes}")
            except Exception as e:
                logger.exception(f"[{str(buy_order.token_info.mint)[:8]}] Failed to analyze transaction balances")

            # Calculate trade duration
            trade_duration_ms = int((time.time() - buy_order.trade_start_time) * 1000)
            time_to_block_ms = int((confirm_result.block_ts - buy_order.trade_start_time) * 1000) if confirm_result.block_ts else None

            # Update order state based on confirmation result
            if confirm_result.success:
                buy_order.state = OrderState.CONFIRMED
                buy_order.block_ts = confirm_result.block_ts
            else:
                buy_order.state = OrderState.FAILED

            if balance_changes:
                buy_order.transaction_fee_raw = balance_changes.transaction_fee_raw
                buy_order.protocol_fee_raw = balance_changes.protocol_fee_raw
                buy_order.creator_fee_raw = balance_changes.creator_fee_raw

            result = TradeResult(
                success=confirm_result.success,
                platform=buy_order.token_info.platform,
                tx_signature=str(buy_order.tx_signature),
                block_time=confirm_result.block_ts,
                transaction_fee_raw=balance_changes.transaction_fee_raw if balance_changes else None,
                token_swap_amount_raw=balance_changes.token_swap_amount_raw if balance_changes else None,
                net_sol_swap_amount_raw=balance_changes.net_sol_swap_amount_raw if balance_changes else None,
                platform_fee_raw=balance_changes.protocol_fee_raw + balance_changes.creator_fee_raw if balance_changes else None,
                tip_fee_raw=balance_changes.tip_fee_raw if balance_changes else None,
                rent_exemption_amount_raw=balance_changes.rent_exemption_amount_raw if balance_changes else None,
                unattributed_sol_amount_raw=balance_changes.unattributed_sol_amount_raw if balance_changes else None,
                trade_duration_ms=trade_duration_ms,
                time_to_block_ms=time_to_block_ms,
                sol_swap_amount_raw=balance_changes.sol_amount_raw if balance_changes else None,
            )
            if not confirm_result.success:
                result.error_message = confirm_result.error_message or f"Transaction failed to confirm: {buy_order.tx_signature}"

            # logger.info(f"Buy trade completed in {trade_duration_ms}ms")
            return result

        except Exception as e:
            trade_duration_ms = int((time.time() - buy_order.trade_start_time) * 1000)
            logger.exception("Buy order processing failed, trade_duration_ms={trade_duration_ms}")
            
            # Set order state to FAILED on exception
            buy_order.state = OrderState.FAILED
            
            return TradeResult(
                block_time=None,
                success=False, 
                platform=buy_order.token_info.platform, 
                error_message=str(e),
                trade_duration_ms=trade_duration_ms,
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

    async def _prepare_sell_order(self, token_info: TokenInfo, position: Position) -> SellOrder:
        """Prepare sell order by calculating all parameters."""
        implementations = get_platform_implementations(token_info.platform, self.client)
        address_provider = implementations.address_provider
        curve_manager = implementations.curve_manager
        instruction_builder = implementations.instruction_builder

        # Create order with input
        order = SellOrder(token_info=token_info, token_amount_raw=position.get_current_token_balance_raw())

        # Calculate decimal amount for logging
        token_balance_decimal = order.token_amount_raw / 10**TOKEN_DECIMALS
        # logger.info(f"Token balance: {token_balance_decimal}")

        order.token_price_sol = await curve_manager.calculate_price(order.token_info.mint, token_info.bonding_curve)

        # logger.info(f"Price per Token: {order.token_price_sol} SOL")

        # Calculate expected SOL output
        expected_sol_output = token_balance_decimal * order.token_price_sol
        order.expected_sol_swap_amount_raw = int(expected_sol_output * LAMPORTS_PER_SOL)

        # Calculate minimum SOL output with slippage protection
        order.minimum_sol_swap_amount_raw = int(
            (expected_sol_output * (1 - self.slippage)) * LAMPORTS_PER_SOL
        )

        # logger.info(f"Selling {token_balance_decimal} tokens on {token_info.platform.value}")
        # logger.info(f"Expected SOL output: {expected_sol_output} SOL")
        # logger.info(
        #     f"Minimum SOL output (with {self.slippage * 100:.1f}% slippage): {order.minimum_sol_swap_amount_raw / LAMPORTS_PER_SOL} SOL"
        # )

        # Get compute units and priority fee
        order.compute_unit_limit = instruction_builder.get_sell_compute_unit_limit(
            self._get_cu_override("sell", token_info.platform)
        )
        priority_accounts = instruction_builder.get_required_accounts_for_sell(
            token_info, self.wallet.pubkey, address_provider
        )
        order.priority_fee = await self.priority_fee_manager.calculate_priority_fee(priority_accounts)
        order.account_data_size_limit = self._get_cu_override(
             "account_data_size_sell", token_info.platform
        )

        return order

    async def _execute_transaction(self, order: SellOrder) -> SellOrder:
        """Execute the transaction (overridden in dry-run)."""
        implementations = get_platform_implementations(order.token_info.platform, self.client)
        instruction_builder = implementations.instruction_builder
        address_provider = implementations.address_provider

        # Build sell instructions using platform-specific builder
        instructions = await instruction_builder.build_sell_instruction(
            order.token_info,
            self.wallet.pubkey,
            order.token_amount_raw,  # amount_in (tokens)
            order.minimum_sol_swap_amount_raw,  # minimum_amount_out (SOL)
            address_provider,
        )

        # Send transaction
        order.tx_signature = await self.client.build_and_send_transaction(
            instructions,
            self.wallet.keypair,
            skip_preflight=True,
            max_retries=self.max_retries,
            priority_fee=order.priority_fee,
            compute_unit_limit=order.compute_unit_limit,
            account_data_size_limit=order.account_data_size_limit,
        )

        return order

    async def _confirm_transaction(self, order: SellOrder) -> SolanaClient.ConfirmationResult:
        """Confirm transaction (overridden in dry-run)."""
        return await self.client.confirm_transaction(order.tx_signature)

    async def _analyze_balance_changes(self, order: SellOrder) -> BalanceChangeResult | None:
        """Analyze balance changes (overridden in dry-run)."""
        # Get transaction with full metadata for balance analysis
        tx_with_meta = await self.client.get_transaction(order.tx_signature)
        if tx_with_meta:
            # Get instruction accounts for balance analysis
            implementations = get_platform_implementations(order.token_info.platform, self.client)
            instruction_accounts = implementations.address_provider.get_sell_instruction_accounts(order.token_info, self.wallet.pubkey)
            
            # Extract user_token_account from transaction instruction (not from ATA derivation)
            # This works even if the account was created with CreateAccountWithSeed
            if order.token_info.platform == Platform.PUMP_FUN:
                address_provider = PumpFunAddressProvider()
                user_token_account = address_provider.extract_user_token_account_from_transaction(tx_with_meta)
                if user_token_account:
                    instruction_accounts["user_token_account"] = user_token_account
                else:
                    logger.warning(
                        f"Could not extract user_token_account from transaction {order.tx_signature[:8]}..., "
                        f"token balance analysis may fail"
                    )
            
            return implementations.balance_analyzer.analyze_balance_changes(
                tx_with_meta, order.token_info, self.wallet.pubkey, instruction_accounts
            )
        else:
            logger.info(f"Failed to analyze transaction balances for lack of transaction {order.tx_signature} : {tx_with_meta}")
            return None

    async def prepare_and_send_order(self, token_info: TokenInfo, position: Position) -> SellOrder:
        """Prepare and send sell order without waiting for confirmation.
        
        Args:
            token_info: Token information for the sell order
            position: Position to sell from
            
        Returns:
            SellOrder with state set to SENT
        """
        # Prepare order (shared with dry-run)
        order = await self._prepare_sell_order(token_info, position)

        # Record start time before sending transaction
        order.trade_start_time = time.time()

        # Execute transaction (overridden in dry-run)
        order = await self._execute_transaction(order)
        
        # Set state to SENT after transaction is sent
        order.state = OrderState.SENT
        
        return order

    async def process_order(self, sell_order: SellOrder) -> TradeResult:
        """Confirm and process sell order, returning trade result.
        
        Args:
            sell_order: SellOrder that was previously sent (state should be SENT)
            
        Returns:
            TradeResult with confirmation status and balance changes
        """
        try:
            # Confirm and analyze
            confirm_result = await self._confirm_transaction(sell_order)

            balance_changes = None
            try:
                balance_changes = await self._analyze_balance_changes(sell_order)
                logger.debug(f"[{str(sell_order.token_info.mint)[:8]}] Balance analysis resulted in {balance_changes}")
            except Exception as e:
                logger.exception(f"[{str(sell_order.token_info.mint)[:8]}] Failed to analyze transaction balances")

            # Calculate trade duration
            trade_duration_ms = int((time.time() - sell_order.trade_start_time) * 1000)
            time_to_block_ms = int((confirm_result.block_ts - sell_order.trade_start_time) * 1000) if confirm_result.block_ts else None

            # Update order state based on confirmation result
            if confirm_result.success:
                sell_order.state = OrderState.CONFIRMED
                sell_order.block_ts = confirm_result.block_ts
                if balance_changes:
                    sell_order.transaction_fee_raw = balance_changes.transaction_fee_raw
                    sell_order.protocol_fee_raw = balance_changes.protocol_fee_raw
                    sell_order.creator_fee_raw = balance_changes.creator_fee_raw
            else:
                sell_order.state = OrderState.FAILED

            result = TradeResult(
                success=confirm_result.success,
                platform=sell_order.token_info.platform,
                tx_signature=str(sell_order.tx_signature),
                block_time=confirm_result.block_ts,
                transaction_fee_raw=balance_changes.transaction_fee_raw if balance_changes else None,
                token_swap_amount_raw=balance_changes.token_swap_amount_raw if balance_changes else None,
                net_sol_swap_amount_raw=balance_changes.net_sol_swap_amount_raw if balance_changes else None,
                sol_swap_amount_raw=balance_changes.sol_amount_raw if balance_changes else None,
                platform_fee_raw=balance_changes.protocol_fee_raw + balance_changes.creator_fee_raw if balance_changes else None,
                tip_fee_raw=balance_changes.tip_fee_raw if balance_changes else None,
                rent_exemption_amount_raw=balance_changes.rent_exemption_amount_raw if balance_changes else None,
                unattributed_sol_amount_raw=balance_changes.unattributed_sol_amount_raw if balance_changes else None,
                trade_duration_ms=trade_duration_ms,
                time_to_block_ms=time_to_block_ms,
            )
            if not confirm_result.success:
                result.error_message = confirm_result.error_message or f"Transaction failed to confirm: {sell_order.tx_signature}"

            # logger.info(f"Sell trade completed in {trade_duration_ms}ms")
            return result

        except Exception as e:
            trade_duration_ms = int((time.time() - trade_start_time) * 1000)
            logger.exception("Sell order processing failed")
            logger.info(f"Failed sell trade took {trade_duration_ms}ms")
            
            # Set order state to FAILED on exception
            sell_order.state = OrderState.FAILED
            
            return TradeResult(
                block_time=None,
                success=False, 
                platform=sell_order.token_info.platform, 
                error_message=str(e),
                trade_duration_ms=trade_duration_ms,
            )

    async def execute(self, token_info: TokenInfo, position: Position) -> TradeResult:
        """Execute sell operation using the order pattern (legacy method for compatibility).
        
        This method is kept for backward compatibility but is now a wrapper around
        prepare_and_send_order() and process_order().
        """
        order = await self.prepare_and_send_order(token_info, position)
        return await self.process_order(order)

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
