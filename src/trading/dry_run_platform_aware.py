"""Dry-run implementations that override only transaction execution."""

import asyncio
from time import time
from core.pubkeys import LAMPORTS_PER_SOL, TOKEN_ACCOUNT_RENT_EXEMPT_RESERVE, TOKEN_DECIMALS
from interfaces.core import CurveManager
from platforms import Platform
from platforms.pumpfun import curve_manager
from trading.platform_aware import PlatformAwareBuyer, PlatformAwareSeller
from trading.trade_order import BuyOrder, Order, OrderState, SellOrder
from trading.position import Position
from utils.logger import get_logger
from core.client import SolanaClient
from platforms.pumpfun.balance_analyzer import BalanceChangeResult

logger = get_logger(__name__)


class DryRunPlatformAwareBuyer(PlatformAwareBuyer):
    """Dry-run buyer that simulates execution without blockchain calls."""
    
    def __init__(self, *args, curve_manager: CurveManager, dry_run_wait_time: float = 0.5, **kwargs):
        # Remove database_manager from kwargs since PlatformAwareBuyer doesn't accept it
        kwargs.pop('database_manager', None)
        super().__init__(*args, **kwargs)
        self.dry_run_wait_time = dry_run_wait_time
        self.curve_manager = curve_manager
        self.DATA_PROPAGATION_SLEEP_TIME = 3.0
    
    async def _execute_transaction(self, order: BuyOrder) -> BuyOrder:
        """Override to simulate instead of actually sending transaction."""

        # Generate fake signature for successful transaction
        order.tx_signature = f"DRYRUN_BUY_{order.token_info.mint}_{int(time()*1000)}"
                            
        return order
    
    async def _confirm_transaction(self, position: Position) -> SolanaClient.ConfirmationResult:
        """Override to simulate transaction confirmation."""
        order = position.buy_order

        # Simulate network latency
        logger.info(f"[{str(order.token_info.mint)[:8]}] DRY RUN BUY: Simulating buy transaction (wait: {self.dry_run_wait_time}s)")
        await asyncio.sleep(self.dry_run_wait_time)

        # Simulate slippage validation - calculate actual SOL cost for fixed token amount
        net_sol_swapped_raw = None
        while net_sol_swapped_raw is None:
            try:
                # Calculate actual SOL cost for the fixed token amount using curve manager
                pool_address = self._get_pool_address(order.token_info, None)
                net_sol_swapped_raw = - await self.curve_manager.calculate_buy_amount_out(
                    mint=order.token_info.mint,
                    pool_address=pool_address, 
                    amount_in=order.token_amount_raw
                )
                #This is incomplete, it's doesn't contain the fees yet: see next step
                order.sol_amount_raw=net_sol_swapped_raw
                
                                
                # Calculate actual price based on order's token amount and actual SOL cost
                trade_price_sol_per_token = abs((net_sol_swapped_raw / 1_000_000_000) / (order.token_amount_raw / (10 ** TOKEN_DECIMALS)))
                order.token_price_sol = trade_price_sol_per_token
                # logger.info(f"[{str(buy_order.token_info.mint)[:8]}] DRY RUN BUY:  Amount of token swapped would be {buy_order.token_amount_raw} ({buy_order.token_amount_raw/10**TOKEN_DECIMALS} tokens) Net SOL swapped {net_sol_swapped_raw} ({net_sol_swapped_raw/1_000_000_000} SOL), trade_price_sol_per_token={trade_price_sol_per_token} SOL")

            except Exception:
                logger.exception(f"[{str(order.token_info.mint)[:8]}] DRY RUN BUY: Could not retrieve SOL amount swapped for {str(order.token_info.mint)}, account isn't propagated yet. Sleep for {self.DATA_PROPAGATION_SLEEP_TIME}s and retrying")
                await asyncio.sleep(self.DATA_PROPAGATION_SLEEP_TIME)

        order.block_ts=int(time()*1000)

        # Check if actual SOL cost exceeds slippage tolerance
        if -net_sol_swapped_raw > order.max_sol_amount_raw:
            # Simulate slippage failure - still charge transaction fees
            logger.info(f"[{str(order.token_info.mint)[:8]}] DRY RUN BUY: Simulating slippage failure - expected max {order.max_sol_amount_raw / LAMPORTS_PER_SOL} SOL, actual cost {-net_sol_swapped_raw / LAMPORTS_PER_SOL} SOL")
            order.tx_signature = f"DRYRUN_BUY_FAILED_{order.token_info.mint}_{int(time()*1000)}"
            order.slippage_failed = True  # Add flag to indicate slippage failure
        else:
            logger.info(f"[{str(order.token_info.mint)[:8]}] DRY RUN BUY: Slippage check passed - actual cost {-net_sol_swapped_raw / LAMPORTS_PER_SOL} SOL (max allowed: {order.max_sol_amount_raw / LAMPORTS_PER_SOL} SOL)")
        
        return SolanaClient.ConfirmationResult(
            success=not order.slippage_failed,
            tx=order.tx_signature,
            block_ts=order.block_ts,
            error_message=f"Slippage tolerance exceeded" if order.slippage_failed else None,
        )
    
    async def _analyze_balance_changes(self, position: Position):
        """Override to simulate balance analysis for dry-run."""
        order = position.buy_order
        
        # Create a mock balance change result
        from platforms.pumpfun.balance_analyzer import BalanceChangeResult
#        logger.info(f"Analyzing balance changes for buy order {buy_order}")

        order.transaction_fee_raw = 5000 + int((order.compute_unit_limit * order.priority_fee) / 1_000_000)

        net_sol_swapped_raw=0
        rent_exemption_amount_raw=0
        is_slippage_failure = order.tx_signature.startswith("DRYRUN_BUY_FAILED_")
        if is_slippage_failure:
            order.protocol_fee_raw=0
            order.creator_fee_raw=0
            tip_fee_raw=0
            order.token_amount_raw=0
        else:        
            net_sol_swapped_raw=order.sol_amount_raw

            # Get platform fee percentage from curve manager
            platform_constants = await self.curve_manager.get_platform_constants()
            protocol_fee_percentage = 0 if is_slippage_failure else float(platform_constants.get("fee_percentage", 0.95))/100 
            creator_fee_percentage = 0 if is_slippage_failure else float(platform_constants.get("creator_fee_percentage", 0.3))/100 
            # Still charge transaction fees even on slippage failure
            order.protocol_fee_raw = -int(net_sol_swapped_raw * protocol_fee_percentage)
            order.creator_fee_raw = -int(net_sol_swapped_raw * creator_fee_percentage)
            
            tip_fee_raw = self.client.tip_amount_lamports if self.client.send_method == "helius_sender" else 0
            rent_exemption_amount_raw=-TOKEN_ACCOUNT_RENT_EXEMPT_RESERVE

        # minus the fees because sol_amount_raw is negative
        order.sol_amount_raw = (
            rent_exemption_amount_raw
            +net_sol_swapped_raw
            -order.protocol_fee_raw
            -order.creator_fee_raw
            -order.transaction_fee_raw
            -tip_fee_raw
        )

        result = BalanceChangeResult(
            token_swap_amount_raw=order.token_amount_raw,
            net_sol_swap_amount_raw=net_sol_swapped_raw,
            protocol_fee_raw=order.protocol_fee_raw,
            creator_fee_raw=order.creator_fee_raw,
            transaction_fee_raw=order.transaction_fee_raw,
            tip_fee_raw=tip_fee_raw,
            rent_exemption_amount_raw=rent_exemption_amount_raw,
            unattributed_sol_amount_raw=0,
            sol_amount_raw=order.sol_amount_raw,
        )
        
        # Record simulated trade for price tracking
        if not is_slippage_failure:
            self.curve_manager.record_simulated_trade(
                mint=order.token_info.mint,
                sol_swap_raw=net_sol_swapped_raw,  # negative for buy
                token_swap_raw=order.token_amount_raw,  # positive for buy
                timestamp=time(),
            )
        
        return result

class DryRunPlatformAwareSeller(PlatformAwareSeller):
    """Dry-run seller that simulates execution without blockchain calls."""
    
    def __init__(self, *args, curve_manager: CurveManager, dry_run_wait_time: float = 0.5, **kwargs):
        # Remove database_manager from kwargs since PlatformAwareBuyer doesn't accept it
        kwargs.pop('database_manager', None)
        super().__init__(*args, **kwargs)
        self.dry_run_wait_time = dry_run_wait_time
        self.curve_manager = curve_manager
    
    async def _execute_transaction(self, order: SellOrder) -> SellOrder:
        """Override to simulate instead of actually sending transaction."""
        # Generate fake signature
        order.tx_signature = f"DRYRUN_SELL_{order.token_info.mint}_{int(time()*1000)}"
        return order
    
    async def _confirm_transaction(self, position: Position) -> SolanaClient.ConfirmationResult:
        """Override to simulate transaction confirmation."""
        order = position.sell_order
        
        # Simulate network latency
        logger.info(f"[{str(order.token_info.mint)[:8]}] DRY RUN SELL: Simulating sell transaction (wait: {self.dry_run_wait_time}s)")
        await asyncio.sleep(self.dry_run_wait_time)


        if position.buy_order.state != OrderState.CONFIRMED:
            logger.info(f"[{str(order.token_info.mint)[:8]}] DRY RUN SELL: Simulating sell transaction failure because buy order is not confirmed so we can't sell")
            order.tx_signature = f"DRYRUN_SELL_FAILED_{order.token_info.mint}_{int(time()*1000)}"
            return SolanaClient.ConfirmationResult(
                success=False,
                tx=order.tx_signature,
                error_message="No such token account (=buy order is not confirmed)",
                block_ts=None,
            )

        order.block_ts=int(time()*1000)
        
        net_sol_swap_raw = await self.curve_manager.calculate_sell_amount_out(
            mint=order.token_info.mint,
            pool_address=self._get_pool_address(order.token_info,None), 
            amount_in=order.token_amount_raw
            )
        #This is incomplete, it's doesn't contain the fees yet
        order.expected_sol_swap_amount_raw=net_sol_swap_raw

        trade_price_sol_per_token = abs((net_sol_swap_raw / 1_000_000_000) / (order.token_amount_raw / (10 ** TOKEN_DECIMALS)))
        order.token_price_sol = trade_price_sol_per_token

        # logger.info(f"[{str(sell_order.token_info.mint)[:8]}] DRY RUN SELL: Amount of token swapped would be  {sell_order.token_amount_raw} ({sell_order.token_amount_raw/10**TOKEN_DECIMALS} tokens) Net SOL swapped {net_sol_swap_raw} ({net_sol_swap_raw/1_000_000_000} SOL), trade_price_sol_per_token={trade_price_sol_per_token} SOL")
        
        return SolanaClient.ConfirmationResult(
            success=True,
            tx=order.tx_signature,
            error_message=None,
            block_ts=order.block_ts,  # Current time for dry run
        )
    
    async def _analyze_balance_changes(self, position: Position):
        """Override to simulate balance analysis for dry-run."""
        order = position.sell_order
        
        net_sol_swap_amount_raw = 0
        rent_exemption_amount_raw = 0
        tip_fee_raw = 0
        is_sell_failure = order.tx_signature.startswith("DRYRUN_SELL_FAILED_")
        if is_sell_failure:
            order.token_amount_raw=0
            order.protocol_fee_raw=0
            order.creator_fee_raw=0
            order.tip_fee_raw=0
            order.rent_exemption_amount_raw=0
            order.unattributed_sol_amount_raw=0
        else: # not a failure
            net_sol_swap_amount_raw = order.expected_sol_swap_amount_raw
            rent_exemption_amount_raw = TOKEN_ACCOUNT_RENT_EXEMPT_RESERVE

            # Get platform fee percentage from curve manager
            platform_constants = await self.curve_manager.get_platform_constants()
            protocol_fee_percentage = float(platform_constants.get("fee_percentage", 0.95))/100 
            creator_fee_percentage = float(platform_constants.get("creator_fee_percentage", 0.3))/100 
            order.protocol_fee_raw = int(net_sol_swap_amount_raw * protocol_fee_percentage)
            order.creator_fee_raw = int(net_sol_swap_amount_raw * creator_fee_percentage)
            tip_fee_raw = self.client.tip_amount_lamports if self.client.send_method == "helius_sender" else 0

        order.transaction_fee_raw = 5000 + int((order.compute_unit_limit * order.priority_fee) / 1_000_000)
        sol_amount_raw=(
            rent_exemption_amount_raw
            +net_sol_swap_amount_raw
            -order.protocol_fee_raw
            -order.creator_fee_raw
            -order.transaction_fee_raw
            -tip_fee_raw
        )
        order.sol_amount_raw=sol_amount_raw

        result = BalanceChangeResult(
            token_swap_amount_raw=-order.token_amount_raw,
            net_sol_swap_amount_raw=net_sol_swap_amount_raw,
            transaction_fee_raw=order.transaction_fee_raw,
            protocol_fee_raw=order.protocol_fee_raw,
            creator_fee_raw=order.creator_fee_raw,
            tip_fee_raw=tip_fee_raw,
            rent_exemption_amount_raw=rent_exemption_amount_raw,
            unattributed_sol_amount_raw=0,
            sol_amount_raw=sol_amount_raw,
        )
        
        if not is_sell_failure:
            # Record simulated trade for price tracking
            self.curve_manager.record_simulated_trade(
                mint=order.token_info.mint,
                sol_swap_raw=net_sol_swap_amount_raw,  # positive for sell
                token_swap_raw=-order.token_amount_raw,  # negative for sell
                timestamp=time(),
            )
        
        return result