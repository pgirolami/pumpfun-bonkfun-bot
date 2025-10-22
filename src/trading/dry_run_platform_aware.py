"""Dry-run implementations that override only transaction execution."""

import asyncio
from time import time
from core.pubkeys import LAMPORTS_PER_SOL, TOKEN_ACCOUNT_RENT_EXEMPT_RESERVE, TOKEN_DECIMALS
from interfaces.core import CurveManager
from platforms import Platform
from platforms.pumpfun import curve_manager
from solders.instruction import Instruction
from solders.pubkey import Pubkey
from trading.platform_aware import PlatformAwareBuyer, PlatformAwareSeller
from trading.trade_order import BuyOrder, Order, SellOrder
from utils.logger import get_logger

logger = get_logger(__name__)


class DryRunPlatformAwareBuyer(PlatformAwareBuyer):
    """Dry-run buyer that simulates execution without blockchain calls."""
    
    def __init__(self, *args, curve_manager: CurveManager, dry_run_wait_time: float = 0.5, **kwargs):
        # Remove database_manager from kwargs since PlatformAwareBuyer doesn't accept it
        kwargs.pop('database_manager', None)
        super().__init__(*args, **kwargs)
        self.dry_run_wait_time = dry_run_wait_time
        self.curve_manager = curve_manager
        self.PROPAGATION_SLEEP_TIME = 2.0
    
    async def _execute_transaction(self, order: BuyOrder) -> BuyOrder:
        """Override to simulate instead of actually sending transaction."""
        # Simulate network latency
        logger.info(f"Simulating buy transaction (wait: {self.dry_run_wait_time}s)")
        await asyncio.sleep(self.dry_run_wait_time)

        order.transaction_fee_raw = 5000 + int((order.compute_unit_limit * order.priority_fee) / 1_000_000)

        # Generate fake signature for successful transaction
        order.tx_signature = f"DRYRUN_BUY_{order.token_info.mint}_{int(time()*1000)}"

        # Simulate slippage validation - calculate actual SOL cost for fixed token amount
        actual_sol_cost_raw = None
        while not actual_sol_cost_raw:
            try:
                # Calculate actual SOL cost for the fixed token amount using curve manager
                pool_address = self._get_pool_address(order.token_info, None)
                actual_sol_cost_raw = await self.curve_manager.calculate_sell_amount_out(
                    pool_address=pool_address, 
                    amount_in=order.token_amount_raw
                )
                
                # Update the order with current price for accurate entry price calculation
                current_price = await self.curve_manager.calculate_price(pool_address)
                order.token_price_sol = current_price
            except Exception:
                logger.info(f"Could not retrieve SOL amount swapped, account isn't propagated yet. Sleep for {self.PROPAGATION_SLEEP_TIME}s and retrying")
                await asyncio.sleep(self.PROPAGATION_SLEEP_TIME)
        
        # Check if actual SOL cost exceeds slippage tolerance
        if actual_sol_cost_raw > order.max_sol_amount_raw:
            # Simulate slippage failure - still charge transaction fees
            from core.pubkeys import LAMPORTS_PER_SOL
            logger.warning(f"DRY RUN: Simulating slippage failure - expected max {order.max_sol_amount_raw / LAMPORTS_PER_SOL:.6f} SOL, actual cost {actual_sol_cost_raw / LAMPORTS_PER_SOL:.6f} SOL")
            order.tx_signature = f"DRYRUN_BUY_FAILED_{order.token_info.mint}_{int(time()*1000)}"
            order.slippage_failed = True  # Add flag to indicate slippage failure
            
            # Still charge transaction fees even on slippage failure
            order.transaction_fee_raw = 5000 + int((order.compute_unit_limit * order.priority_fee) / 1_000_000)
            order.platform_fee_raw = 0  # No platform fee since no tokens were acquired
            return order
        else:
            from core.pubkeys import LAMPORTS_PER_SOL
            logger.info(f"DRY RUN: Slippage check passed - actual cost {actual_sol_cost_raw / LAMPORTS_PER_SOL:.6f} SOL (max allowed: {order.max_sol_amount_raw / LAMPORTS_PER_SOL:.6f} SOL)")
                    
        
        
        # Get platform fee percentage from curve manager
        platform_constants = await self.curve_manager.get_platform_constants()
        platform_fee_percentage = float(platform_constants.get("fee_percentage", 0.95))/100  # Default to 0.8% if not available
        order.platform_fee_raw = int(order.sol_amount_raw * platform_fee_percentage)
        
        logger.info(f"Buy transaction simulated: {order.tx_signature}")
        return order
    
    async def _confirm_transaction(self, tx_signature: str):
        """Override to simulate transaction confirmation."""
        
        # Check if this was a slippage failure
        is_slippage_failure = tx_signature.startswith("DRYRUN_BUY_FAILED_")
        
        # Return a mock confirmation result
        from core.client import SolanaClient

        return SolanaClient.ConfirmationResult(
            success=not is_slippage_failure,
            tx=tx_signature,
            error_message=f"Slippage tolerance exceeded" if is_slippage_failure else None,
            block_ts=int(time() * 1000),  # Current time for dry run
        )
    
    async def _analyze_balance_changes(self, order: BuyOrder):
        """Override to simulate balance analysis for dry-run."""
        
        # Create a mock balance change result
        from platforms.pumpfun.balance_analyzer import BalanceChangeResult
#        logger.info(f"Analyzing balance changes for buy order {order}")

        sol_swap_amount_raw = None
        while not sol_swap_amount_raw:
            try:
                sol_swap_amount_raw = - await self.curve_manager.calculate_sell_amount_out(pool_address=self._get_pool_address(order.token_info,None), amount_in=order.token_amount_raw)
            except Exception:
                logger.info("Could not retrieve SOL amount swapped, account isn't propagated yet. Sleep for 2s and retrying")
                await asyncio.sleep(2.0)

        return BalanceChangeResult(
            token_swap_amount_raw=order.token_amount_raw,
            sol_amount_raw=sol_swap_amount_raw-(order.platform_fee_raw+order.transaction_fee_raw),
            platform_fee_raw=order.platform_fee_raw,
            transaction_fee_raw=order.transaction_fee_raw,
            rent_exemption_amount_raw=0,
            sol_swap_amount_raw=sol_swap_amount_raw,
        )

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
        # Simulate network latency
        logger.info(f"Simulating sell transaction (wait: {self.dry_run_wait_time}s)")
        await asyncio.sleep(self.dry_run_wait_time)
        
        # Generate fake signature
        order.tx_signature = f"DRYRUN_SELL_{order.token_info.mint}_{int(time()*1000)}"
        
        # Calculate fees
        order.transaction_fee_raw = 5000 + int((order.compute_unit_limit * order.priority_fee) / 1_000_000)
        # Get platform fee percentage from curve manager
        platform_constants = await self.curve_manager.get_platform_constants()
        platform_fee_percentage = float(platform_constants.get("fee_percentage", 0.95))/100  # Default to 0.8% if not available
        order.platform_fee_raw = int(order.minimum_sol_swap_amount_raw * platform_fee_percentage)
        
        logger.info(f"Sell transaction simulated: {order.tx_signature}")
        return order
    
    async def _confirm_transaction(self, tx_signature: str):
        """Override to simulate transaction confirmation."""
        
        # Return a mock confirmation result
        from core.client import SolanaClient
        from time import time
        return SolanaClient.ConfirmationResult(
            success=True,
            tx=tx_signature,
            error_message=None,
            block_ts=int(time() * 1000),  # Current time for dry run
        )
    
    async def _analyze_balance_changes(self, order: SellOrder):
        """Override to simulate balance analysis for dry-run."""
        # logger.info(f"Simulating balance analysis for sell order")
        
        # Create a mock balance change result
        from platforms.pumpfun.balance_analyzer import BalanceChangeResult
        
        sol_swap_amount_raw = await self.curve_manager.calculate_sell_amount_out(pool_address=self._get_pool_address(order.token_info,None), amount_in=order.token_amount_raw)

        return BalanceChangeResult(
            token_swap_amount_raw=-order.token_amount_raw,
            sol_amount_raw=sol_swap_amount_raw-(order.platform_fee_raw+order.transaction_fee_raw),
            platform_fee_raw=order.platform_fee_raw,
            transaction_fee_raw=order.transaction_fee_raw,
            rent_exemption_amount_raw=0,
            sol_swap_amount_raw=sol_swap_amount_raw,
        )