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
    
    async def _execute_transaction(self, order: BuyOrder) -> BuyOrder:
        """Override to simulate instead of actually sending transaction."""
        # Simulate network latency
        logger.info(f"Simulating buy transaction (wait: {self.dry_run_wait_time}s)")
        await asyncio.sleep(self.dry_run_wait_time)
        
        # Generate fake signature
        order.tx_signature = f"DRYRUN_BUY_{order.token_info.mint}_{int(time()*1000)}"
        
        # Calculate fees
        order.transaction_fee_raw = 5000 + int((order.compute_unit_limit * order.priority_fee) / 1_000_000)
        order.platform_fee_raw = int(order.sol_amount_raw * 0.008)  # 0.8% estimate
        
        logger.info(f"Buy transaction simulated: {order.tx_signature}")
        return order
    
    async def _confirm_transaction(self, tx_signature: str):
        """Override to simulate transaction confirmation."""
        logger.info(f"Simulating transaction confirmation: {tx_signature}")
        await asyncio.sleep(self.dry_run_wait_time)
        
        # Return a mock confirmation result
        from core.client import SolanaClient

        return SolanaClient.ConfirmationResult(
            success=True,
            tx=tx_signature,
            error_message=None,
            block_ts=int(time() * 1000),  # Current time for dry run
        )
    
    async def _analyze_balance_changes(self, order: BuyOrder):
        """Override to simulate balance analysis for dry-run."""
        
        # Create a mock balance change result
        from platforms.pumpfun.balance_analyzer import BalanceChangeResult

        sol_swap_amount_raw = await self.curve_manager.calculate_sell_amount_out(pool_address=self._get_pool_address(order.token_info,None), amount_in=order.token_amount_raw)

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
        order.platform_fee_raw = int(order.minimum_sol_swap_amount_raw * 0.008)  # 0.8% estimate
        
        logger.info(f"Sell transaction simulated: {order.tx_signature}")
        return order
    
    async def _confirm_transaction(self, tx_signature: str):
        """Override to simulate transaction confirmation."""
        logger.info(f"Simulating transaction confirmation: {tx_signature} by sleeping for {self.dry_run_wait_time} seconds")
        await asyncio.sleep(self.dry_run_wait_time)
        logger.info(f"Simulating transaction confirmation: {tx_signature} - proceding")
        
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
        logger.info(f"Simulating balance analysis for sell order")
        
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