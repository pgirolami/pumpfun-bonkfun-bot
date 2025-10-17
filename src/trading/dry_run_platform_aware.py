"""Dry-run implementations that override only transaction execution."""

import asyncio
import time
from core.pubkeys import TOKEN_ACCOUNT_RENT_EXEMPT_RESERVE
from solders.instruction import Instruction
from trading.platform_aware import PlatformAwareBuyer, PlatformAwareSeller
from trading.trade_order import BuyOrder, SellOrder
from utils.logger import get_logger

logger = get_logger(__name__)


class DryRunPlatformAwareBuyer(PlatformAwareBuyer):
    """Dry-run buyer that simulates execution without blockchain calls."""
    
    def __init__(self, *args, dry_run_wait_time: float = 0.5, **kwargs):
        # Remove database_manager from kwargs since PlatformAwareBuyer doesn't accept it
        kwargs.pop('database_manager', None)
        super().__init__(*args, **kwargs)
        self.dry_run_wait_time = dry_run_wait_time
    
    async def _execute_transaction(self, order: BuyOrder) -> BuyOrder:
        """Override to simulate instead of actually sending transaction."""
        # Simulate network latency
        logger.info(f"Simulating buy transaction (wait: {self.dry_run_wait_time}s)")
        await asyncio.sleep(self.dry_run_wait_time)
        
        # Generate fake signature
        order.tx_signature = f"DRYRUN_BUY_{order.token_info.mint}_{int(time.time()*1000)}"
        
        # Calculate fees
        order.transaction_fee_raw = 5000 + int((order.compute_unit_limit * order.priority_fee) / 1_000_000)
        order.platform_fee_raw = int(order.sol_amount_raw * 0.008)  # 0.8% estimate
        
        logger.info(f"Buy transaction simulated: {order.tx_signature}")
        return order
    
    async def _confirm_transaction(self, tx_signature: str):
        """Override to simulate transaction confirmation."""
        logger.info(f"Simulating transaction confirmation: {tx_signature}")
        await asyncio.sleep(0.1)  # Brief simulation delay
        
        # Return a mock confirmation result
        from core.client import SolanaClient
        from time import time
        return SolanaClient.ConfirmationResult(
            success=True,
            tx=tx_signature,
            error_message=None,
            block_ts=int(time() * 1000),  # Current time for dry run
        )
    
    async def _analyze_balance_changes(self, order: BuyOrder):
        """Override to simulate balance analysis for dry-run."""
        logger.info(f"Simulating balance analysis for buy order")
        
        # Create a mock balance change result
        from platforms.pumpfun.balance_analyzer import BalanceChangeResult
        
        return BalanceChangeResult(
            token_swap_amount_raw=order.minimum_token_swap_amount_raw,
            sol_amount_raw=-(order.sol_amount_raw+TOKEN_ACCOUNT_RENT_EXEMPT_RESERVE),
            platform_fee_raw=order.platform_fee_raw,
            transaction_fee_raw=order.transaction_fee_raw,
            rent_exemption_amount_raw=TOKEN_ACCOUNT_RENT_EXEMPT_RESERVE,
            sol_swap_amount_raw=-order.sol_amount_raw,
        )


class DryRunPlatformAwareSeller(PlatformAwareSeller):
    """Dry-run seller that simulates execution without blockchain calls."""
    
    def __init__(self, *args, dry_run_wait_time: float = 0.5, **kwargs):
        # Remove database_manager from kwargs since PlatformAwareSeller doesn't accept it
        kwargs.pop('database_manager', None)
        super().__init__(*args, **kwargs)
        self.dry_run_wait_time = dry_run_wait_time
    
    async def _execute_transaction(self, order: SellOrder) -> SellOrder:
        """Override to simulate instead of actually sending transaction."""
        # Simulate network latency
        logger.info(f"Simulating sell transaction (wait: {self.dry_run_wait_time}s)")
        await asyncio.sleep(self.dry_run_wait_time)
        
        # Generate fake signature
        order.tx_signature = f"DRYRUN_SELL_{order.token_info.mint}_{int(time.time()*1000)}"
        
        # Calculate fees
        order.transaction_fee_raw = 5000 + int((order.compute_unit_limit * order.priority_fee) / 1_000_000)
        order.platform_fee_raw = int(order.minimum_sol_swap_amount_raw * 0.008)  # 0.8% estimate
        
        logger.info(f"Sell transaction simulated: {order.tx_signature}")
        return order
    
    async def _confirm_transaction(self, tx_signature: str):
        """Override to simulate transaction confirmation."""
        logger.info(f"Simulating transaction confirmation: {tx_signature}")
        await asyncio.sleep(0.1)  # Brief simulation delay
        
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
        
        return BalanceChangeResult(
            token_swap_amount_raw=-order.token_amount_raw,
            sol_amount_raw=order.minimum_sol_swap_amount_raw-(order.platform_fee_raw+order.transaction_fee_raw),
            platform_fee_raw=order.platform_fee_raw,
            transaction_fee_raw=order.transaction_fee_raw,
            rent_exemption_amount_raw=0,
            sol_swap_amount_raw=order.minimum_sol_swap_amount_raw,
        )