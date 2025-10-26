"""
Wallet balance logger for monitoring SOL balances across all bot instances.
"""

import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Set

from solders.pubkey import Pubkey

from core.client import SolanaClient
from core.pubkeys import LAMPORTS_PER_SOL
from utils.logger import get_logger, setup_file_logging

logger = get_logger(__name__)


class WalletBalanceLogger:
    """Manages wallet balance logging across all bot instances."""

    def __init__(self, rpc_endpoint: str):
        """Initialize the wallet balance logger.

        Args:
            rpc_endpoint: Solana RPC endpoint for balance queries
        """
        self.rpc_endpoint = rpc_endpoint
        self.client = SolanaClient(rpc_endpoint)
        self.wallets: Set[Pubkey] = set()
        self.wallet_names: Dict[Pubkey, str] = {}
        self._running = False
        self._task: asyncio.Task | None = None
        
        # Set up file logging for wallet balance reports
        self._setup_file_logging()

    def _setup_file_logging(self) -> None:
        """Set up file logging for wallet balance reports."""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = log_dir / f"wallet_balances_{timestamp}.log"
        
        setup_file_logging(str(log_filename))

    def register_wallet(self, wallet_pubkey: Pubkey, bot_name: str) -> None:
        """Register a wallet for balance monitoring.

        Args:
            wallet_pubkey: Public key of the wallet
            bot_name: Name of the bot using this wallet
        """
        self.wallets.add(wallet_pubkey)
        self.wallet_names[wallet_pubkey] = bot_name
        logger.debug(f"Registered wallet {wallet_pubkey} for bot '{bot_name}'")

    def unregister_wallet(self, wallet_pubkey: Pubkey) -> None:
        """Unregister a wallet from balance monitoring.

        Args:
            wallet_pubkey: Public key of the wallet to unregister
        """
        if wallet_pubkey in self.wallets:
            self.wallets.remove(wallet_pubkey)
            bot_name = self.wallet_names.pop(wallet_pubkey, "unknown")
            logger.debug(f"Unregistered wallet {wallet_pubkey} from bot '{bot_name}'")

    async def log_wallet_balances(self) -> None:
        """Log the current SOL balance for all registered wallets."""
        if not self.wallets:
            logger.debug("No wallets registered for balance logging")
            return

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logger.info(f"=== Wallet Balance Report - {timestamp} ===")

        for wallet_pubkey in self.wallets:
            try:
                balance_lamports = await self.client.get_sol_balance(wallet_pubkey)
                balance_sol = balance_lamports / LAMPORTS_PER_SOL
                bot_name = self.wallet_names.get(wallet_pubkey, "unknown")
                
                logger.info(
                    f"Wallet {wallet_pubkey} (bot: {bot_name}): "
                    f"{balance_sol:.6f} SOL ({balance_lamports:,} lamports)"
                )
            except Exception as e:
                bot_name = self.wallet_names.get(wallet_pubkey, "unknown")
                logger.error(
                    f"Failed to get balance for wallet {wallet_pubkey} "
                    f"(bot: {bot_name}): {e}"
                )

        logger.info("=== End Wallet Balance Report ===")

    async def start_balance_monitoring(self, interval_minutes: int = 1) -> None:
        """Start the background balance monitoring task.

        Args:
            interval_minutes: Interval in minutes between balance logs
        """
        if self._running:
            logger.warning("Balance monitoring is already running")
            return

        self._running = True
        interval_seconds = interval_minutes * 60
        
        logger.info(f"Starting wallet balance monitoring (every {interval_minutes} minute(s))")
        logger.debug(f"Monitoring {len(self.wallets)} wallets")
        
        # Log initial balances
        await self.log_wallet_balances()

        # Start background task
        self._task = asyncio.create_task(self._monitoring_loop(interval_seconds))
        logger.debug("Balance monitoring task started")

    async def stop_balance_monitoring(self) -> None:
        """Stop the background balance monitoring task."""
        if not self._running:
            logger.warning("Balance monitoring is not running")
            return

        self._running = False
        
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

        logger.debug("Stopped wallet balance monitoring")

    async def _monitoring_loop(self, interval_seconds: float) -> None:
        """Background loop for balance monitoring."""
        logger.debug(f"Balance monitoring loop started with {interval_seconds}s interval")
        while self._running:
            try:
                await asyncio.sleep(interval_seconds)
                if self._running:  # Check again after sleep
                    logger.debug("Running scheduled balance check")
                    await self.log_wallet_balances()
            except asyncio.CancelledError:
                logger.debug("Balance monitoring task cancelled")
                break
            except Exception as e:
                logger.exception(f"Error in balance monitoring loop: {e}")
                # Continue monitoring even if one iteration fails
                await asyncio.sleep(5)  # Short delay before retrying

    async def close(self) -> None:
        """Close the balance logger and cleanup resources."""
        await self.stop_balance_monitoring()
        await self.client.close()
        logger.debug("Wallet balance logger closed")


# Global instance for use across the application
_balance_logger: WalletBalanceLogger | None = None


def get_balance_logger(rpc_endpoint: str) -> WalletBalanceLogger:
    """Get or create the global wallet balance logger instance.

    Args:
        rpc_endpoint: Solana RPC endpoint for balance queries

    Returns:
        Global WalletBalanceLogger instance
    """
    global _balance_logger
    if _balance_logger is None:
        _balance_logger = WalletBalanceLogger(rpc_endpoint)
    return _balance_logger


async def cleanup_balance_logger() -> None:
    """Cleanup the global balance logger instance."""
    global _balance_logger
    if _balance_logger is not None:
        await _balance_logger.close()
        _balance_logger = None
