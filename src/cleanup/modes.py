from cleanup.manager import AccountCleanupManager
from utils.logger import get_logger

logger = get_logger(__name__)


async def cleanup_after_sell(
    client,
    wallet,
    mint,
    priority_fee_manager,
    cleanup_with_prior_fee,
    force_burn,
):
    """Always cleanup ATA after successful sell."""
    logger.info("[Cleanup] After sell.")
    manager = AccountCleanupManager(
        client, wallet, priority_fee_manager, cleanup_with_prior_fee, force_burn
    )
    await manager.cleanup_ata(mint)
