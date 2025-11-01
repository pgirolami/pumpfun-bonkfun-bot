"""
Market quality controller for probabilistic buy decisions based on recent trade performance.
"""

import random
from time import time

from database.manager import DatabaseManager
from trading.position import Position
from utils.logger import get_logger

logger = get_logger(__name__)


class MarketQualityController:
    """Controller for market quality analysis and buy decisions."""

    def __init__(
        self,
        database_manager: DatabaseManager,
        lookback_minutes: int,
        exploration_probability: float,
        min_trades_for_analysis: int,
    ) -> None:
        """Initialize market quality controller.

        Args:
            database_manager: Database manager instance
            lookback_minutes: Time window in minutes for analysis
            exploration_probability: Probability of buying regardless of quality (0.0-1.0)
            min_trades_for_analysis: Minimum number of closed positions required for analysis
        """
        if database_manager is None:
            raise ValueError("database_manager is required")
        if lookback_minutes is None:
            raise ValueError("lookback_minutes is required")
        if exploration_probability is None:
            raise ValueError("exploration_probability is required")
        if min_trades_for_analysis is None:
            raise ValueError("min_trades_for_analysis is required")

        self.database_manager = database_manager
        self.lookback_minutes = lookback_minutes
        self.exploration_probability = exploration_probability
        self.min_trades_for_analysis = min_trades_for_analysis

        # Initialize cached quality score (neutral)
        self._cached_quality_score = 1.0
        self._last_update_ts = 0.0

    async def _calculate_quality_score(self, run_id: str) -> float:
        """Calculate quality score from recent closed positions.

        Args:
            run_id: Bot run identifier

        Returns:
            Quality score (0.0-1.0)
        """
        # Query recent closed positions
        positions = await self.database_manager.get_recent_closed_positions(
            self.lookback_minutes, run_id
        )

        # If fewer positions than minimum required, return optimistic default
        if len(positions) < self.min_trades_for_analysis:
            return 1.0

        # Calculate win rate
        wins = sum(
            1
            for pos in positions
            if pos.realized_pnl_sol_decimal is not None
            and pos.realized_pnl_sol_decimal > 0
        )
        total_closed = len(positions)
        win_rate = wins / total_closed if total_closed > 0 else 0.0

        # Quality score mapping: 50% win rate = 1.0
        # Formula: score = min(1.0, win_rate / 0.5)
        quality_score = min(1.0, win_rate / 0.5)

        logger.info(
            f"Market quality calculated: {wins}/{total_closed} wins "
            f"(win_rate: {win_rate:.2%}, quality_score: {quality_score:.2f})"
        )

        return quality_score

    async def update_quality_score(self, run_id: str) -> None:
        """Update cached quality score from database.

        Args:
            run_id: Bot run identifier
        """
        try:
            quality_score = await self._calculate_quality_score(run_id)
            self._cached_quality_score = quality_score
            self._last_update_ts = time()
            logger.info(
                f"Market quality score updated: {quality_score:.2f} "
                f"(lookback: {self.lookback_minutes} minutes)"
            )
        except Exception as e:
            logger.exception(f"Failed to update market quality score: {e}")

    def should_buy(self) -> bool:
        """Determine if bot should buy based on market quality.

        Returns:
            True if should buy, False otherwise
        """
        # Exploration: random chance to buy regardless of quality
        if random.random() < self.exploration_probability:
            logger.info(
                f"Buy decision: EXPLORATION (prob: {self.exploration_probability:.2%}, "
                f"cached_quality: {self._cached_quality_score:.2f})"
            )
            return True

        # Exploitation: buy probability proportional to quality score
        buy_probability = self._cached_quality_score
        should_buy = random.random() < buy_probability

        decision_type = "EXPLOITATION (buy)" if should_buy else "EXPLOITATION (skip)"
        logger.info(
            f"Buy decision: {decision_type} "
            f"(quality_score: {self._cached_quality_score:.2f}, "
            f"buy_probability: {buy_probability:.2%})"
        )

        return should_buy

    def get_cached_quality_score(self) -> float:
        """Get current cached quality score.

        Returns:
            Cached quality score (0.0-1.0)
        """
        return self._cached_quality_score

