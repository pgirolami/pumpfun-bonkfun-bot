"""
Market quality controller for probabilistic buy decisions based on recent trade performance.
"""

import random
from time import time

from core.pubkeys import LAMPORTS_PER_SOL
from database.manager import DatabaseManager
from database.models import PositionPnLData
from solders.pubkey import Pubkey
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
        algorithm: str = "win_rate",
    ) -> None:
        """Initialize market quality controller.

        Args:
            database_manager: Database manager instance
            lookback_minutes: Time window in minutes for analysis
            exploration_probability: Probability of buying regardless of quality (0.0-1.0)
            min_trades_for_analysis: Minimum number of closed positions required for analysis
            algorithm: Algorithm to use for quality score calculation ("win_rate", "avg_pnl", or "walk_pnl")
        """
        if database_manager is None:
            raise ValueError("database_manager is required")
        if lookback_minutes is None:
            raise ValueError("lookback_minutes is required")
        if exploration_probability is None:
            raise ValueError("exploration_probability is required")
        if min_trades_for_analysis is None:
            raise ValueError("min_trades_for_analysis is required")
        if algorithm not in ("win_rate", "avg_pnl", "walk_pnl"):
            raise ValueError(f"Invalid algorithm: {algorithm}. Must be 'win_rate', 'avg_pnl', or 'walk_pnl'")

        self.database_manager = database_manager
        self.lookback_minutes = lookback_minutes
        self.exploration_probability = exploration_probability
        self.min_trades_for_analysis = min_trades_for_analysis
        self.algorithm = algorithm

        # Initialize cached quality score (neutral)
        self._cached_quality_score = 1.0
        self._last_update_ts = 0.0
        
        # For walk_pnl algorithm: track current buy probability (starts at 100%)
        self._walk_pnl_buy_probability = 1.0

    async def _calculate_quality_score(self, run_id: str) -> float:
        """Calculate quality score from recent closed positions.

        Args:
            run_id: Bot run identifier

        Returns:
            Quality score (0.0-1.0)
        """
        # Query recent closed positions (only PnL data, not full Position objects)
        positions = await self.database_manager.get_recent_closed_positions_pnl_data(
            self.lookback_minutes, run_id
        )

        # If fewer positions than minimum required, return optimistic default
        if len(positions) < self.min_trades_for_analysis:
            logger.info(
                f"Market quality calculated ({self.algorithm}): {len(positions)} <= min_trades_for_analysis={self.min_trades_for_analysis} positions, returning 1.0"
            )
            return 1.0

        # Dispatch to algorithm-specific method
        if self.algorithm == "win_rate":
            return await self._calculate_win_rate_score(positions)
        elif self.algorithm == "avg_pnl":
            return await self._calculate_avg_pnl_score(positions)
        elif self.algorithm == "walk_pnl":
            return await self._calculate_walk_pnl_score(positions)
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")

    async def _calculate_win_rate_score(self, positions: list[PositionPnLData]) -> float:
        """Calculate quality score based on win rate.

        Args:
            positions: List of closed position PnL data

        Returns:
            Quality score (0.0-1.0)
        """
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
        quality_score = min(1.0, win_rate / 0.25)

        logger.info(
            f"Market quality calculated ({self.algorithm}): {wins}/{total_closed} wins "
            f"(win_rate: {win_rate:.2%}, quality_score: {quality_score:.2f})"
        )

        return quality_score

    async def _calculate_avg_pnl_score(self, positions: list[PositionPnLData]) -> float:
        """Calculate quality score based on average normalized PnL.

        Args:
            positions: List of closed position PnL data

        Returns:
            Quality score (0.0-1.0)
        """
        # Calculate normalized PnL for each position
        normalized_pnl_values = []
        for pos in positions:
            # Calculate actual amount spent in SOL (including all fees)
            amount_spent_lamports = (
                abs(pos.total_net_sol_swapout_amount_raw or 0)
                + (pos.transaction_fee_raw or 0)
                + (pos.platform_fee_raw or 0)
                + (pos.tip_fee_raw or 0)
            )
            amount_spent_sol = amount_spent_lamports / LAMPORTS_PER_SOL

            # Calculate normalized PnL
            if amount_spent_sol > 0 and pos.realized_pnl_sol_decimal is not None:
                normalized_pnl = pos.realized_pnl_sol_decimal / amount_spent_sol
                normalized_pnl_values.append(normalized_pnl)

        # Average all normalized PnL values
        if not normalized_pnl_values:
            logger.warning(
                f"Market quality calculated ({self.algorithm}): No valid normalized PnL values, returning 1.0"
            )
            return 1.0
        
        avg_normalized_pnl = sum(normalized_pnl_values) / len(normalized_pnl_values)

        UPPER = 0.01
        LOWER = -0.005
        # Apply linear function:
        # - normalized_pnl >= 0.01 → quality_score = 1.0
        # - normalized_pnl <= -0.01 → quality_score = 0.0
        # - -0.01 < normalized_pnl < 0.01 → linear interpolation
        if avg_normalized_pnl >= UPPER:
            quality_score = 1.0
        elif avg_normalized_pnl <= LOWER:
            quality_score = 0.0
        else:
            # Linear interpolation: y = (x + 0.01) / 0.02
            # At x = -0.01: y = 0.0
            # At x = 0.0: y = 0.5
            # At x = 0.01: y = 1.0
            quality_score = (avg_normalized_pnl + UPPER) / (UPPER - LOWER)

        # Clamp to [0.0, 1.0] range (safety check, though linear function should already be in range)
        quality_score = max(0.0, min(1.0, quality_score))

        logger.info(
            f"Market quality calculated ({self.algorithm}): "
            f"avg_normalized_pnl: {avg_normalized_pnl:.4f}, "
            f"quality_score: {quality_score:.2f} "
            f"(positions: {len(normalized_pnl_values)})"
        )

        return quality_score

    async def _calculate_walk_pnl_score(self, positions: list[PositionPnLData]) -> float:
        """Calculate quality score based on walking PnL (incremental updates).
        
        Starts at 100% buy probability. Each call multiplies current probability
        by (1 + 10 * normalized_pnl) based on the most recent position.

        Args:
            positions: List of closed position PnL data (ordered by exit time, most recent first)

        Returns:
            Quality score (0.0-1.0)
        """
        if not positions:
            # No positions, keep current probability
            return self._walk_pnl_buy_probability
        
        # Get the most recent position (first in list, as ordered by exit time DESC)
        most_recent_pos = positions[0]
        
        # Calculate normalized PnL for the most recent position
        amount_spent_lamports = (
            abs(most_recent_pos.total_net_sol_swapout_amount_raw or 0)
            + (most_recent_pos.transaction_fee_raw or 0)
            + (most_recent_pos.platform_fee_raw or 0)
            + (most_recent_pos.tip_fee_raw or 0)
        )
        amount_spent_sol = amount_spent_lamports / LAMPORTS_PER_SOL
        
        if amount_spent_sol > 0 and most_recent_pos.realized_pnl_sol_decimal is not None:
            normalized_pnl = most_recent_pos.realized_pnl_sol_decimal / amount_spent_sol
        else:
            normalized_pnl = 0.0
        
        clamped_normalized_pnl = max(-1.0, normalized_pnl)

        # Update buy probability: multiply by (1 + 10 * normalized_pnl)
        multiplier = max(0.05,1.0 + clamped_normalized_pnl)
        old_buy_probability = self._walk_pnl_buy_probability
        self._walk_pnl_buy_probability = self._walk_pnl_buy_probability * multiplier
        
        # Clamp to [0.0, 1.0] range
        self._walk_pnl_buy_probability = max(0.0, min(1.0, self._walk_pnl_buy_probability))
        
        logger.info(
            f"Market quality calculated ({self.algorithm}) using position {most_recent_pos.position_id[:8]} for mint {most_recent_pos.mint[:8]}: "
            f"amount_spent: {amount_spent_sol:.6f} SOL, "
            f"realized_pnl: {most_recent_pos.realized_pnl_sol_decimal:.6f} SOL, "
            f"last_normalized_pnl: {normalized_pnl:.4f}, "
            f"multiplier: {multiplier:.4f}, "
            f"buy_probability: {old_buy_probability:.4f} -> {self._walk_pnl_buy_probability:.4f}"
        )
        
        return self._walk_pnl_buy_probability

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
                f"Market quality score updated ({self.algorithm}): {quality_score:.2f} "
                f"(lookback: {self.lookback_minutes} minutes)"
            )
        except Exception as e:
            logger.exception(f"Failed to update market quality score: {e}")

    def should_buy(self, mint: Pubkey) -> bool:
        """Determine if bot should buy based on market quality.

        Returns:
            True if should buy, False otherwise
        """
        # Exploration: random chance to buy regardless of quality
        if random.random() < self.exploration_probability:
            logger.info(
                f"[{str(mint)[:8]}] Buy decision: EXPLORATION (prob: {self.exploration_probability:.2%}, "
                f"cached_quality: {self._cached_quality_score:.2f})"
            )
            return True

        # Exploitation: buy probability proportional to quality score
        buy_probability = self._cached_quality_score
        should_buy = random.random() < buy_probability

        decision_type = "EXPLOITATION (buy)" if should_buy else "EXPLOITATION (skip)"
        logger.info(
            f"[{str(mint)[:8]}] Buy decision: {decision_type} "
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

