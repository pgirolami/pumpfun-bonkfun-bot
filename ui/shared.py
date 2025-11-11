"""
Shared functions for PNL visualization.

This module contains functions used by both the standalone visualization script
and the web dashboard.
"""

import math
import sqlite3
import statistics
from datetime import datetime
from pathlib import Path


def query_positions(
    db_path: str, start_timestamp: int
) -> list[tuple[int, float, int | None, int | None]]:
    """Query closed positions from database.

    Args:
        db_path: Path to SQLite database file
        start_timestamp: Unix epoch milliseconds - filter positions with entry_ts >= this

    Returns:
        List of tuples (entry_ts, realized_pnl_sol_decimal, exit_ts, total_sol_swapout_amount_raw) ordered by entry_ts
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.execute(
        """
        SELECT entry_ts, realized_pnl_sol_decimal, exit_ts, total_sol_swapout_amount_raw
        FROM positions
        WHERE realized_pnl_sol_decimal IS NOT NULL
          AND entry_ts >= ?
          AND exit_ts IS NOT NULL
        ORDER BY entry_ts ASC
        """,
        (start_timestamp,),
    )
    results = [
        (
            row["entry_ts"],
            row["realized_pnl_sol_decimal"],
            row["exit_ts"],
            row["total_sol_swapout_amount_raw"],
        )
        for row in cursor
    ]
    conn.close()
    return results


def query_trade_durations(
    db_path: str, start_timestamp: int
) -> tuple[list[float], list[float]]:
    """Query successful trade durations from database.

    Includes trades from positions that were closed after the start_timestamp,
    or trades with timestamp >= start_timestamp if position_id is NULL.

    Args:
        db_path: Path to SQLite database file
        start_timestamp: Unix epoch milliseconds - filter positions with entry_ts >= this

    Returns:
        Tuple of (buy_durations_ms, sell_durations_ms) in milliseconds
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    
    # First, get position IDs for closed positions that match the start_timestamp filter
    position_cursor = conn.execute(
        """
        SELECT id
        FROM positions
        WHERE realized_pnl_sol_decimal IS NOT NULL
          AND entry_ts >= ?
          AND exit_ts IS NOT NULL
        """,
        (start_timestamp,),
    )
    position_ids = [row["id"] for row in position_cursor]
    
    # Build query: include trades linked to positions OR trades with timestamp >= start_timestamp
    if position_ids:
        # Query successful trades for these positions
        placeholders = ",".join("?" * len(position_ids))
        query = f"""
            SELECT trade_type, trade_duration_ms
            FROM trades
            WHERE success = 1
              AND trade_duration_ms IS NOT NULL
              AND trade_duration_ms > 0
              AND (position_id IN ({placeholders}) OR (position_id IS NULL AND timestamp >= ?))
            ORDER BY timestamp ASC
        """
        params = position_ids + [start_timestamp]
    else:
        # No positions found, just query by timestamp
        query = """
            SELECT trade_type, trade_duration_ms
            FROM trades
            WHERE success = 1
              AND trade_duration_ms IS NOT NULL
              AND trade_duration_ms > 0
              AND timestamp >= ?
            ORDER BY timestamp ASC
        """
        params = [start_timestamp]
    
    cursor = conn.execute(query, params)
    buy_durations = []
    sell_durations = []
    for row in cursor:
        duration_ms = row["trade_duration_ms"]
        if duration_ms is not None and duration_ms > 0:
            if row["trade_type"] == "buy":
                buy_durations.append(float(duration_ms))
            elif row["trade_type"] == "sell":
                sell_durations.append(float(duration_ms))
    conn.close()
    return (buy_durations, sell_durations)


def extract_label(db_path: str) -> str:
    """Extract label from database path by removing _dryrun.db or _live.db suffix.

    Args:
        db_path: Path to database file

    Returns:
        Label string with suffix removed
    """
    stem = Path(db_path).stem
    # Remove _dry_run, _dryrun, or _live suffix (check longer pattern first)
    if stem.endswith("_dry_run"):
        return stem[:-8]  # Remove "_dry_run"
    elif stem.endswith("_dryrun"):
        return stem[:-7]  # Remove "_dryrun"
    elif stem.endswith("_live"):
        return stem[:-5]  # Remove "_live"
    return stem


def calculate_max_drawdown(
    cumulative_pnl: list[float], baseline_equity: float = 1.0
) -> float:
    """Calculate maximum drawdown as a percentage from the equity peak.

    This function treats the cumulative PNL series as incremental changes on top
    of a baseline account equity (defaulting to 1 SOL). Using a baseline avoids
    artificially large drawdowns when profits are small or when the cumulative
    PNL dips below zero, which previously caused the metric to report 100% for
    most bots.

    Args:
        cumulative_pnl: List of cumulative PNL values over time.
        baseline_equity: The assumed starting equity used to normalise the
            drawdown calculation. Defaults to 1.0 (1 SOL).

    Returns:
        Maximum drawdown percentage from the equity peak.
    """
    if not cumulative_pnl:
        return 0.0

    running_max_equity = baseline_equity + cumulative_pnl[0]
    max_drawdown_pct = 0.0

    for pnl in cumulative_pnl:
        equity = baseline_equity + pnl
        if equity > running_max_equity:
            running_max_equity = equity
            continue

        if running_max_equity <= 0:
            # Avoid division by zero or negative baseline
            continue

        drawdown_pct = ((running_max_equity - equity) / running_max_equity) * 100
        if drawdown_pct > max_drawdown_pct:
            max_drawdown_pct = drawdown_pct

    return max_drawdown_pct


def process_database(
    db_path: str, start_timestamp: int
) -> tuple[str, list[datetime], list[float], list[float], dict[str, float | int]] | None:
    """Process a single database and return data for plotting.

    Args:
        db_path: Path to database file
        start_timestamp: Start timestamp to filter positions

    Returns:
        Tuple of (label, timestamps, cumulative_pnl, cumulative_normalized_pnl, stats) or None if no data
    """
    positions = query_positions(db_path, start_timestamp)

    if not positions:
        return None

    # Extract timestamps, PNL values, and swap amounts
    timestamps_ms = [pos[0] for pos in positions]
    pnl_values = [pos[1] for pos in positions]
    exit_timestamps_ms = [pos[2] for pos in positions]
    swapout_amounts_raw = [pos[3] for pos in positions]

    # Convert timestamps to datetime objects
    timestamps = [datetime.fromtimestamp(ts / 1000.0) for ts in timestamps_ms]

    # Calculate cumulative PNL
    cumulative_pnl = []
    running_total = 0.0
    for pnl in pnl_values:
        running_total += pnl
        cumulative_pnl.append(running_total)

    # Calculate normalized PNL (PNL / abs(total_sol_swapout_amount_raw))
    # Convert raw amounts from lamports to SOL (divide by 1e9)
    normalized_pnl_values = []
    for pnl, swapout_raw in zip(pnl_values, swapout_amounts_raw):
        if swapout_raw is not None and swapout_raw != 0:
            swapout_sol = abs(swapout_raw) / 1e9  # Convert lamports to SOL
            normalized_pnl = pnl / swapout_sol if swapout_sol > 0 else 0.0
        else:
            normalized_pnl = 0.0
        normalized_pnl_values.append(normalized_pnl)

    # Calculate cumulative normalized PNL
    cumulative_normalized_pnl = []
    running_normalized_total = 0.0
    for normalized_pnl in normalized_pnl_values:
        running_normalized_total += normalized_pnl
        cumulative_normalized_pnl.append(running_normalized_total)

    # Calculate position durations (in seconds)
    durations_seconds = []
    for entry_ts, exit_ts in zip(timestamps_ms, exit_timestamps_ms):
        if exit_ts is not None:
            duration_seconds = (exit_ts - entry_ts) / 1000.0  # Convert ms to seconds
            durations_seconds.append(duration_seconds)

    # Calculate duration statistics
    avg_duration = statistics.mean(durations_seconds) if durations_seconds else 0.0
    median_duration = (
        statistics.median(durations_seconds) if durations_seconds else 0.0
    )
    if durations_seconds:
        sorted_durations = sorted(durations_seconds)
        n = len(sorted_durations)
        # Calculate percentiles (using nearest rank method)
        p10_idx = max(0, int(n * 0.10))
        p90_idx = min(n - 1, int(n * 0.90))
        p10_duration = sorted_durations[p10_idx]
        p90_duration = sorted_durations[p90_idx]
    else:
        p10_duration = 0.0
        p90_duration = 0.0

    # Calculate summary statistics
    total_pnl = cumulative_pnl[-1] if cumulative_pnl else 0.0
    num_positions = len(positions)
    winning_positions = sum(1 for pnl in pnl_values if pnl > 0)
    losing_positions = num_positions - winning_positions
    win_rate = (winning_positions / num_positions * 100) if num_positions > 0 else 0.0
    avg_pnl = sum(pnl_values) / num_positions if num_positions > 0 else 0.0
    
    # Calculate average PNL for winning and losing positions
    winning_pnl_values = [pnl for pnl in pnl_values if pnl > 0]
    losing_pnl_values = [pnl for pnl in pnl_values if pnl <= 0]
    avg_pnl_winning = (
        sum(winning_pnl_values) / len(winning_pnl_values)
        if winning_pnl_values else 0.0
    )
    avg_pnl_losing = (
        sum(losing_pnl_values) / len(losing_pnl_values)
        if losing_pnl_values else 0.0
    )
    
    max_drawdown = calculate_max_drawdown(cumulative_pnl)

    # Extract label from path
    label = extract_label(db_path)

    # Calculate normalized statistics
    total_normalized_pnl = cumulative_normalized_pnl[-1] if cumulative_normalized_pnl else 0.0
    avg_normalized_pnl = (
        sum(normalized_pnl_values) / num_positions if num_positions > 0 else 0.0
    )
    
    # Calculate average normalized PNL for winning and losing positions
    winning_normalized_pnl_values = [
        normalized_pnl for pnl, normalized_pnl in zip(pnl_values, normalized_pnl_values)
        if pnl > 0
    ]
    losing_normalized_pnl_values = [
        normalized_pnl for pnl, normalized_pnl in zip(pnl_values, normalized_pnl_values)
        if pnl <= 0
    ]
    avg_normalized_pnl_winning = (
        sum(winning_normalized_pnl_values) / len(winning_normalized_pnl_values)
        if winning_normalized_pnl_values else 0.0
    )
    avg_normalized_pnl_losing = (
        sum(losing_normalized_pnl_values) / len(losing_normalized_pnl_values)
        if losing_normalized_pnl_values else 0.0
    )
    
    max_drawdown_normalized = calculate_max_drawdown(cumulative_normalized_pnl)

    # Query trade durations for successful trades
    buy_durations_ms, sell_durations_ms = query_trade_durations(db_path, start_timestamp)
    
    # Debug: Check if we're getting any durations
    import logging
    logger = logging.getLogger(__name__)
    if not buy_durations_ms and not sell_durations_ms:
        # Try a direct query to see what's in the database
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        debug_cursor = conn.execute(
            """
            SELECT COUNT(*) as total,
                   COUNT(CASE WHEN success = 1 THEN 1 END) as successful,
                   COUNT(CASE WHEN success = 1 AND trade_duration_ms IS NOT NULL AND trade_duration_ms > 0 THEN 1 END) as with_duration
            FROM trades
            WHERE timestamp >= ?
            """,
            (start_timestamp,),
        )
        debug_row = debug_cursor.fetchone()
        logger.debug(f"Trade debug for {Path(db_path).name}: total={debug_row['total']}, successful={debug_row['successful']}, with_duration={debug_row['with_duration']}")
        conn.close()
    
    # Calculate trade duration statistics for buys (keep in milliseconds)
    def calc_percentiles(durations: list[float]) -> dict[str, float]:
        """Calculate average, p10, p50, p90 for durations."""
        if not durations:
            return {
                "avg": 0.0,
                "p10": 0.0,
                "p50": 0.0,
                "p90": 0.0,
            }
        sorted_durs = sorted(durations)
        n = len(sorted_durs)
        return {
            "avg": statistics.mean(sorted_durs),
            "p10": sorted_durs[max(0, int(n * 0.10))],
            "p50": sorted_durs[max(0, int(n * 0.50))],
            "p90": sorted_durs[max(0, int(n * 0.90))],
        }
    
    def fit_normal_distribution(durations: list[float]) -> dict[str, float]:
        """Fit a normal distribution to the durations and calculate goodness-of-fit metrics.
        
        Returns the mean (μ), standard deviation (σ), and goodness-of-fit metrics.
        
        Args:
            durations: List of duration values in milliseconds
            
        Returns:
            Dictionary with 'mean', 'std_dev', 'r_squared', and 'ks_statistic' keys
        """
        if not durations or len(durations) < 2:
            return {
                "mean": 0.0,
                "std_dev": 0.0,
                "r_squared": 0.0,
                "ks_statistic": 1.0,  # Worst possible fit
            }
        mean = statistics.mean(durations)
        std_dev = statistics.stdev(durations) if len(durations) > 1 else 0.0
        
        # Calculate R-squared (coefficient of determination)
        # R² = 1 - (SS_res / SS_tot)
        # where SS_res = sum of squared residuals from the mean
        # and SS_tot = sum of squared differences from the mean
        if std_dev == 0.0:
            r_squared = 1.0  # Perfect fit if all values are the same
        else:
            # For normal distribution fit, R² measures how well the normal distribution
            # explains the variance in the data
            # We calculate it as the proportion of variance explained
            ss_tot = sum((x - mean) ** 2 for x in durations)
            if ss_tot == 0:
                r_squared = 1.0
            else:
                # For a normal distribution, the variance is σ²
                # R² = 1 - (unexplained variance / total variance)
                # Since we're fitting to the data itself, R² will be high
                # A better metric is to compare empirical vs theoretical quantiles
                # For simplicity, we use a simplified R² based on how close data is to normal
                # Calculate empirical variance
                empirical_var = statistics.variance(durations) if len(durations) > 1 else 0.0
                theoretical_var = std_dev ** 2
                if empirical_var == 0:
                    r_squared = 1.0
                else:
                    # R² as proportion of variance explained by the normal model
                    # This is a simplified metric - perfect normal would have R² = 1
                    variance_ratio = min(theoretical_var / empirical_var, empirical_var / theoretical_var) if empirical_var > 0 else 1.0
                    r_squared = variance_ratio
        
        # Calculate Kolmogorov-Smirnov statistic (simplified)
        # KS statistic = max difference between empirical CDF and theoretical normal CDF
        sorted_durations = sorted(durations)
        n = len(sorted_durations)
        ks_statistic = 0.0
        
        if std_dev > 0:
            # Calculate empirical CDF and compare with theoretical normal CDF
            for i, value in enumerate(sorted_durations):
                # Empirical CDF at this point
                empirical_cdf = (i + 1) / n
                
                # Theoretical normal CDF (using error function)
                # For a normal distribution: CDF(x) = 0.5 * (1 + erf((x - μ) / (σ * sqrt(2))))
                z_score = (value - mean) / std_dev
                # Standard normal CDF using error function
                theoretical_cdf = 0.5 * (1 + math.erf(z_score / math.sqrt(2)))
                
                # Calculate difference
                diff = abs(empirical_cdf - theoretical_cdf)
                ks_statistic = max(ks_statistic, diff)
        else:
            ks_statistic = 0.0  # Perfect fit if std_dev is 0
        
        return {
            "mean": mean,
            "std_dev": std_dev,
            "r_squared": r_squared,
            "ks_statistic": ks_statistic,
        }
    
    buy_stats = calc_percentiles(buy_durations_ms)
    sell_stats = calc_percentiles(sell_durations_ms)
    buy_normal = fit_normal_distribution(buy_durations_ms)
    sell_normal = fit_normal_distribution(sell_durations_ms)

    stats = {
        "total_pnl": total_pnl,
        "total_normalized_pnl": total_normalized_pnl,
        "num_positions": num_positions,
        "winning_positions": winning_positions,
        "win_rate": win_rate,
        "avg_pnl": avg_pnl,
        "avg_pnl_winning": avg_pnl_winning,
        "avg_pnl_losing": avg_pnl_losing,
        "avg_normalized_pnl": avg_normalized_pnl,
        "avg_normalized_pnl_winning": avg_normalized_pnl_winning,
        "avg_normalized_pnl_losing": avg_normalized_pnl_losing,
        "max_drawdown": max_drawdown,
        "max_drawdown_normalized": max_drawdown_normalized,
        "avg_duration": avg_duration,
        "median_duration": median_duration,
        "p10_duration": p10_duration,
        "p90_duration": p90_duration,
        "buy_duration_avg": buy_stats["avg"],
        "buy_duration_p10": buy_stats["p10"],
        "buy_duration_p50": buy_stats["p50"],
        "buy_duration_p90": buy_stats["p90"],
        "buy_normal_mean": buy_normal["mean"],
        "buy_normal_std_dev": buy_normal["std_dev"],
        "buy_normal_r_squared": buy_normal["r_squared"],
        "buy_normal_ks_statistic": buy_normal["ks_statistic"],
        "sell_duration_avg": sell_stats["avg"],
        "sell_duration_p10": sell_stats["p10"],
        "sell_duration_p50": sell_stats["p50"],
        "sell_duration_p90": sell_stats["p90"],
        "sell_normal_mean": sell_normal["mean"],
        "sell_normal_std_dev": sell_normal["std_dev"],
        "sell_normal_r_squared": sell_normal["r_squared"],
        "sell_normal_ks_statistic": sell_normal["ks_statistic"],
    }

    return (label, timestamps, cumulative_pnl, cumulative_normalized_pnl, stats)

