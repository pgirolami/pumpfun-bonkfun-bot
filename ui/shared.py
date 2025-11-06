"""
Shared functions for PNL visualization.

This module contains functions used by both the standalone visualization script
and the web dashboard.
"""

import sqlite3
import statistics
from datetime import datetime
from pathlib import Path


def query_positions(
    db_path: str, start_timestamp: int
) -> list[tuple[int, float, int | None]]:
    """Query closed positions from database.

    Args:
        db_path: Path to SQLite database file
        start_timestamp: Unix epoch milliseconds - filter positions with entry_ts >= this

    Returns:
        List of tuples (entry_ts, realized_pnl_sol_decimal, exit_ts) ordered by entry_ts
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.execute(
        """
        SELECT entry_ts, realized_pnl_sol_decimal, exit_ts
        FROM positions
        WHERE realized_pnl_sol_decimal IS NOT NULL
          AND entry_ts >= ?
          AND exit_ts IS NOT NULL
        ORDER BY entry_ts ASC
        """,
        (start_timestamp,),
    )
    results = [
        (row["entry_ts"], row["realized_pnl_sol_decimal"], row["exit_ts"])
        for row in cursor
    ]
    conn.close()
    return results


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


def calculate_max_drawdown(cumulative_pnl: list[float]) -> float:
    """Calculate maximum drawdown as a percentage from the peak.

    Max drawdown is the largest peak-to-trough decline in the cumulative PNL,
    expressed as a percentage of the peak value. Drawdown is only calculated
    when the value is below the peak, and is capped at 100%.

    Args:
        cumulative_pnl: List of cumulative PNL values over time

    Returns:
        Maximum drawdown percentage (0-100, representing the largest drop from peak)
    """
    if not cumulative_pnl:
        return 0.0

    max_drawdown_pct = 0.0
    peak = cumulative_pnl[0]

    for value in cumulative_pnl:
        if value > peak:
            # New peak reached, update peak
            peak = value
        elif value < peak:
            # We're in a drawdown (below the peak)
            # Only calculate drawdown if peak is positive
            if peak > 0:
                drawdown_pct = ((peak - value) / peak) * 100
                # Cap at 100% (you can't lose more than 100% of the peak)
                drawdown_pct = min(drawdown_pct, 100.0)
                if drawdown_pct > max_drawdown_pct:
                    max_drawdown_pct = drawdown_pct

    return max_drawdown_pct


def process_database(
    db_path: str, start_timestamp: int
) -> tuple[str, list[datetime], list[float], dict[str, float | int]] | None:
    """Process a single database and return data for plotting.

    Args:
        db_path: Path to database file
        start_timestamp: Start timestamp to filter positions

    Returns:
        Tuple of (label, timestamps, cumulative_pnl, stats) or None if no data
    """
    positions = query_positions(db_path, start_timestamp)

    if not positions:
        return None

    # Extract timestamps and PNL values
    timestamps_ms = [pos[0] for pos in positions]
    pnl_values = [pos[1] for pos in positions]
    exit_timestamps_ms = [pos[2] for pos in positions]

    # Convert timestamps to datetime objects
    timestamps = [datetime.fromtimestamp(ts / 1000.0) for ts in timestamps_ms]

    # Calculate cumulative PNL
    cumulative_pnl = []
    running_total = 0.0
    for pnl in pnl_values:
        running_total += pnl
        cumulative_pnl.append(running_total)

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
    win_rate = (winning_positions / num_positions * 100) if num_positions > 0 else 0.0
    avg_pnl = sum(pnl_values) / num_positions if num_positions > 0 else 0.0
    max_drawdown = calculate_max_drawdown(cumulative_pnl)

    # Extract label from path
    label = extract_label(db_path)

    stats = {
        "total_pnl": total_pnl,
        "num_positions": num_positions,
        "winning_positions": winning_positions,
        "win_rate": win_rate,
        "avg_pnl": avg_pnl,
        "max_drawdown": max_drawdown,
        "avg_duration": avg_duration,
        "median_duration": median_duration,
        "p10_duration": p10_duration,
        "p90_duration": p90_duration,
    }

    return (label, timestamps, cumulative_pnl, stats)

