"""
Compare dry-run and live trading results from databases.

This script matches positions and trades between dry-run and live databases,
calculates differences in PNL, fees, timestamps, and other metrics, and
generates detailed comparison reports.

Usage:
    uv run analysis/compare_dryrun_live.py <dryrun_db_path> <live_db_path> \\
        [--output-dir <dir>] \\
        [--start-timestamp <ms>] \\
        [--end-timestamp <ms>] \\
        [--position-tolerance <ms>] \\
        [--trade-tolerance <ms>]

Examples:
    # Compare all positions and trades
    uv run analysis/compare_dryrun_live.py data/bot_dryrun.db data/bot_live.db

    # Compare only positions/trades from a specific time range (using date-time strings)
    uv run analysis/compare_dryrun_live.py data/bot_dryrun.db data/bot_live.db \\
        --start-timestamp "2024-01-01 00:00:00" \\
        --end-timestamp "2024-01-02 00:00:00"

    # Or use Unix epoch milliseconds
    uv run analysis/compare_dryrun_live.py data/bot_dryrun.db data/bot_live.db \\
        --start-timestamp 1704067200000 \\
        --end-timestamp 1704153600000
"""

import argparse
import csv
import json
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import plotly.graph_objects as go
from plotly.subplots import make_subplots


def parse_timestamp(value: str | int | None) -> int | None:
    """Parse timestamp from string or integer.

    Args:
        value: Timestamp as string (YYYY-MM-DD HH:MM:SS) or integer (Unix epoch milliseconds)

    Returns:
        Timestamp in Unix epoch milliseconds, or None if value is None

    Raises:
        ValueError: If timestamp format is invalid
    """
    if value is None:
        return None

    # If it's already an integer, assume it's milliseconds
    if isinstance(value, int):
        return value

    # Try parsing as date-time string
    try:
        # Try format: YYYY-MM-DD HH:MM:SS
        dt = datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
        # Convert to Unix epoch milliseconds
        return int(dt.timestamp() * 1000)
    except ValueError:
        # If that fails, try parsing as integer string
        try:
            return int(value)
        except ValueError:
            raise ValueError(
                f"Invalid timestamp format: {value}. "
                "Expected format: 'YYYY-MM-DD HH:MM:SS' or Unix epoch milliseconds (integer)"
            )


def query_all_positions(
    db_path: str,
    start_timestamp: int | None = None,
    end_timestamp: int | None = None,
) -> list[dict[str, Any]]:
    """Query all closed positions from database.

    Args:
        db_path: Path to SQLite database file
        start_timestamp: Optional start timestamp in milliseconds (filters by entry_ts >= start_timestamp)
        end_timestamp: Optional end timestamp in milliseconds (filters by entry_ts <= end_timestamp)

    Returns:
        List of position dictionaries with all fields
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    
    # Build query with optional timestamp filters
    query = """
        SELECT 
            id, mint, platform, entry_net_price_decimal, token_decimals,
            total_token_swapin_amount_raw, total_token_swapout_amount_raw,
            entry_ts, exit_strategy, highest_price, max_no_price_change_time,
            last_price_change_ts, is_active, exit_reason, exit_net_price_decimal,
            exit_ts, transaction_fee_raw, platform_fee_raw, tip_fee_raw,
            rent_exemption_amount_raw, unattributed_sol_amount_raw,
            realized_pnl_sol_decimal, realized_net_pnl_sol_decimal,
            buy_amount, total_net_sol_swapout_amount_raw, total_net_sol_swapin_amount_raw,
            total_sol_swapout_amount_raw, total_sol_swapin_amount_raw,
            created_ts, updated_ts
        FROM positions
        WHERE exit_ts IS NOT NULL
          AND realized_pnl_sol_decimal IS NOT NULL
    """
    
    params = []
    if start_timestamp is not None:
        query += " AND entry_ts >= ?"
        params.append(start_timestamp)
    if end_timestamp is not None:
        query += " AND entry_ts <= ?"
        params.append(end_timestamp)
    
    query += " ORDER BY entry_ts ASC"
    
    cursor = conn.execute(query, params)
    positions = [dict(row) for row in cursor]
    conn.close()
    return positions


def query_all_trades(
    db_path: str,
    start_timestamp: int | None = None,
    end_timestamp: int | None = None,
) -> list[dict[str, Any]]:
    """Query all trades from database.

    Args:
        db_path: Path to SQLite database file
        start_timestamp: Optional start timestamp in milliseconds (filters by timestamp >= start_timestamp)
        end_timestamp: Optional end timestamp in milliseconds (filters by timestamp <= end_timestamp)

    Returns:
        List of trade dictionaries with all fields
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    
    # Build query with optional timestamp filters
    query = """
        SELECT 
            mint, timestamp, position_id, success, platform, trade_type,
            tx_signature, error_message, token_swap_amount_raw,
            net_sol_swap_amount_raw, transaction_fee_raw, platform_fee_raw,
            tip_fee_raw, rent_exemption_amount_raw, unattributed_sol_amount_raw,
            sol_swap_amount_raw, price_decimal, net_price_decimal,
            trade_duration_ms, time_to_block_ms, run_id, block_time
        FROM trades
        WHERE 1=1
    """
    
    params = []
    if start_timestamp is not None:
        query += " AND timestamp >= ?"
        params.append(start_timestamp)
    if end_timestamp is not None:
        query += " AND timestamp <= ?"
        params.append(end_timestamp)
    
    query += " ORDER BY timestamp ASC"
    
    cursor = conn.execute(query, params)
    trades = [dict(row) for row in cursor]
    conn.close()
    return trades


def match_positions(
    dryrun_positions: list[dict[str, Any]],
    live_positions: list[dict[str, Any]],
    tolerance_ms: int = 5000,
) -> tuple[
    list[tuple[dict[str, Any], dict[str, Any]]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """Match positions between dry-run and live databases.

    Args:
        dryrun_positions: List of dry-run positions
        live_positions: List of live positions
        tolerance_ms: Maximum time difference in milliseconds for matching

    Returns:
        Tuple of (matched_pairs, unmatched_dryrun, unmatched_live)
    """
    # Group positions by mint
    dryrun_by_mint: dict[str, list[dict[str, Any]]] = defaultdict(list)
    live_by_mint: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for pos in dryrun_positions:
        dryrun_by_mint[pos["mint"]].append(pos)
    for pos in live_positions:
        live_by_mint[pos["mint"]].append(pos)

    matched_pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    used_dryrun: set[str] = set()  # Track used position IDs
    used_live: set[str] = set()

    # Match positions by mint and closest entry_ts
    for mint in set(dryrun_by_mint.keys()) & set(live_by_mint.keys()):
        dryrun_list = dryrun_by_mint[mint]
        live_list = live_by_mint[mint]

        # For each dryrun position, find closest live position
        for dryrun_pos in dryrun_list:
            if dryrun_pos["id"] in used_dryrun:
                continue

            best_match = None
            best_diff = float("inf")

            for live_pos in live_list:
                if live_pos["id"] in used_live:
                    continue

                time_diff = abs(dryrun_pos["entry_ts"] - live_pos["entry_ts"])
                if time_diff <= tolerance_ms and time_diff < best_diff:
                    best_match = live_pos
                    best_diff = time_diff

            if best_match:
                matched_pairs.append((dryrun_pos, best_match))
                used_dryrun.add(dryrun_pos["id"])
                used_live.add(best_match["id"])

    # Find unmatched positions
    unmatched_dryrun = [
        pos for pos in dryrun_positions if pos["id"] not in used_dryrun
    ]
    unmatched_live = [pos for pos in live_positions if pos["id"] not in used_live]

    return matched_pairs, unmatched_dryrun, unmatched_live


def match_trades(
    dryrun_trades: list[dict[str, Any]],
    live_trades: list[dict[str, Any]],
    position_mapping: dict[str, str],
    tolerance_ms: int = 2000,
) -> tuple[
    list[tuple[dict[str, Any], dict[str, Any]]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """Match trades between dry-run and live databases.

    Args:
        dryrun_trades: List of dry-run trades
        live_trades: List of live trades
        position_mapping: Mapping from dryrun position_id to live position_id
        tolerance_ms: Maximum time difference in milliseconds for matching

    Returns:
        Tuple of (matched_pairs, unmatched_dryrun, unmatched_live)
    """
    # Group trades by mint and trade_type (more flexible than position_id)
    dryrun_by_mint_type: dict[tuple[str, str], list[dict[str, Any]]] = (
        defaultdict(list)
    )
    live_by_mint_type: dict[tuple[str, str], list[dict[str, Any]]] = (
        defaultdict(list)
    )

    for trade in dryrun_trades:
        key = (trade["mint"], trade["trade_type"])
        dryrun_by_mint_type[key].append(trade)

    for trade in live_trades:
        key = (trade["mint"], trade["trade_type"])
        live_by_mint_type[key].append(trade)

    matched_pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    used_dryrun: set[tuple[str, int]] = set()  # (mint, timestamp) as unique key
    used_live: set[tuple[str, int]] = set()

    # Match trades by mint + trade_type + closest timestamp
    # This is more flexible than requiring position_id matches
    for key in set(dryrun_by_mint_type.keys()) & set(live_by_mint_type.keys()):
        dryrun_list = dryrun_by_mint_type[key]
        live_list = live_by_mint_type[key]

        for dryrun_trade in dryrun_list:
            trade_key = (dryrun_trade["mint"], dryrun_trade["timestamp"])
            if trade_key in used_dryrun:
                continue

            best_match = None
            best_diff = float("inf")

            for live_trade in live_list:
                live_key = (live_trade["mint"], live_trade["timestamp"])
                if live_key in used_live:
                    continue

                time_diff = abs(dryrun_trade["timestamp"] - live_trade["timestamp"])
                if time_diff <= tolerance_ms and time_diff < best_diff:
                    best_match = live_trade
                    best_diff = time_diff

            if best_match:
                matched_pairs.append((dryrun_trade, best_match))
                used_dryrun.add(trade_key)
                used_live.add((best_match["mint"], best_match["timestamp"]))

    # Find unmatched trades
    unmatched_dryrun = [
        trade
        for trade in dryrun_trades
        if (trade["mint"], trade["timestamp"]) not in used_dryrun
    ]
    unmatched_live = [
        trade
        for trade in live_trades
        if (trade["mint"], trade["timestamp"]) not in used_live
    ]

    return matched_pairs, unmatched_dryrun, unmatched_live


def calculate_position_differences(
    dryrun_pos: dict[str, Any], live_pos: dict[str, Any]
) -> dict[str, Any]:
    """Calculate differences between matched positions.

    Args:
        dryrun_pos: Dry-run position dictionary
        live_pos: Live position dictionary

    Returns:
        Dictionary with comparison metrics
    """
    def safe_diff(val1: Any, val2: Any) -> Any:
        """Calculate difference, handling None values."""
        if val1 is None or val2 is None:
            return None
        try:
            return val2 - val1  # live - dryrun
        except TypeError:
            return None

    def safe_pct_diff(val1: Any, val2: Any) -> Any:
        """Calculate percentage difference, handling None and zero values."""
        if val1 is None or val2 is None:
            return None
        try:
            if val1 == 0:
                return None if val2 == 0 else float("inf")
            return ((val2 - val1) / abs(val1)) * 100
        except (TypeError, ZeroDivisionError):
            return None

    entry_ts_diff = safe_diff(dryrun_pos["entry_ts"], live_pos["entry_ts"])
    exit_ts_diff = safe_diff(dryrun_pos["exit_ts"], live_pos["exit_ts"])
    duration_diff = (
        (exit_ts_diff - entry_ts_diff) / 1000.0
        if exit_ts_diff is not None and entry_ts_diff is not None
        else None
    )

    return {
        "mint": dryrun_pos["mint"],
        "dryrun_position_id": dryrun_pos["id"],
        "live_position_id": live_pos["id"],
        # Entry metrics
        "entry_ts_dryrun": dryrun_pos["entry_ts"],
        "entry_ts_live": live_pos["entry_ts"],
        "entry_ts_diff_ms": entry_ts_diff,
        "entry_price_dryrun": dryrun_pos["entry_net_price_decimal"],
        "entry_price_live": live_pos["entry_net_price_decimal"],
        "entry_price_diff": safe_diff(
            dryrun_pos["entry_net_price_decimal"],
            live_pos["entry_net_price_decimal"],
        ),
        "entry_price_pct_diff": safe_pct_diff(
            dryrun_pos["entry_net_price_decimal"],
            live_pos["entry_net_price_decimal"],
        ),
        # Exit metrics
        "exit_ts_dryrun": dryrun_pos["exit_ts"],
        "exit_ts_live": live_pos["exit_ts"],
        "exit_ts_diff_ms": exit_ts_diff,
        "exit_price_dryrun": dryrun_pos["exit_net_price_decimal"],
        "exit_price_live": live_pos["exit_net_price_decimal"],
        "exit_price_diff": safe_diff(
            dryrun_pos["exit_net_price_decimal"],
            live_pos["exit_net_price_decimal"],
        ),
        "exit_price_pct_diff": safe_pct_diff(
            dryrun_pos["exit_net_price_decimal"],
            live_pos["exit_net_price_decimal"],
        ),
        # Duration
        "duration_dryrun_s": (
            (dryrun_pos["exit_ts"] - dryrun_pos["entry_ts"]) / 1000.0
            if dryrun_pos["exit_ts"] and dryrun_pos["entry_ts"]
            else None
        ),
        "duration_live_s": (
            (live_pos["exit_ts"] - live_pos["entry_ts"]) / 1000.0
            if live_pos["exit_ts"] and live_pos["entry_ts"]
            else None
        ),
        "duration_diff_s": duration_diff,
        # PNL
        "pnl_dryrun": dryrun_pos["realized_pnl_sol_decimal"],
        "pnl_live": live_pos["realized_pnl_sol_decimal"],
        "pnl_diff": safe_diff(
            dryrun_pos["realized_pnl_sol_decimal"],
            live_pos["realized_pnl_sol_decimal"],
        ),
        "net_pnl_dryrun": dryrun_pos["realized_net_pnl_sol_decimal"],
        "net_pnl_live": live_pos["realized_net_pnl_sol_decimal"],
        "net_pnl_diff": safe_diff(
            dryrun_pos["realized_net_pnl_sol_decimal"],
            live_pos["realized_net_pnl_sol_decimal"],
        ),
        # Fees
        "transaction_fee_dryrun": dryrun_pos["transaction_fee_raw"],
        "transaction_fee_live": live_pos["transaction_fee_raw"],
        "transaction_fee_diff": safe_diff(
            dryrun_pos["transaction_fee_raw"], live_pos["transaction_fee_raw"]
        ),
        "platform_fee_dryrun": dryrun_pos["platform_fee_raw"],
        "platform_fee_live": live_pos["platform_fee_raw"],
        "platform_fee_diff": safe_diff(
            dryrun_pos["platform_fee_raw"], live_pos["platform_fee_raw"]
        ),
        "tip_fee_dryrun": dryrun_pos["tip_fee_raw"],
        "tip_fee_live": live_pos["tip_fee_raw"],
        "tip_fee_diff": safe_diff(dryrun_pos["tip_fee_raw"], live_pos["tip_fee_raw"]),
        "total_fee_dryrun": (
            (dryrun_pos["transaction_fee_raw"] or 0)
            + (dryrun_pos["platform_fee_raw"] or 0)
            + (dryrun_pos["tip_fee_raw"] or 0)
        ),
        "total_fee_live": (
            (live_pos["transaction_fee_raw"] or 0)
            + (live_pos["platform_fee_raw"] or 0)
            + (live_pos["tip_fee_raw"] or 0)
        ),
        "total_fee_diff": safe_diff(
            (dryrun_pos["transaction_fee_raw"] or 0)
            + (dryrun_pos["platform_fee_raw"] or 0)
            + (dryrun_pos["tip_fee_raw"] or 0),
            (live_pos["transaction_fee_raw"] or 0)
            + (live_pos["platform_fee_raw"] or 0)
            + (live_pos["tip_fee_raw"] or 0),
        ),
        # SOL amounts
        "sol_swapout_dryrun": dryrun_pos["total_sol_swapout_amount_raw"],
        "sol_swapout_live": live_pos["total_sol_swapout_amount_raw"],
        "sol_swapout_diff": safe_diff(
            dryrun_pos["total_sol_swapout_amount_raw"],
            live_pos["total_sol_swapout_amount_raw"],
        ),
        "sol_swapin_dryrun": dryrun_pos["total_sol_swapin_amount_raw"],
        "sol_swapin_live": live_pos["total_sol_swapin_amount_raw"],
        "sol_swapin_diff": safe_diff(
            dryrun_pos["total_sol_swapin_amount_raw"],
            live_pos["total_sol_swapin_amount_raw"],
        ),
        # Token amounts
        "token_swapin_dryrun": dryrun_pos["total_token_swapin_amount_raw"],
        "token_swapin_live": live_pos["total_token_swapin_amount_raw"],
        "token_swapin_diff": safe_diff(
            dryrun_pos["total_token_swapin_amount_raw"],
            live_pos["total_token_swapin_amount_raw"],
        ),
        "token_swapout_dryrun": dryrun_pos["total_token_swapout_amount_raw"],
        "token_swapout_live": live_pos["total_token_swapout_amount_raw"],
        "token_swapout_diff": safe_diff(
            dryrun_pos["total_token_swapout_amount_raw"],
            live_pos["total_token_swapout_amount_raw"],
        ),
        # Other metrics
        "highest_price_dryrun": dryrun_pos["highest_price"],
        "highest_price_live": live_pos["highest_price"],
        "highest_price_diff": safe_diff(
            dryrun_pos["highest_price"], live_pos["highest_price"]
        ),
        "exit_reason_dryrun": dryrun_pos["exit_reason"],
        "exit_reason_live": live_pos["exit_reason"],
        "exit_reason_match": dryrun_pos["exit_reason"] == live_pos["exit_reason"],
    }


def calculate_trade_differences(
    dryrun_trade: dict[str, Any], live_trade: dict[str, Any]
) -> dict[str, Any]:
    """Calculate differences between matched trades.

    Args:
        dryrun_trade: Dry-run trade dictionary
        live_trade: Live trade dictionary

    Returns:
        Dictionary with comparison metrics
    """
    def safe_diff(val1: Any, val2: Any) -> Any:
        """Calculate difference, handling None values."""
        if val1 is None or val2 is None:
            return None
        try:
            return val2 - val1  # live - dryrun
        except TypeError:
            return None

    return {
        "mint": dryrun_trade["mint"],
        "trade_type": dryrun_trade["trade_type"],
        "timestamp_dryrun": dryrun_trade["timestamp"],
        "timestamp_live": live_trade["timestamp"],
        "timestamp_diff_ms": safe_diff(
            dryrun_trade["timestamp"], live_trade["timestamp"]
        ),
        "success_dryrun": bool(dryrun_trade["success"]),
        "success_live": bool(live_trade["success"]),
        "success_match": dryrun_trade["success"] == live_trade["success"],
        "token_amount_dryrun": dryrun_trade["token_swap_amount_raw"],
        "token_amount_live": live_trade["token_swap_amount_raw"],
        "token_amount_diff": safe_diff(
            dryrun_trade["token_swap_amount_raw"],
            live_trade["token_swap_amount_raw"],
        ),
        "sol_amount_dryrun": dryrun_trade["net_sol_swap_amount_raw"],
        "sol_amount_live": live_trade["net_sol_swap_amount_raw"],
        "sol_amount_diff": safe_diff(
            dryrun_trade["net_sol_swap_amount_raw"],
            live_trade["net_sol_swap_amount_raw"],
        ),
        "price_dryrun": dryrun_trade["net_price_decimal"],
        "price_live": live_trade["net_price_decimal"],
        "price_diff": safe_diff(
            dryrun_trade["net_price_decimal"], live_trade["net_price_decimal"]
        ),
        "transaction_fee_dryrun": dryrun_trade["transaction_fee_raw"],
        "transaction_fee_live": live_trade["transaction_fee_raw"],
        "transaction_fee_diff": safe_diff(
            dryrun_trade["transaction_fee_raw"], live_trade["transaction_fee_raw"]
        ),
        "platform_fee_dryrun": dryrun_trade["platform_fee_raw"],
        "platform_fee_live": live_trade["platform_fee_raw"],
        "platform_fee_diff": safe_diff(
            dryrun_trade["platform_fee_raw"], live_trade["platform_fee_raw"]
        ),
        "tip_fee_dryrun": dryrun_trade["tip_fee_raw"],
        "tip_fee_live": live_trade["tip_fee_raw"],
        "tip_fee_diff": safe_diff(
            dryrun_trade["tip_fee_raw"], live_trade["tip_fee_raw"]
        ),
        "trade_duration_dryrun_ms": dryrun_trade["trade_duration_ms"],
        "trade_duration_live_ms": live_trade["trade_duration_ms"],
        "trade_duration_diff_ms": safe_diff(
            dryrun_trade["trade_duration_ms"], live_trade["trade_duration_ms"]
        ),
    }


def export_csv(
    position_comparisons: list[dict[str, Any]],
    trade_comparisons: list[dict[str, Any]],
    output_dir: Path,
) -> None:
    """Export comparison data to CSV files.

    Args:
        position_comparisons: List of position comparison dictionaries
        trade_comparisons: List of trade comparison dictionaries
        output_dir: Directory to save CSV files
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Export positions
    if position_comparisons:
        positions_file = output_dir / "position_comparison.csv"
        with open(positions_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=position_comparisons[0].keys())
            writer.writeheader()
            writer.writerows(position_comparisons)
        print(f"✓ Exported {len(position_comparisons)} position comparisons to {positions_file}")

    # Export trades
    if trade_comparisons:
        trades_file = output_dir / "trade_comparison.csv"
        with open(trades_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=trade_comparisons[0].keys())
            writer.writeheader()
            writer.writerows(trade_comparisons)
        print(f"✓ Exported {len(trade_comparisons)} trade comparisons to {trades_file}")


def generate_html_report(
    position_comparisons: list[dict[str, Any]],
    trade_comparisons: list[dict[str, Any]],
    unmatched_dryrun_positions: list[dict[str, Any]],
    unmatched_live_positions: list[dict[str, Any]],
    unmatched_dryrun_trades: list[dict[str, Any]],
    unmatched_live_trades: list[dict[str, Any]],
    summary_stats: dict[str, Any],
    output_dir: Path,
) -> None:
    """Generate interactive HTML report with visualizations.

    Args:
        position_comparisons: List of position comparison dictionaries
        trade_comparisons: List of trade comparison dictionaries
        unmatched_dryrun_positions: Unmatched dry-run positions
        unmatched_live_positions: Unmatched live positions
        unmatched_dryrun_trades: Unmatched dry-run trades
        unmatched_live_trades: Unmatched live trades
        summary_stats: Summary statistics dictionary
        output_dir: Directory to save HTML file
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    html_file = output_dir / "comparison_report.html"

    # Calculate cumulative PNL for both modes
    cumulative_pnl_dryrun = []
    cumulative_pnl_live = []
    cumulative_normalized_pnl_dryrun = []
    cumulative_normalized_pnl_live = []
    running_dryrun = 0.0
    running_live = 0.0
    running_normalized_dryrun = 0.0
    running_normalized_live = 0.0
    timestamps = []

    for comp in sorted(position_comparisons, key=lambda x: x["entry_ts_dryrun"] or 0):
        if comp["pnl_dryrun"] is not None:
            running_dryrun += comp["pnl_dryrun"]
        if comp["pnl_live"] is not None:
            running_live += comp["pnl_live"]
        cumulative_pnl_dryrun.append(running_dryrun)
        cumulative_pnl_live.append(running_live)
        
        # Calculate normalized PNL for this position
        investment_dryrun = abs(comp.get("sol_swapout_dryrun") or 0) / 1e9
        investment_live = abs(comp.get("sol_swapout_live") or 0) / 1e9
        if investment_dryrun > 0 and comp["pnl_dryrun"] is not None:
            normalized_pnl_dryrun = comp["pnl_dryrun"] / investment_dryrun
            running_normalized_dryrun += normalized_pnl_dryrun
        if investment_live > 0 and comp["pnl_live"] is not None:
            normalized_pnl_live = comp["pnl_live"] / investment_live
            running_normalized_live += normalized_pnl_live
        
        cumulative_normalized_pnl_dryrun.append(running_normalized_dryrun)
        cumulative_normalized_pnl_live.append(running_normalized_live)
        
        if comp["entry_ts_dryrun"]:
            timestamps.append(datetime.fromtimestamp(comp["entry_ts_dryrun"] / 1000.0))
        else:
            timestamps.append(None)

    # Create subplots - 2 rows, 3 columns to fit normalized PNL chart
    fig = make_subplots(
        rows=2,
        cols=3,
        subplot_titles=(
            "Cumulative PNL Comparison",
            "Cumulative Normalized PNL Comparison",
            "PNL Difference per Position",
            "Fee Comparison",
            "Entry/Exit Timing Differences",
            "",  # Empty subplot for spacing
        ),
        specs=[
            [{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}],
            [{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}],
        ],
    )

    # Cumulative PNL
    if timestamps and cumulative_pnl_dryrun:
        valid_timestamps = [ts for ts in timestamps if ts is not None]
        valid_dryrun = [
            pnl for pnl, ts in zip(cumulative_pnl_dryrun, timestamps) if ts is not None
        ]
        valid_live = [
            pnl for pnl, ts in zip(cumulative_pnl_live, timestamps) if ts is not None
        ]
        if valid_timestamps:
            fig.add_trace(
                go.Scatter(
                    x=valid_timestamps,
                    y=valid_dryrun,
                    mode="lines+markers",
                    name="Dry-run",
                    line=dict(color="blue"),
                    showlegend=True,
                ),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=valid_timestamps,
                    y=valid_live,
                    mode="lines+markers",
                    name="Live",
                    line=dict(color="red"),
                    showlegend=False,  # Only show legend once
                ),
                row=1,
                col=1,
            )

    # Cumulative Normalized PNL
    if timestamps and cumulative_normalized_pnl_dryrun:
        valid_timestamps = [ts for ts in timestamps if ts is not None]
        valid_normalized_dryrun = [
            pnl for pnl, ts in zip(cumulative_normalized_pnl_dryrun, timestamps) if ts is not None
        ]
        valid_normalized_live = [
            pnl for pnl, ts in zip(cumulative_normalized_pnl_live, timestamps) if ts is not None
        ]
        if valid_timestamps:
            fig.add_trace(
                go.Scatter(
                    x=valid_timestamps,
                    y=[p * 100 for p in valid_normalized_dryrun],  # Convert to percentage
                    mode="lines+markers",
                    name="Dry-run (Normalized)",
                    line=dict(color="blue", dash="dash"),
                    showlegend=True,
                ),
                row=1,
                col=2,
            )
            fig.add_trace(
                go.Scatter(
                    x=valid_timestamps,
                    y=[p * 100 for p in valid_normalized_live],  # Convert to percentage
                    mode="lines+markers",
                    name="Live (Normalized)",
                    line=dict(color="red", dash="dash"),
                    showlegend=False,
                ),
                row=1,
                col=2,
            )

    # PNL differences
    pnl_diffs = [
        comp["pnl_diff"] for comp in position_comparisons if comp["pnl_diff"] is not None
    ]
    if pnl_diffs:
        fig.add_trace(
            go.Bar(
                y=pnl_diffs,
                name="PNL Diff (Live - Dryrun)",
                marker_color=["green" if d > 0 else "red" for d in pnl_diffs],
                showlegend=False,
            ),
            row=1,
            col=3,
        )

    # Fee comparison
    total_fees_dryrun = [
        comp["total_fee_dryrun"] / 1e9
        for comp in position_comparisons
        if comp["total_fee_dryrun"] is not None
    ]
    total_fees_live = [
        comp["total_fee_live"] / 1e9
        for comp in position_comparisons
        if comp["total_fee_live"] is not None
    ]
    if total_fees_dryrun and total_fees_live:
        fig.add_trace(
            go.Bar(
                x=["Dry-run", "Live"],
                y=[sum(total_fees_dryrun), sum(total_fees_live)],
                name="Total Fees",
                marker_color=["blue", "red"],
                showlegend=False,
            ),
            row=2,
            col=1,
        )

    # Timing differences
    entry_diffs = [
        comp["entry_ts_diff_ms"] / 1000.0
        for comp in position_comparisons
        if comp["entry_ts_diff_ms"] is not None
    ]
    exit_diffs = [
        comp["exit_ts_diff_ms"] / 1000.0
        for comp in position_comparisons
        if comp["exit_ts_diff_ms"] is not None
    ]
    if entry_diffs or exit_diffs:
        fig.add_trace(
            go.Box(
                y=entry_diffs,
                name="Entry Time Diff (s)",
                boxmean="sd",
                showlegend=False,
            ),
            row=2,
            col=2,
        )
        fig.add_trace(
            go.Box(
                y=exit_diffs,
                name="Exit Time Diff (s)",
                boxmean="sd",
                showlegend=False,
            ),
            row=2,
            col=2,
        )
    
    # Update y-axis labels
    fig.update_yaxes(title_text="Cumulative PNL (SOL)", row=1, col=1)
    fig.update_yaxes(title_text="Cumulative Normalized PNL (%)", row=1, col=2)
    fig.update_yaxes(title_text="PNL Difference (SOL)", row=1, col=3)
    fig.update_yaxes(title_text="Total Fees (SOL)", row=2, col=1)
    fig.update_yaxes(title_text="Time Difference (seconds)", row=2, col=2)

    fig.update_layout(
        height=900,
        title_text="Dry-run vs Live Trading Comparison",
        showlegend=True,
        legend=dict(x=1.02, y=1),
    )

    # Convert figure to JSON for embedding
    plotly_json = fig.to_json()

    # Generate HTML tables for positions and trades
    def format_value(val, is_sol=False, is_percentage=False, decimals=6):
        """Format a value for HTML display."""
        if val is None:
            return "<span style='color: #999;'>N/A</span>"
        if is_percentage:
            return f"{val*100:.2f}%"
        if is_sol:
            return f"{val:.{decimals}f} SOL"
        return str(val)
    
    def generate_positions_table(comparisons):
        """Generate HTML table for position comparisons."""
        if not comparisons:
            return "<p>No matched positions.</p>"
        
        # Sort by exit timestamp (use dry-run exit_ts, fallback to live if not available)
        sorted_comparisons = sorted(
            comparisons,
            key=lambda x: x.get("exit_ts_dryrun") or x.get("exit_ts_live") or 0
        )
        
        html = """
        <table class="data-table">
            <thead>
                <tr>
                    <th>Mint</th>
                    <th>Entry Time (Dry-run)</th>
                    <th>Entry Time (Live)</th>
                    <th>Exit Time (Dry-run)</th>
                    <th>Exit Time (Live)</th>
                    <th>Exit Reason (Dry-run)</th>
                    <th>Exit Reason (Live)</th>
                    <th>Normalized PNL (Dry-run)</th>
                    <th>Normalized PNL (Live)</th>
                    <th>Cumulative Normalized PNL (Dry-run)</th>
                    <th>Cumulative Normalized PNL (Live)</th>
                    <th>Total Fees (Dry-run)</th>
                    <th>Total Fees (Live)</th>
                    <th>Entry Price (Dry-run)</th>
                    <th>Entry Price (Live)</th>
                    <th>Exit Price (Dry-run)</th>
                    <th>Exit Price (Live)</th>
                    <th>Duration (s)</th>
                    <th>PNL (Dry-run)</th>
                    <th>PNL (Live)</th>
                    <th>PNL Diff</th>
                    <th>Investment (Dry-run)</th>
                    <th>Investment (Live)</th>
                </tr>
            </thead>
            <tbody>
        """
        
        # Track cumulative normalized PNL
        cumulative_normalized_dryrun = 0.0
        cumulative_normalized_live = 0.0
        
        for comp in sorted_comparisons:
            mint_short = comp["mint"][:8] + "..." if len(comp["mint"]) > 8 else comp["mint"]
            
            # Entry times (with milliseconds)
            entry_ts_dryrun = comp.get("entry_ts_dryrun")
            entry_ts_live = comp.get("entry_ts_live")
            if entry_ts_dryrun:
                dt_dryrun = datetime.fromtimestamp(entry_ts_dryrun / 1000.0)
                ms_dryrun = entry_ts_dryrun % 1000
                entry_time_dryrun = f"{dt_dryrun.strftime('%Y-%m-%d %H:%M:%S')}.{ms_dryrun:03d}"
            else:
                entry_time_dryrun = "N/A"
            if entry_ts_live:
                dt_live = datetime.fromtimestamp(entry_ts_live / 1000.0)
                ms_live = entry_ts_live % 1000
                entry_time_live = f"{dt_live.strftime('%Y-%m-%d %H:%M:%S')}.{ms_live:03d}"
            else:
                entry_time_live = "N/A"
            
            # Exit times (with milliseconds)
            exit_ts_dryrun = comp.get("exit_ts_dryrun")
            exit_ts_live = comp.get("exit_ts_live")
            if exit_ts_dryrun:
                dt_dryrun = datetime.fromtimestamp(exit_ts_dryrun / 1000.0)
                ms_dryrun = exit_ts_dryrun % 1000
                exit_time_dryrun = f"{dt_dryrun.strftime('%Y-%m-%d %H:%M:%S')}.{ms_dryrun:03d}"
            else:
                exit_time_dryrun = "N/A"
            if exit_ts_live:
                dt_live = datetime.fromtimestamp(exit_ts_live / 1000.0)
                ms_live = exit_ts_live % 1000
                exit_time_live = f"{dt_live.strftime('%Y-%m-%d %H:%M:%S')}.{ms_live:03d}"
            else:
                exit_time_live = "N/A"
            
            # Entry prices
            entry_price_dryrun = comp.get("entry_price_dryrun")
            entry_price_live = comp.get("entry_price_live")
            
            # Exit prices
            exit_price_dryrun = comp.get("exit_price_dryrun")
            exit_price_live = comp.get("exit_price_live")
            
            duration = comp.get("duration_dryrun_s")
            
            investment_dryrun = abs(comp.get("sol_swapout_dryrun") or 0) / 1e9
            investment_live = abs(comp.get("sol_swapout_live") or 0) / 1e9
            normalized_dryrun = (comp.get("pnl_dryrun") or 0) / investment_dryrun if investment_dryrun > 0 else None
            normalized_live = (comp.get("pnl_live") or 0) / investment_live if investment_live > 0 else None
            
            # Update cumulative normalized PNL
            if normalized_dryrun is not None:
                cumulative_normalized_dryrun += normalized_dryrun
            if normalized_live is not None:
                cumulative_normalized_live += normalized_live
            
            total_fee_dryrun = (comp.get("total_fee_dryrun") or 0) / 1e9
            total_fee_live = (comp.get("total_fee_live") or 0) / 1e9
            
            pnl_diff = comp.get("pnl_diff") or 0
            pnl_diff_class = "positive" if pnl_diff > 0 else "negative" if pnl_diff < 0 else ""
            
            # Color classes for normalized PNL (not for cumulative)
            normalized_dryrun_class = "positive" if (normalized_dryrun or 0) > 0 else "negative" if (normalized_dryrun or 0) < 0 else ""
            normalized_live_class = "positive" if (normalized_live or 0) > 0 else "negative" if (normalized_live or 0) < 0 else ""
            
            html += f"""
                <tr>
                    <td title="{comp['mint']}">{mint_short}</td>
                    <td>{entry_time_dryrun}</td>
                    <td>{entry_time_live}</td>
                    <td>{exit_time_dryrun}</td>
                    <td>{exit_time_live}</td>
                    <td>{comp.get('exit_reason_dryrun') or 'N/A'}</td>
                    <td>{comp.get('exit_reason_live') or 'N/A'}</td>
                    <td class="pnl {normalized_dryrun_class}">{format_value(normalized_dryrun, is_percentage=True) if normalized_dryrun is not None else 'N/A'}</td>
                    <td class="pnl {normalized_live_class}">{format_value(normalized_live, is_percentage=True) if normalized_live is not None else 'N/A'}</td>
                    <td>{format_value(cumulative_normalized_dryrun, is_percentage=True)}</td>
                    <td>{format_value(cumulative_normalized_live, is_percentage=True)}</td>
                    <td>{format_value(total_fee_dryrun, is_sol=True)}</td>
                    <td>{format_value(total_fee_live, is_sol=True)}</td>
                    <td>{format_value(entry_price_dryrun, decimals=8) if entry_price_dryrun is not None else 'N/A'}</td>
                    <td>{format_value(entry_price_live, decimals=8) if entry_price_live is not None else 'N/A'}</td>
                    <td>{format_value(exit_price_dryrun, decimals=8) if exit_price_dryrun is not None else 'N/A'}</td>
                    <td>{format_value(exit_price_live, decimals=8) if exit_price_live is not None else 'N/A'}</td>
                    <td>{format_value(duration, decimals=1) if duration else 'N/A'}</td>
                    <td class="pnl">{format_value(comp.get('pnl_dryrun'), is_sol=True)}</td>
                    <td class="pnl">{format_value(comp.get('pnl_live'), is_sol=True)}</td>
                    <td class="pnl {pnl_diff_class}">{format_value(pnl_diff, is_sol=True)}</td>
                    <td>{format_value(investment_dryrun, is_sol=True)}</td>
                    <td>{format_value(investment_live, is_sol=True)}</td>
                </tr>
            """
        
        html += """
            </tbody>
        </table>
        """
        return html
    
    def generate_trades_table(comparisons):
        """Generate HTML table for trade comparisons."""
        if not comparisons:
            return "<p>No matched trades.</p>"
        
        html = """
        <table class="data-table">
            <thead>
                <tr>
                    <th>Mint</th>
                    <th>Type</th>
                    <th>Timestamp</th>
                    <th>Success (Dry-run)</th>
                    <th>Success (Live)</th>
                    <th>Token Amount (Dry-run)</th>
                    <th>Token Amount (Live)</th>
                    <th>SOL Amount (Dry-run)</th>
                    <th>SOL Amount (Live)</th>
                    <th>Price (Dry-run)</th>
                    <th>Price (Live)</th>
                    <th>Transaction Fee (Dry-run)</th>
                    <th>Transaction Fee (Live)</th>
                    <th>Platform Fee (Dry-run)</th>
                    <th>Platform Fee (Live)</th>
                    <th>Tip Fee (Dry-run)</th>
                    <th>Tip Fee (Live)</th>
                    <th>Duration (Dry-run, ms)</th>
                    <th>Duration (Live, ms)</th>
                </tr>
            </thead>
            <tbody>
        """
        
        for comp in comparisons:
            mint_short = comp["mint"][:8] + "..." if len(comp["mint"]) > 8 else comp["mint"]
            timestamp = comp.get("timestamp_dryrun")
            if timestamp:
                dt = datetime.fromtimestamp(timestamp / 1000.0)
                ms = timestamp % 1000
                time_str = f"{dt.strftime('%Y-%m-%d %H:%M:%S')}.{ms:03d}"
            else:
                time_str = "N/A"
            
            token_amount_dryrun = (comp.get("token_amount_dryrun") or 0) / 1e6  # Assuming 6 decimals
            token_amount_live = (comp.get("token_amount_live") or 0) / 1e6
            sol_amount_dryrun = (comp.get("sol_amount_dryrun") or 0) / 1e9
            sol_amount_live = (comp.get("sol_amount_live") or 0) / 1e9
            
            tx_fee_dryrun = (comp.get("transaction_fee_dryrun") or 0) / 1e9
            tx_fee_live = (comp.get("transaction_fee_live") or 0) / 1e9
            platform_fee_dryrun = (comp.get("platform_fee_dryrun") or 0) / 1e9
            platform_fee_live = (comp.get("platform_fee_live") or 0) / 1e9
            tip_fee_dryrun = (comp.get("tip_fee_dryrun") or 0) / 1e9
            tip_fee_live = (comp.get("tip_fee_live") or 0) / 1e9
            
            success_dryrun = comp.get("success_dryrun", False)
            success_live = comp.get("success_live", False)
            success_match = comp.get("success_match", False)
            
            html += f"""
                <tr>
                    <td title="{comp['mint']}">{mint_short}</td>
                    <td>{comp.get('trade_type', 'N/A')}</td>
                    <td>{time_str}</td>
                    <td class="{'success' if success_dryrun else 'failure'}">{'✓' if success_dryrun else '✗'}</td>
                    <td class="{'success' if success_live else 'failure'}">{'✓' if success_live else '✗'}</td>
                    <td>{format_value(token_amount_dryrun, decimals=6)}</td>
                    <td>{format_value(token_amount_live, decimals=6)}</td>
                    <td>{format_value(sol_amount_dryrun, is_sol=True)}</td>
                    <td>{format_value(sol_amount_live, is_sol=True)}</td>
                    <td>{format_value(comp.get('price_dryrun'), decimals=8)}</td>
                    <td>{format_value(comp.get('price_live'), decimals=8)}</td>
                    <td>{format_value(tx_fee_dryrun, is_sol=True)}</td>
                    <td>{format_value(tx_fee_live, is_sol=True)}</td>
                    <td>{format_value(platform_fee_dryrun, is_sol=True)}</td>
                    <td>{format_value(platform_fee_live, is_sol=True)}</td>
                    <td>{format_value(tip_fee_dryrun, is_sol=True)}</td>
                    <td>{format_value(tip_fee_live, is_sol=True)}</td>
                    <td>{format_value(comp.get('trade_duration_dryrun_ms'), decimals=0) if comp.get('trade_duration_dryrun_ms') else 'N/A'}</td>
                    <td>{format_value(comp.get('trade_duration_live_ms'), decimals=0) if comp.get('trade_duration_live_ms') else 'N/A'}</td>
                </tr>
            """
        
        html += """
            </tbody>
        </table>
        """
        return html

    # Generate HTML with embedded data
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Dry-run vs Live Comparison Report</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .summary {{ background: #f0f0f0; padding: 20px; border-radius: 5px; margin-bottom: 20px; }}
        .section {{ margin: 20px 0; }}
        .data-table {{ border-collapse: collapse; width: 100%; margin: 20px 0; font-size: 12px; }}
        .data-table th, .data-table td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        .data-table th {{ background-color: #4CAF50; color: white; position: sticky; top: 0; z-index: 10; }}
        .data-table tbody tr:nth-child(even) {{ background-color: #f9f9f9; }}
        .data-table tbody tr:hover {{ background-color: #f5f5f5; }}
        .pnl.positive {{ color: green; font-weight: bold; }}
        .pnl.negative {{ color: red; font-weight: bold; }}
        .success {{ color: green; font-weight: bold; text-align: center; }}
        .failure {{ color: red; font-weight: bold; text-align: center; }}
        .table-container {{ overflow-x: auto; max-height: 600px; overflow-y: auto; }}
        h2 {{ margin-top: 30px; border-bottom: 2px solid #4CAF50; padding-bottom: 5px; }}
    </style>
</head>
<body>
    <h1>Dry-run vs Live Trading Comparison Report</h1>
    <div class="summary">
        <h2>Summary Statistics</h2>
        <ul>
            <li>Total Positions Matched: {summary_stats['matched_positions']}</li>
            <li>Total PNL Difference: {summary_stats['total_pnl_diff']:.6f} SOL</li>
            <li>Average PNL per Position (Dry-run): {summary_stats['avg_pnl_dryrun']:.6f} SOL</li>
            <li>Average PNL per Position (Live): {summary_stats['avg_pnl_live']:.6f} SOL</li>
            <li>Total Fee Difference: {summary_stats['total_fee_diff']:.6f} SOL</li>
            <li>Unmatched Dry-run Positions: {summary_stats['unmatched_dryrun_positions']}</li>
            <li>Unmatched Live Positions: {summary_stats['unmatched_live_positions']}</li>
        </ul>
    </div>
    
    <div id="plotly-chart"></div>
    <script>
        var plotlyData = {plotly_json};
        Plotly.newPlot('plotly-chart', plotlyData.data, plotlyData.layout);
    </script>
    
    <h2>Position Comparisons</h2>
    <div class="table-container">
        {generate_positions_table(position_comparisons)}
    </div>
    
    <h2>Trade Comparisons</h2>
    <div class="table-container">
        {generate_trades_table(trade_comparisons)}
    </div>
</body>
</html>
    """

    with open(html_file, "w") as f:
        f.write(html_content)

    print(f"✓ Generated HTML report: {html_file}")


def print_console_summary(
    position_comparisons: list[dict[str, Any]],
    trade_comparisons: list[dict[str, Any]],
    unmatched_dryrun_positions: list[dict[str, Any]],
    unmatched_live_positions: list[dict[str, Any]],
    unmatched_dryrun_trades: list[dict[str, Any]],
    unmatched_live_trades: list[dict[str, Any]],
) -> dict[str, Any]:
    """Print console summary and return statistics.

    Args:
        position_comparisons: List of position comparison dictionaries
        trade_comparisons: List of trade comparison dictionaries
        unmatched_dryrun_positions: Unmatched dry-run positions
        unmatched_live_positions: Unmatched live positions
        unmatched_dryrun_trades: Unmatched dry-run trades
        unmatched_live_trades: Unmatched live trades

    Returns:
        Summary statistics dictionary
    """
    print("\n" + "=" * 80)
    print("DRY-RUN vs LIVE TRADING COMPARISON SUMMARY")
    print("=" * 80)

    # Calculate summary statistics
    total_pnl_dryrun = sum(
        comp["pnl_dryrun"] or 0 for comp in position_comparisons
    )
    total_pnl_live = sum(comp["pnl_live"] or 0 for comp in position_comparisons)
    total_pnl_diff = total_pnl_live - total_pnl_dryrun

    # Calculate normalized PNL (PNL / investment amount)
    # Use total_net_sol_swapout_amount_raw which is the net SOL spent on buys (excluding fees)
    # This represents the actual investment in tokens
    total_investment_dryrun = sum(
        abs(comp.get("sol_swapout_dryrun") or 0) / 1e9 for comp in position_comparisons
    )
    total_investment_live = sum(
        abs(comp.get("sol_swapout_live") or 0) / 1e9 for comp in position_comparisons
    )
    
    # Debug: Show investment calculation details
    if position_comparisons:
        print(f"\n   Investment Calculation Debug:")
        print(f"      Number of matched positions: {len(position_comparisons)}")
        sample_comp = position_comparisons[0]
        print(f"      Sample position - sol_swapout_dryrun: {sample_comp.get('sol_swapout_dryrun')} lamports = {abs(sample_comp.get('sol_swapout_dryrun') or 0) / 1e9:.6f} SOL")
        print(f"      Sample position - sol_swapout_live: {sample_comp.get('sol_swapout_live')} lamports = {abs(sample_comp.get('sol_swapout_live') or 0) / 1e9:.6f} SOL")
        print(f"      Total investment (dryrun): {total_investment_dryrun:.6f} SOL")
        print(f"      Total investment (live): {total_investment_live:.6f} SOL")
    normalized_pnl_dryrun = (
        total_pnl_dryrun / total_investment_dryrun if total_investment_dryrun > 0 else 0.0
    )
    normalized_pnl_live = (
        total_pnl_live / total_investment_live if total_investment_live > 0 else 0.0
    )
    normalized_pnl_diff = normalized_pnl_live - normalized_pnl_dryrun

    # Calculate fee breakdowns
    total_transaction_fee_dryrun = sum(
        (comp["transaction_fee_dryrun"] or 0) / 1e9 for comp in position_comparisons
    )
    total_transaction_fee_live = sum(
        (comp["transaction_fee_live"] or 0) / 1e9 for comp in position_comparisons
    )
    total_platform_fee_dryrun = sum(
        (comp["platform_fee_dryrun"] or 0) / 1e9 for comp in position_comparisons
    )
    total_platform_fee_live = sum(
        (comp["platform_fee_live"] or 0) / 1e9 for comp in position_comparisons
    )
    total_tip_fee_dryrun = sum(
        (comp["tip_fee_dryrun"] or 0) / 1e9 for comp in position_comparisons
    )
    total_tip_fee_live = sum(
        (comp["tip_fee_live"] or 0) / 1e9 for comp in position_comparisons
    )
    
    total_fee_dryrun = (
        total_transaction_fee_dryrun + total_platform_fee_dryrun + total_tip_fee_dryrun
    )
    total_fee_live = (
        total_transaction_fee_live + total_platform_fee_live + total_tip_fee_live
    )
    total_fee_diff = total_fee_live - total_fee_dryrun

    avg_pnl_dryrun = (
        total_pnl_dryrun / len(position_comparisons)
        if position_comparisons
        else 0.0
    )
    avg_pnl_live = (
        total_pnl_live / len(position_comparisons) if position_comparisons else 0.0
    )

    print(f"\n📊 Position Statistics:")
    print(f"   Matched Positions: {len(position_comparisons)}")
    print(f"   Unmatched Dry-run Positions: {len(unmatched_dryrun_positions)}")
    print(f"   Unmatched Live Positions: {len(unmatched_live_positions)}")

    print(f"\n💰 PNL Comparison (Matched Positions):")
    print(f"   Total PNL (Dry-run): {total_pnl_dryrun:.6f} SOL")
    print(f"   Total PNL (Live): {total_pnl_live:.6f} SOL")
    print(f"   PNL Difference: {total_pnl_diff:.6f} SOL ({total_pnl_diff/total_pnl_dryrun*100:.2f}%)" if total_pnl_dryrun != 0 else "   PNL Difference: N/A")
    print(f"   Average PNL per Position (Dry-run): {avg_pnl_dryrun:.6f} SOL")
    print(f"   Average PNL per Position (Live): {avg_pnl_live:.6f} SOL")
    print(f"\n   Normalized PNL (PNL / Investment):")
    print(f"      Dry-run: {normalized_pnl_dryrun*100:.2f}% (PNL: {total_pnl_dryrun:.6f} SOL / Investment: {total_investment_dryrun:.6f} SOL)")
    print(f"      Live: {normalized_pnl_live*100:.2f}% (PNL: {total_pnl_live:.6f} SOL / Investment: {total_investment_live:.6f} SOL)")
    print(f"      Difference: {normalized_pnl_diff*100:.2f}%")

    # Calculate unmatched position statistics
    unmatched_dryrun_pnl = sum(
        (pos.get("realized_pnl_sol_decimal") or 0) for pos in unmatched_dryrun_positions
    )
    unmatched_live_pnl = sum(
        (pos.get("realized_pnl_sol_decimal") or 0) for pos in unmatched_live_positions
    )
    unmatched_dryrun_investment = sum(
        abs(pos.get("total_sol_swapout_amount_raw") or 0) / 1e9
        for pos in unmatched_dryrun_positions
    )
    unmatched_live_investment = sum(
        abs(pos.get("total_sol_swapout_amount_raw") or 0) / 1e9
        for pos in unmatched_live_positions
    )
    unmatched_dryrun_transaction_fees = sum(
        (pos.get("transaction_fee_raw") or 0) / 1e9
        for pos in unmatched_dryrun_positions
    )
    unmatched_live_transaction_fees = sum(
        (pos.get("transaction_fee_raw") or 0) / 1e9
        for pos in unmatched_live_positions
    )
    unmatched_dryrun_platform_fees = sum(
        (pos.get("platform_fee_raw") or 0) / 1e9
        for pos in unmatched_dryrun_positions
    )
    unmatched_live_platform_fees = sum(
        (pos.get("platform_fee_raw") or 0) / 1e9
        for pos in unmatched_live_positions
    )
    unmatched_dryrun_tip_fees = sum(
        (pos.get("tip_fee_raw") or 0) / 1e9
        for pos in unmatched_dryrun_positions
    )
    unmatched_live_tip_fees = sum(
        (pos.get("tip_fee_raw") or 0) / 1e9
        for pos in unmatched_live_positions
    )
    unmatched_dryrun_fees = (
        unmatched_dryrun_transaction_fees + unmatched_dryrun_platform_fees + unmatched_dryrun_tip_fees
    )
    unmatched_live_fees = (
        unmatched_live_transaction_fees + unmatched_live_platform_fees + unmatched_live_tip_fees
    )
    
    avg_unmatched_dryrun_pnl = (
        unmatched_dryrun_pnl / len(unmatched_dryrun_positions)
        if unmatched_dryrun_positions
        else 0.0
    )
    avg_unmatched_live_pnl = (
        unmatched_live_pnl / len(unmatched_live_positions)
        if unmatched_live_positions
        else 0.0
    )
    normalized_unmatched_dryrun_pnl = (
        unmatched_dryrun_pnl / unmatched_dryrun_investment
        if unmatched_dryrun_investment > 0
        else 0.0
    )
    normalized_unmatched_live_pnl = (
        unmatched_live_pnl / unmatched_live_investment
        if unmatched_live_investment > 0
        else 0.0
    )

    if unmatched_dryrun_positions or unmatched_live_positions:
        print(f"\n💰 PNL Comparison (Unmatched Positions):")
        if unmatched_dryrun_positions:
            print(f"   Unmatched Dry-run Positions:")
            print(f"      Total PNL: {unmatched_dryrun_pnl:.6f} SOL")
            print(f"      Average PNL per Position: {avg_unmatched_dryrun_pnl:.6f} SOL")
            print(f"      Normalized PNL: {normalized_unmatched_dryrun_pnl*100:.2f}%")
            print(f"      Total Fees: {unmatched_dryrun_fees:.6f} SOL")
        if unmatched_live_positions:
            print(f"   Unmatched Live Positions:")
            print(f"      Total PNL: {unmatched_live_pnl:.6f} SOL")
            print(f"      Average PNL per Position: {avg_unmatched_live_pnl:.6f} SOL")
            print(f"      Normalized PNL: {normalized_unmatched_live_pnl*100:.2f}%")
            print(f"      Total Fees: {unmatched_live_fees:.6f} SOL")
        if unmatched_dryrun_positions and unmatched_live_positions:
            unmatched_pnl_diff = unmatched_live_pnl - unmatched_dryrun_pnl
            print(f"   Unmatched PNL Difference (Live - Dry-run): {unmatched_pnl_diff:.6f} SOL")

    print(f"\n💸 Fee Comparison (Matched Positions):")
    print(f"   Transaction Fees:")
    print(f"      Dry-run: {total_transaction_fee_dryrun:.6f} SOL")
    print(f"      Live: {total_transaction_fee_live:.6f} SOL")
    print(f"      Difference: {total_transaction_fee_live - total_transaction_fee_dryrun:.6f} SOL")
    print(f"   Platform Fees:")
    print(f"      Dry-run: {total_platform_fee_dryrun:.6f} SOL")
    print(f"      Live: {total_platform_fee_live:.6f} SOL")
    print(f"      Difference: {total_platform_fee_live - total_platform_fee_dryrun:.6f} SOL")
    print(f"   Tip Fees:")
    print(f"      Dry-run: {total_tip_fee_dryrun:.6f} SOL")
    print(f"      Live: {total_tip_fee_live:.6f} SOL")
    print(f"      Difference: {total_tip_fee_live - total_tip_fee_dryrun:.6f} SOL")
    print(f"   Total Fees:")
    print(f"      Dry-run: {total_fee_dryrun:.6f} SOL")
    print(f"      Live: {total_fee_live:.6f} SOL")
    print(f"      Difference: {total_fee_diff:.6f} SOL")

    # Find positions with largest PNL discrepancies
    if position_comparisons:
        sorted_by_diff = sorted(
            position_comparisons,
            key=lambda x: abs(x["pnl_diff"]) if x["pnl_diff"] is not None else 0,
            reverse=True,
        )
        print(f"\n🔍 Top 5 Positions with Largest PNL Differences:")
        for i, comp in enumerate(sorted_by_diff[:5], 1):
            mint_short = comp["mint"][:8] + "..." if len(comp["mint"]) > 8 else comp["mint"]
            pnl_diff = comp["pnl_diff"] or 0
            transaction_fee_dryrun = (comp.get("transaction_fee_dryrun") or 0) / 1e9
            transaction_fee_live = (comp.get("transaction_fee_live") or 0) / 1e9
            platform_fee_dryrun = (comp.get("platform_fee_dryrun") or 0) / 1e9
            platform_fee_live = (comp.get("platform_fee_live") or 0) / 1e9
            tip_fee_dryrun = (comp.get("tip_fee_dryrun") or 0) / 1e9
            tip_fee_live = (comp.get("tip_fee_live") or 0) / 1e9
            total_fee_dryrun = transaction_fee_dryrun + platform_fee_dryrun + tip_fee_dryrun
            total_fee_live = transaction_fee_live + platform_fee_live + tip_fee_live
            fee_diff = total_fee_live - total_fee_dryrun
            
            print(
                f"   {i}. {mint_short}: PNL diff = {pnl_diff:.6f} SOL "
                f"(Dry-run: {comp['pnl_dryrun']:.6f}, Live: {comp['pnl_live']:.6f})"
            )
            print(
                f"      Fees: Dry-run = {total_fee_dryrun:.6f} SOL "
                f"(tx: {transaction_fee_dryrun:.6f}, platform: {platform_fee_dryrun:.6f}, tip: {tip_fee_dryrun:.6f})"
            )
            print(
                f"             Live = {total_fee_live:.6f} SOL "
                f"(tx: {transaction_fee_live:.6f}, platform: {platform_fee_live:.6f}, tip: {tip_fee_live:.6f})"
            )
            print(f"      Fee diff = {fee_diff:.6f} SOL")

    print(f"\n📈 Trade Statistics:")
    print(f"   Matched Trades: {len(trade_comparisons)}")
    print(f"   Unmatched Dry-run Trades: {len(unmatched_dryrun_trades)}")
    print(f"   Unmatched Live Trades: {len(unmatched_live_trades)}")

    summary_stats = {
        "matched_positions": len(position_comparisons),
        "matched_trades": len(trade_comparisons),
        "total_pnl_dryrun": total_pnl_dryrun,
        "total_pnl_live": total_pnl_live,
        "total_pnl_diff": total_pnl_diff,
        "avg_pnl_dryrun": avg_pnl_dryrun,
        "avg_pnl_live": avg_pnl_live,
        "total_fee_dryrun": total_fee_dryrun,
        "total_fee_live": total_fee_live,
        "total_fee_diff": total_fee_diff,
        "unmatched_dryrun_positions": len(unmatched_dryrun_positions),
        "unmatched_live_positions": len(unmatched_live_positions),
        "unmatched_dryrun_trades": len(unmatched_dryrun_trades),
        "unmatched_live_trades": len(unmatched_live_trades),
        "unmatched_dryrun_pnl": unmatched_dryrun_pnl,
        "unmatched_live_pnl": unmatched_live_pnl,
        "unmatched_dryrun_fees": unmatched_dryrun_fees,
        "unmatched_live_fees": unmatched_live_fees,
        "avg_unmatched_dryrun_pnl": avg_unmatched_dryrun_pnl,
        "avg_unmatched_live_pnl": avg_unmatched_live_pnl,
    }

    return summary_stats


def main():
    """Main function to run the comparison."""
    parser = argparse.ArgumentParser(
        description="Compare dry-run and live trading results"
    )
    parser.add_argument(
        "dryrun_db", type=str, help="Path to dry-run database file"
    )
    parser.add_argument("live_db", type=str, help="Path to live database file")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="comparison_output",
        help="Output directory for CSV and HTML files (default: comparison_output)",
    )
    parser.add_argument(
        "--position-tolerance",
        type=int,
        default=5000,
        help="Position matching tolerance in milliseconds (default: 5000)",
    )
    parser.add_argument(
        "--trade-tolerance",
        type=int,
        default=2000,
        help="Trade matching tolerance in milliseconds (default: 2000)",
    )
    parser.add_argument(
        "--start-timestamp",
        type=str,
        default=None,
        help="Start timestamp as 'YYYY-MM-DD HH:MM:SS' or Unix epoch milliseconds (filters positions/trades with entry_ts >= start_timestamp)",
    )
    parser.add_argument(
        "--end-timestamp",
        type=str,
        default=None,
        help="End timestamp as 'YYYY-MM-DD HH:MM:SS' or Unix epoch milliseconds (filters positions/trades with entry_ts <= end_timestamp)",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Optional path to save console output to a log file (default: output only to console)",
    )

    args = parser.parse_args()

    # Set up logging to file if requested
    log_file = None
    if args.log_file:
        log_file = open(args.log_file, "w", encoding="utf-8")
        # Create a wrapper to write to both console and file
        class TeeOutput:
            def __init__(self, *files):
                self.files = files
            def write(self, obj):
                for f in self.files:
                    f.write(obj)
                    f.flush()
            def flush(self):
                for f in self.files:
                    f.flush()
        
        import sys
        sys.stdout = TeeOutput(sys.stdout, log_file)
        sys.stderr = TeeOutput(sys.stderr, log_file)
        print(f"Logging output to: {args.log_file}")

    # Validate database files
    dryrun_path = Path(args.dryrun_db)
    live_path = Path(args.live_db)

    if not dryrun_path.exists():
        print(f"Error: Dry-run database not found: {dryrun_path}")
        sys.exit(1)

    if not live_path.exists():
        print(f"Error: Live database not found: {live_path}")
        sys.exit(1)

    # Parse timestamps
    try:
        start_timestamp_ms = parse_timestamp(args.start_timestamp)
        end_timestamp_ms = parse_timestamp(args.end_timestamp)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    print(f"Loading dry-run database: {dryrun_path}")
    print(f"Loading live database: {live_path}")

    # Display timestamp filter info if provided
    if start_timestamp_ms or end_timestamp_ms:
        print("\nTimestamp filters:")
        if start_timestamp_ms:
            start_dt = datetime.fromtimestamp(start_timestamp_ms / 1000.0)
            print(f"  Start: {start_timestamp_ms} ms ({start_dt})")
        if end_timestamp_ms:
            end_dt = datetime.fromtimestamp(end_timestamp_ms / 1000.0)
            print(f"  End: {end_timestamp_ms} ms ({end_dt})")

    # Query positions and trades
    print("\nQuerying positions and trades...")
    dryrun_positions = query_all_positions(
        str(dryrun_path), start_timestamp_ms, end_timestamp_ms
    )
    live_positions = query_all_positions(
        str(live_path), start_timestamp_ms, end_timestamp_ms
    )
    dryrun_trades = query_all_trades(
        str(dryrun_path), start_timestamp_ms, end_timestamp_ms
    )
    live_trades = query_all_trades(
        str(live_path), start_timestamp_ms, end_timestamp_ms
    )

    print(f"Found {len(dryrun_positions)} dry-run positions, {len(live_positions)} live positions")
    print(f"Found {len(dryrun_trades)} dry-run trades, {len(live_trades)} live trades")
    
    # Debug: Show timestamp ranges of positions
    if dryrun_positions:
        dryrun_entry_times = [pos["entry_ts"] for pos in dryrun_positions if pos.get("entry_ts")]
        if dryrun_entry_times:
            min_dryrun_ts = min(dryrun_entry_times)
            max_dryrun_ts = max(dryrun_entry_times)
            print(f"   Dry-run position entry_ts range: {min_dryrun_ts} ({datetime.fromtimestamp(min_dryrun_ts/1000.0)}) to {max_dryrun_ts} ({datetime.fromtimestamp(max_dryrun_ts/1000.0)})")
    if live_positions:
        live_entry_times = [pos["entry_ts"] for pos in live_positions if pos.get("entry_ts")]
        if live_entry_times:
            min_live_ts = min(live_entry_times)
            max_live_ts = max(live_entry_times)
            print(f"   Live position entry_ts range: {min_live_ts} ({datetime.fromtimestamp(min_live_ts/1000.0)}) to {max_live_ts} ({datetime.fromtimestamp(max_live_ts/1000.0)})")

    # Match positions
    print("\nMatching positions...")
    matched_positions, unmatched_dryrun_pos, unmatched_live_pos = match_positions(
        dryrun_positions, live_positions, tolerance_ms=args.position_tolerance
    )
    print(f"Matched {len(matched_positions)} positions")

    # Create position mapping for trade matching (for reference, but not strictly required)
    position_mapping = {
        dryrun_pos["id"]: live_pos["id"]
        for dryrun_pos, live_pos in matched_positions
    }

    # Match trades (now uses mint + trade_type + timestamp, more flexible)
    print("Matching trades...")
    matched_trades, unmatched_dryrun_trade, unmatched_live_trade = match_trades(
        dryrun_trades,
        live_trades,
        position_mapping,
        tolerance_ms=args.trade_tolerance,
    )
    print(f"Matched {len(matched_trades)} trades")
    
    # Debug: Show some statistics about trade matching
    if len(matched_trades) == 0 and (len(dryrun_trades) > 0 or len(live_trades) > 0):
        print("\n⚠️  Warning: No trades matched despite having trades in databases")
        print(f"   Dry-run trades by type: {sum(1 for t in dryrun_trades if t['trade_type'] == 'buy')} buys, {sum(1 for t in dryrun_trades if t['trade_type'] == 'sell')} sells")
        print(f"   Live trades by type: {sum(1 for t in live_trades if t['trade_type'] == 'buy')} buys, {sum(1 for t in live_trades if t['trade_type'] == 'sell')} sells")
        
        # Check if there are common mints
        dryrun_mints = set(t["mint"] for t in dryrun_trades)
        live_mints = set(t["mint"] for t in live_trades)
        common_mints = dryrun_mints & live_mints
        print(f"   Common mints between trades: {len(common_mints)}")
        
        if common_mints:
            # Check timestamp ranges
            sample_mint = list(common_mints)[0]
            dryrun_times = [t["timestamp"] for t in dryrun_trades if t["mint"] == sample_mint]
            live_times = [t["timestamp"] for t in live_trades if t["mint"] == sample_mint]
            if dryrun_times and live_times:
                min_diff = min(abs(dt - lt) for dt in dryrun_times for lt in live_times)
                print(f"   Sample mint '{sample_mint[:8]}...': min timestamp diff = {min_diff} ms (tolerance = {args.trade_tolerance} ms)")

    # Calculate differences
    print("\nCalculating differences...")
    position_comparisons = [
        calculate_position_differences(dryrun_pos, live_pos)
        for dryrun_pos, live_pos in matched_positions
    ]
    trade_comparisons = [
        calculate_trade_differences(dryrun_trade, live_trade)
        for dryrun_trade, live_trade in matched_trades
    ]

    # Print console summary
    summary_stats = print_console_summary(
        position_comparisons,
        trade_comparisons,
        unmatched_dryrun_pos,
        unmatched_live_pos,
        unmatched_dryrun_trade,
        unmatched_live_trade,
    )

    # Export CSV
    output_dir = Path(args.output_dir)
    print(f"\nExporting results to {output_dir}...")
    export_csv(position_comparisons, trade_comparisons, output_dir)

    # Generate HTML report
    print("Generating HTML report...")
    generate_html_report(
        position_comparisons,
        trade_comparisons,
        unmatched_dryrun_pos,
        unmatched_live_pos,
        unmatched_dryrun_trade,
        unmatched_live_trade,
        summary_stats,
        output_dir,
    )

    print("\n✓ Comparison complete!")
    
    # Close log file if opened (restore stdout/stderr first)
    if args.log_file and log_file:
        import sys
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        log_file.close()
        print(f"✓ Log saved to: {args.log_file}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

