"""
Flask web server for PNL visualization dashboard.

This server provides an interactive web dashboard for visualizing cumulative PNL
from trading bot databases. It automatically discovers all databases in the data/
directory and serves them via a web interface.
"""

import time
from pathlib import Path

from flask import Flask, jsonify, render_template, request
from flask_compress import Compress  # type: ignore

from ui.shared import process_database

# Set template folder to ui/templates (relative to this file)
app = Flask(__name__, template_folder=str(Path(__file__).parent / "templates"))
Compress(app)  # Enable gzip compression

# Path to data directory (relative to project root)
DATA_DIR = Path(__file__).parent.parent / "data"

# Project root directory (parent of ui/)
PROJECT_ROOT = Path(__file__).parent.parent

# Default start timestamp (set when bot_runner starts)
# This will be the bot_runner start time in milliseconds
_default_start_timestamp: int | None = None

# Set of database paths for currently running bots
_running_bot_databases: set[str] = set()


def set_default_start_timestamp(timestamp_ms: int) -> None:
    """Set the default start timestamp (called from bot_runner when it starts).

    Args:
        timestamp_ms: Unix epoch milliseconds when bot_runner started
    """
    global _default_start_timestamp
    _default_start_timestamp = timestamp_ms


def register_running_bot_database(db_path: str) -> None:
    """Register a database path for a running bot.

    Args:
        db_path: Path to the database file (relative or absolute)
    """
    global _running_bot_databases
    # Normalize path to absolute for consistent matching
    # If relative, resolve relative to project root (same as DATA_DIR)
    # If absolute, use as-is
    if Path(db_path).is_absolute():
        abs_path = str(Path(db_path).resolve())
    else:
        # Resolve relative to project root to match how DATA_DIR resolves paths
        abs_path = str((PROJECT_ROOT / db_path).resolve())
    
    _running_bot_databases.add(abs_path)
    
    # Debug logging
    import logging
    logger = logging.getLogger(__name__)
    logger.info(f"Registered database: '{db_path}' -> '{abs_path}' (total registered: {len(_running_bot_databases)})")


def unregister_running_bot_database(db_path: str) -> None:
    """Unregister a database path when a bot stops.

    Args:
        db_path: Path to the database file (relative or absolute)
    """
    global _running_bot_databases
    abs_path = str(Path(db_path).resolve())
    _running_bot_databases.discard(abs_path)


def find_databases(only_running: bool = True) -> list[str]:
    """Find database files in the data directory.

    Args:
        only_running: If True, only return databases from running bots.
                     If False, return all databases in the data directory.

    Returns:
        List of database file paths
    """
    if not DATA_DIR.exists():
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(f"Data directory does not exist: {DATA_DIR}")
        return []
    
    # Resolve all database paths to absolute paths for consistent matching
    all_databases = sorted([str(db.resolve()) for db in DATA_DIR.glob("*.db")])
    
    # Filter to only running bot databases if requested
    if only_running:
        if not _running_bot_databases:
            # If no running bots are registered, return empty list
            import logging
            logger = logging.getLogger(__name__)
            logger.info("No running bot databases registered")
            return []
        
        # Normalize registered paths and match against found databases
        # Both should already be absolute paths, but ensure consistency
        registered_abs = {str(Path(p).resolve()) for p in _running_bot_databases}
        
        # Debug logging
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"Registered databases (normalized): {list(registered_abs)}")
        logger.info(f"Found databases in {DATA_DIR}: {all_databases}")
        
        matched = [db for db in all_databases if db in registered_abs]
        logger.info(f"Matched databases: {matched}")
        
        # If no matches, show what didn't match
        if not matched and all_databases:
            logger.warning(f"No matches found. Registered: {list(registered_abs)}, Found: {all_databases}")
        
        return matched
    
    return all_databases


@app.route("/")
def index() -> str:
    """Serve the main dashboard page."""
    return render_template("dashboard.html")


@app.route("/api/data")
def api_data() -> str:
    """API endpoint that returns chart data and statistics.

    Query parameters:
        start_ts: Unix epoch milliseconds (optional, defaults to bot_runner start time)
        only_running: Boolean (optional, defaults to True) - if True, only show running bots

    Returns:
        JSON response with database data and statistics
    """
    # Get start timestamp from query parameter, or use default (bot_runner start time)
    start_timestamp = request.args.get("start_ts", type=int)
    if start_timestamp is None:
        start_timestamp = _default_start_timestamp if _default_start_timestamp is not None else 0

    # Get only_running parameter (default to True)
    only_running = request.args.get("only_running", "true").lower() == "true"

    # Find databases (filtered by only_running setting)
    databases = find_databases(only_running=only_running)
    
    # Debug logging
    import logging
    logger = logging.getLogger(__name__)
    logger.info(f"API request: only_running={only_running}, registered={len(_running_bot_databases)}, found={len(databases)}")
    if _running_bot_databases:
        logger.info(f"Registered databases: {list(_running_bot_databases)}")
    all_dbs = find_databases(only_running=False)
    logger.info(f"Total databases in data directory: {len(all_dbs)}")
    if all_dbs:
        logger.info(f"All databases: {all_dbs[:3]}...")  # Show first 3

    if not databases:
        # Return empty result instead of 404, with informative message
        message = "No databases found"
        if only_running:
            if not _running_bot_databases:
                message = "No running bots are registered. Uncheck 'Only show databases from currently running bots' to see all databases."
            else:
                message = f"No databases match the {len(_running_bot_databases)} registered running bot(s). Check if database paths match."
        elif not DATA_DIR.exists():
            message = f"Data directory does not exist: {DATA_DIR}"
        else:
            message = f"No .db files found in data directory: {DATA_DIR}"
        
        return jsonify({
            "databases": [],
            "message": message
        })

    # Process all databases
    results = []
    for db_path in databases:
        try:
            # Check if database has any positions at all (without timestamp filter)
            from ui.shared import query_positions
            all_positions = query_positions(db_path, 0)  # 0 = no filter
            logger.info(f"Database {Path(db_path).name}: {len(all_positions)} total positions")
            
            if start_timestamp > 0:
                filtered_positions = query_positions(db_path, start_timestamp)
                logger.info(f"Database {Path(db_path).name}: {len(filtered_positions)} positions after start_timestamp filter")
            
            result = process_database(db_path, start_timestamp)
            if result:
                label, timestamps, cumulative_pnl, stats = result
                # Convert datetime objects to ISO format strings for JSON
                timestamps_iso = [ts.isoformat() for ts in timestamps]
                results.append(
                    {
                        "label": label,
                        "timestamps": timestamps_iso,
                        "cumulative_pnl": cumulative_pnl,
                        "stats": stats,
                    }
                )
                logger.info(f"Successfully processed database {Path(db_path).name}: {stats['num_positions']} positions")
            else:
                # Database exists but has no positions (or all filtered out by start_timestamp)
                logger.warning(f"Database {Path(db_path).name} returned no data (no positions or all filtered by start_timestamp={start_timestamp})")
        except Exception as e:
            logger.exception(f"Error processing database {db_path}: {e}")
            # Continue processing other databases even if one fails

    if not results:
        # All databases were processed but returned no data
        message = "Databases found but contain no position data"
        if start_timestamp > 0:
            from datetime import datetime
            start_dt = datetime.fromtimestamp(start_timestamp / 1000.0)
            message = f"Databases found but contain no position data after {start_dt.isoformat()}. Try adjusting the start date."
        return jsonify({
            "databases": [],
            "message": message
        })

    return jsonify({"databases": results})


def main() -> None:
    """Run the Flask development server."""
    print("Starting PNL Dashboard server...")
    print(f"Data directory: {DATA_DIR}")
    print(f"Data directory exists: {DATA_DIR.exists()}")
    databases = find_databases(only_running=False)  # Show all in standalone mode
    print(f"Found {len(databases)} database(s)")
    if databases:
        print("Databases:")
        for db in databases[:5]:  # Show first 5
            print(f"  - {db}")
    print("Access the dashboard at: http://localhost:5000")
    print("Press Ctrl+C to stop the server")
    try:
        app.run(host="0.0.0.0", port=5000, debug=True)
    except Exception as e:
        print(f"Error starting server: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

