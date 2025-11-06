"""
Flask web server for PNL visualization dashboard.

This server provides an interactive web dashboard for visualizing cumulative PNL
from trading bot databases. It automatically discovers all databases in the data/
directory and serves them via a web interface.
"""

from pathlib import Path

from flask import Flask, jsonify, render_template, request
from flask_compress import Compress  # type: ignore

from ui.shared import process_database

# Set template folder to ui/templates (relative to this file)
app = Flask(__name__, template_folder=str(Path(__file__).parent / "templates"))
Compress(app)  # Enable gzip compression

# Path to data directory (relative to project root)
DATA_DIR = Path(__file__).parent.parent / "data"


def find_databases() -> list[str]:
    """Find all database files in the data directory.

    Returns:
        List of database file paths
    """
    if not DATA_DIR.exists():
        return []
    return sorted([str(db) for db in DATA_DIR.glob("*.db")])


@app.route("/")
def index() -> str:
    """Serve the main dashboard page."""
    start_ts = request.args.get("start_ts", type=int)
    return render_template("dashboard.html", start_ts=start_ts)


@app.route("/api/data")
def api_data() -> str:
    """API endpoint that returns chart data and statistics.

    Query parameters:
        start_ts: Unix epoch milliseconds (optional, defaults to 0)

    Returns:
        JSON response with database data and statistics
    """
    # Get start timestamp from query parameter
    start_timestamp = request.args.get("start_ts", type=int, default=0)

    # Find all databases
    databases = find_databases()

    if not databases:
        return jsonify({"error": "No databases found in data directory"}), 404

    # Process all databases
    results = []
    for db_path in databases:
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

    return jsonify({"databases": results})


def main() -> None:
    """Run the Flask development server."""
    print("Starting PNL Dashboard server...")
    print(f"Data directory: {DATA_DIR}")
    print(f"Data directory exists: {DATA_DIR.exists()}")
    databases = find_databases()
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

