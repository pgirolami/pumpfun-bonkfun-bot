"""
Visualize cumulative PNL from trading bot databases.

This script queries closed positions from one or more SQLite databases and creates
an interactive plotly chart showing cumulative PNL over time with overlaid plots.

Usage:
    # Option 1: Run as a script
    uv run analysis/visualize_cumulative_pnl.py
    
    # Option 2: Run interactively in VSCode
    # Install Jupyter extension, then use "Run Cell" buttons above # %% markers
"""

# %%
# Cell 1: Imports and setup
from datetime import datetime
from pathlib import Path

import plotly.graph_objects as go

# Color palette for multiple databases
COLORS = [
    "#1f77b4",  # blue
    "#ff7f0e",  # orange
    "#2ca02c",  # green
    "#d62728",  # red
    "#9467bd",  # purple
    "#8c564b",  # brown
    "#e377c2",  # pink
    "#7f7f7f",  # gray
    "#bcbd22",  # olive
    "#17becf",  # cyan
]


# %%
# Cell 2: Configuration
# Global start timestamp (applies to all databases)
start_timestamp = 0  # Unix epoch milliseconds

# Database paths (labels will be automatically derived from filenames)
databases = [
    # "/Users/philippe/Documents/crypto/pumpfun-bonkfun-bot/data/bot-helius-rpc-with-size-limit_DapP34Bq_live.db",
    # "/Users/philippe/Documents/crypto/pumpfun-bonkfun-bot/data/bot-helius-rpc_66yiD5Cs_live.db",    
    "/tmp/bot-heliussender_66yiD5Cs_dryrun.db",
    "/tmp/bot-old-heliussender-avg_pnl_5_66yiD5Cs_dryrun.db",
    "/tmp/bot-old-heliussender-avg_pnl_66yiD5Cs_dryrun.db",
    "/tmp/bot-old-heliussender_66yiD5Cs_dryrun.db",
    "/tmp/bot-old_66yiD5Cs_dryrun.db",
    "/tmp/bot-sniper-pumpportal_66yiD5Cs_dryrun.db",
]

# %%
# Cell 3: Import shared functions
import sys
from pathlib import Path

# Add parent directory to path to import ui.shared
sys.path.insert(0, str(Path(__file__).parent.parent))

from ui.shared import process_database


def main() -> None:
    """Main function to run the visualization."""
    if not databases:
        print("No databases configured. Please add databases to the 'databases' list in Cell 2.")
        return

    print("=" * 80)
    print("Cumulative PNL Visualization")
    print("=" * 80)
    print()

    # Process all databases
    processed_data = []
    for db_path in databases:
        result = process_database(db_path, start_timestamp)
        if result:
            processed_data.append(result)
            label, _, _, _, stats = result
            print(f"📊 {label}:")
            print(f"   Total positions: {stats['num_positions']}")
            print(f"   Winning positions: {stats['winning_positions']}")
            print(f"   Win rate: {stats['win_rate']:.2f}%")
            print(f"   Average PNL per position: {stats['avg_pnl']:.6f} SOL")
            print(f"   Total cumulative PNL: {stats['total_pnl']:.6f} SOL")
            print(f"   Max drawdown: {stats['max_drawdown']:.2f}%")
            print()
            print(f"   Position Duration Statistics:")
            print(f"      Average: {stats['avg_duration']:.1f} seconds")
            print(f"      Median: {stats['median_duration']:.1f} seconds")
            print(f"      P10: {stats['p10_duration']:.1f} seconds")
            print(f"      P90: {stats['p90_duration']:.1f} seconds")
            print()
        else:
            print(f"⚠️  No closed positions found in {db_path} after timestamp {start_timestamp}")
            print()

    if not processed_data:
        print("❌ No data found in any database. Please check your configurations.")
        return

    # Create the plot
    fig = go.Figure()

    # Add cumulative PNL line for each database
    for idx, (label, timestamps, cumulative_pnl, cumulative_normalized_pnl, stats) in enumerate(processed_data):
        color = COLORS[idx % len(COLORS)]
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=cumulative_pnl,
                mode="lines+markers",
                name=label,
                line=dict(color=color, width=2),
                marker=dict(size=4, color=color),
                hovertemplate=(
                    f"<b>{label}</b><br>"
                    "<b>%{x}</b><br>"
                    "Cumulative PNL: %{y:.6f} SOL<extra></extra>"
                ),
            )
        )

    # Add zero reference line
    fig.add_hline(
        y=0,
        line_dash="dash",
        line_color="gray",
        opacity=0.5,
        annotation_text="Zero PNL",
    )

    # Determine title
    if len(processed_data) == 1:
        title = f"Cumulative PNL Over Time<br><sub>{processed_data[0][0]}</sub>"
    else:
        title = f"Cumulative PNL Over Time<br><sub>Comparing {len(processed_data)} databases</sub>"

    # Update layout
    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            xanchor="center",
        ),
        xaxis_title="Time",
        yaxis_title="Cumulative PNL (SOL)",
        hovermode="x unified",
        template="plotly_white",
        showlegend=True,
        legend=dict(
            x=1.02,
            y=1,
            xanchor="left",
            yanchor="top",
            bgcolor="rgba(255,255,255,0.8)",
        ),
        height=600,
        xaxis=dict(
            showgrid=True,
            gridcolor="lightgray",
            showspikes=True,
            spikecolor="gray",
            spikethickness=1,
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="lightgray",
            showspikes=True,
            spikecolor="gray",
            spikethickness=1,
        ),
    )

    # Show the plot
    fig.show()


# %%
# Cell 5: Run main function (for interactive execution)
# Uncomment the line below if running as a script, or run this cell interactively
if __name__ == "__main__":
    main()
