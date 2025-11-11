import asyncio
import logging
import multiprocessing
import threading
from datetime import datetime
from pathlib import Path

import uvloop

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

from config_loader import (
    get_platform_from_config,
    load_bot_config,
    print_config_summary,
    validate_platform_listener_combination,
)
from trading.universal_trader import UniversalTrader
from utils.logger import setup_file_logging
from database.manager import DatabaseManager
from core.wallet import Wallet


def setup_logging(bot_name: str, mode: str):
    """Set up logging to file for a specific bot instance."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    timestamp = ""#datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = log_dir / f"{bot_name}_{timestamp}-{mode.upper()}.log"

    setup_file_logging(str(log_filename))


async def start_bot(config_path: str):
    """Start a trading bot with the configuration from the specified path."""
    cfg = load_bot_config(config_path)

    # Determine mode (dryrun or live)
    testing = cfg.get("testing", {})
    mode = "dryrun" if testing.get("dry_run", False) else "live"

    setup_logging(cfg.get("name", "unknown"), mode)
    print_config_summary(cfg)

    # Get and validate platform from configuration
    try:
        platform = get_platform_from_config(cfg)
        logging.info(f"Detected platform: {platform.value}")
    except ValueError as e:
        logging.exception(f"Platform configuration error: {e}")
        return

    # Validate platform support
    try:
        from platforms import platform_factory

        if not platform_factory.registry.is_platform_supported(platform):
            logging.error(
                f"Platform {platform.value} is not supported. Available platforms: {[p.value for p in platform_factory.get_supported_platforms()]}"
            )
            return
    except Exception as e:
        logging.exception(f"Could not validate platform support: {e}")
        return

    # Validate listener compatibility
    listener_type = cfg["filters"]["listener_type"]
    if not validate_platform_listener_combination(platform, listener_type):
        from config_loader import get_supported_listeners_for_platform

        supported = get_supported_listeners_for_platform(platform)
        logging.error(
            f"Listener '{listener_type}' is not compatible with platform '{platform.value}'. Supported listeners: {supported}"
        )
        return

    # Derive wallet pubkey for database naming
    wallet = Wallet(cfg["private_key"])
    wallet_pubkey_short = str(wallet.pubkey)[:8]

    # Create database manager
    try:
                
        # Create database path
        bot_name = cfg.get("name", "unknown")
        db_path = f"data/{bot_name}_{wallet_pubkey_short}_{mode}.db"
        database_manager = DatabaseManager(db_path)
        
        logging.info(f"Database initialized: {db_path}")
        
        # Register this bot's database with the dashboard
        # Note: For bots in separate processes, this is already registered in main process
        # But we register again here in case the bot is running in the main process
        try:
            from ui.pnl_dashboard import register_running_bot_database
            register_running_bot_database(db_path)
            logging.debug(f"Registered database {db_path} with dashboard (bot process)")
        except Exception:
            logging.exception(f"Could not register database with dashboard")
            
    except Exception as e:
        logging.exception(f"Failed to initialize database: {e}")
        database_manager = None

    # Initialize universal trader with platform-specific configuration
    try:
        trader = UniversalTrader(
            # Connection settings - pass entire rpc config section
            rpc_config=cfg["rpc"],
            wallet=wallet,
            # Platform configuration - pass platform enum directly
            platform=platform,
            # Trade parameters
            buy_amount=cfg["trade"]["buy_amount"],
            buy_slippage=cfg["trade"]["buy_slippage"],
            sell_slippage=cfg["trade"]["sell_slippage"],
            max_wallet_loss_percentage=cfg["trade"].get("max_wallet_loss_percentage"),
            # Extreme fast mode settings
            extreme_fast_mode=cfg["trade"].get("extreme_fast_mode", False),
            # Exit strategy configuration
            exit_strategy=cfg["trade"].get("exit_strategy", "time_based"),
            take_profit_percentage=cfg["trade"].get("take_profit_percentage"),
            stop_loss_percentage=cfg["trade"].get("stop_loss_percentage"),
            trailing_stop_percentage=cfg["trade"].get("trailing_stop_percentage"),
            max_hold_time=cfg["trade"].get("max_hold_time"),
            max_no_price_change_time=cfg["trade"].get("max_no_price_change_time"),
            price_check_interval=cfg["trade"].get("price_check_interval", 10),
            # Listener configuration
            listener_type=cfg["filters"]["listener_type"],
            # Geyser configuration (if applicable)
            geyser_endpoint=cfg.get("geyser", {}).get("endpoint"),
            geyser_api_token=cfg.get("geyser", {}).get("api_token"),
            geyser_auth_type=cfg.get("geyser", {}).get("auth_type", "x-token"),
            # PumpPortal configuration (if applicable)
            pumpportal_url=cfg.get("pumpportal", {}).get(
                "url", "wss://pumpportal.fun/api/data"
            ),
            # Priority fee configuration
            enable_dynamic_priority_fee=cfg.get("priority_fees", {}).get(
                "enable_dynamic", False
            ),
            enable_fixed_priority_fee=cfg.get("priority_fees", {}).get(
                "enable_fixed", True
            ),
            fixed_priority_fee=cfg.get("priority_fees", {}).get("fixed_amount", 500000),
            extra_priority_fee=cfg.get("priority_fees", {}).get(
                "extra_percentage", 0.0
            ),
            hard_cap_prior_fee=cfg.get("priority_fees", {}).get("hard_cap", 500000),
            # Retry and timeout settings
            max_retries=cfg.get("retries", {}).get("max_attempts", 10),
            wait_time_after_creation=cfg.get("retries", {}).get(
                "wait_after_creation", 15
            ),
            wait_time_after_buy=cfg.get("retries", {}).get("wait_after_buy", 15),
            wait_time_before_new_token=cfg.get("retries", {}).get(
                "wait_before_new_token", 15
            ),
            max_token_age=cfg.get("filters", {}).get("max_token_age", 0.001),
            token_wait_timeout=cfg.get("timing", {}).get("token_wait_timeout", 120),
            # Cleanup settings
            cleanup_force_close_with_burn=cfg.get("cleanup", {}).get(
                "force_close_with_burn", False
            ),
            cleanup_with_priority_fee=cfg.get("cleanup", {}).get(
                "with_priority_fee", False
            ),
            # Trading filters
            match_string=cfg["filters"].get("match_string"),
            bro_address=cfg["filters"].get("bro_address"),
            marry_mode=cfg["filters"].get("marry_mode", False),
            max_buys=cfg["filters"].get("max_buys", None),
            min_initial_buy_sol=cfg["filters"].get("min_initial_buy_sol", 1.0),
            max_active_mints=cfg["trade"].get("max_active_mints", 1),
            # Compute unit configuration
            compute_units=cfg.get("compute_units", {}),
            # Testing configuration
            testing=cfg.get("testing"),
            offset_wallets=cfg.get("testing", {}).get("offset_wallets", []),
            # Database configuration
            database_manager=database_manager,
            # Blockhash caching configuration
            blockhash_update_interval=cfg.get("blockhash", {}).get("update_interval", 10.0),
            # Volatility-based adjustments
            enable_volatility_adjustment=cfg.get("trade", {}).get("enable_volatility_adjustment", False),
            volatility_window_seconds=cfg.get("trade", {}).get("volatility_window_seconds", 5.0),
            volatility_thresholds=cfg.get("trade", {}).get("volatility_thresholds"),
            volatility_tp_adjustments=cfg.get("trade", {}).get("volatility_tp_adjustments"),
            # Insufficient gain exit condition
            min_gain_percentage=cfg.get("trade", {}).get("min_gain_percentage"),
            min_gain_time_window=cfg.get("trade", {}).get("min_gain_time_window", 2),
            # Trade tracking configuration
            enable_trade_tracking=cfg.get("trade", {}).get("enable_trade_tracking", False),
            trade_staleness_threshold=cfg.get("trade", {}).get("trade_staleness_threshold", 30.0),
            # Market quality configuration
            market_quality_config=cfg.get("trade", {}).get("market_quality"),
        )

        await trader.start()

    except Exception as e:
        logging.exception(f"Failed to initialize or start trader: {e}")
        raise


def run_bot_process(config_path):
    asyncio.run(start_bot(config_path))


async def run_all_bots():
    """Run all bots defined in YAML files in the 'bots' directory."""
    bot_dir = Path("bots")
    if not bot_dir.exists():
        logging.error(f"Bot directory '{bot_dir}' not found")
        return

    bot_files = list(bot_dir.glob("*.yaml"))
    if not bot_files:
        logging.error(f"No bot configuration files found in '{bot_dir}'")
        return

    logging.info(f"Found {len(bot_files)} bot configuration files")

    # Check for wallet sharing across bot configurations
    wallet_to_bots = {}  # wallet_pubkey -> list of bot_names
    for file in bot_files:
        try:
            cfg = load_bot_config(str(file))
            bot_name = cfg.get("name", file.stem)
            
            # Skip disabled bots
            if not cfg.get("enabled", True):
                logging.info(f"Skipping wallet checking for bot '{bot_name}' because disabled")
                continue
            
            # Skip dry-run bots (they don't affect actual wallet balances)
            testing = cfg.get("testing", {})
            if testing.get("dry_run", False):
                logging.info(f"Skipping wallet checking for bot '{bot_name}' because in dry-run mode")
                continue
                
            wallet = Wallet(cfg["private_key"])
            wallet_pubkey = wallet.pubkey
            
            if wallet_pubkey not in wallet_to_bots:
                wallet_to_bots[wallet_pubkey] = []
            wallet_to_bots[wallet_pubkey].append(bot_name)
            
        except Exception as e:
            logging.warning(f"Could not process wallet info from {file}: {e}")
    
    # Report wallet sharing
    for wallet_pubkey, bot_names in wallet_to_bots.items():
        if len(bot_names) > 1:
            raise RuntimeError(f"Wallet {wallet_pubkey} is shared across multiple bots: {', '.join(bot_names)}")
        else:
            logging.debug(f"Wallet {wallet_pubkey} used by single bot: {bot_names[0]}")

    processes = []
    skipped_bots = 0

    for file in bot_files:
        try:
            cfg = load_bot_config(str(file))
            bot_name = cfg.get("name", file.stem)

            # Skip bots with enabled=False
            if not cfg.get("enabled", True):
                logging.info(f"Skipping disabled bot '{bot_name}'")
                skipped_bots += 1
                continue

            # Validate platform configuration
            try:
                platform = get_platform_from_config(cfg)

                # Check platform support
                from platforms import platform_factory

                if not platform_factory.registry.is_platform_supported(platform):
                    logging.error(
                        f"Platform {platform.value} is not supported for bot '{bot_name}'. Available platforms: {[p.value for p in platform_factory.get_supported_platforms()]}"
                    )
                    skipped_bots += 1
                    continue

                # Validate listener compatibility
                listener_type = cfg["filters"]["listener_type"]
                if not validate_platform_listener_combination(platform, listener_type):
                    from config_loader import get_supported_listeners_for_platform

                    supported = get_supported_listeners_for_platform(platform)
                    logging.error(
                        f"Listener '{listener_type}' is not compatible with platform '{platform.value}' for bot '{bot_name}'. Supported listeners: {supported}"
                    )
                    skipped_bots += 1
                    continue

            except Exception as e:
                logging.exception(
                    f"Invalid platform configuration for bot '{bot_name}': {e}. Skipping..."
                )
                skipped_bots += 1
                continue

            # Register database path in main process (for dashboard) before starting bot
            # This ensures the dashboard can see the database even if bot runs in separate process
            try:
                # Derive wallet pubkey for database naming (same logic as in start_bot)
                wallet = Wallet(cfg["private_key"])
                wallet_pubkey_short = str(wallet.pubkey)[:8]
                testing = cfg.get("testing", {})
                mode = "dryrun" if testing.get("dry_run", False) else "live"
                db_path = f"data/{bot_name}_{wallet_pubkey_short}_{mode}.db"
                
                # Register with dashboard in main process
                try:
                    from ui.pnl_dashboard import register_running_bot_database
                    register_running_bot_database(db_path)
                    logging.debug(f"Registered database {db_path} with dashboard (main process)")
                except Exception as e:
                    logging.warning(f"Could not register database with dashboard: {e}")
            except Exception as e:
                logging.warning(f"Could not determine database path for bot '{bot_name}': {e}")

            # Start bot in separate process or main process
            if cfg.get("separate_process", False):
                logging.info(
                    f"Starting bot '{bot_name}' ({platform.value}) in separate process"
                )
                p = multiprocessing.Process(
                    target=run_bot_process, args=(str(file),), name=f"bot-{bot_name}"
                )
                p.start()
                processes.append(p)
            else:
                logging.info(
                    f"Starting bot '{bot_name}' ({platform.value}) in main process"
                )
                await start_bot(str(file))

            await asyncio.sleep(5)
            logging.info(
                f"Sleeping 5 seconds to not run into 429s"
            )

        except Exception as e:
            logging.error(f"Failed to start bot from {file}: {e}")
            skipped_bots += 1

    logging.info(
        f"Started {len(bot_files) - skipped_bots} bots, skipped {skipped_bots} disabled/invalid bots"
    )

    # Wait for all processes to complete
    for p in processes:
        p.join()
        logging.info(f"Process {p.name} completed")

def start_dashboard_server(port: int = 5000) -> None:
    """Start the PNL dashboard web server in a separate thread.
    
    Args:
        port: Port number to run the dashboard server on (default: 5000)
    """
    logging.info("Attempting to start PNL dashboard server...")
    try:
        import sys
        import time
        from pathlib import Path
        
        # Add project root to path if needed
        project_root = Path(__file__).parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        
        logging.info(f"Project root: {project_root}")
        logging.info(f"Looking for ui.pnl_dashboard in {project_root / 'ui'}")
        
        from ui.pnl_dashboard import (
            app,
            set_default_start_timestamp,
        )
        logging.info("Successfully imported dashboard app")
        
        # Set the default start timestamp to when bot_runner started (in milliseconds)
        start_timestamp_ms = int(time.time() * 1000)
        set_default_start_timestamp(start_timestamp_ms)
        logging.info(f"Dashboard default start timestamp set to: {start_timestamp_ms}")

        def run_server() -> None:
            """Run Flask server in a thread."""
            try:
                app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)
            except Exception as e:
                logging.error(f"Error running dashboard server: {e}")
                import traceback
                traceback.print_exc()

        dashboard_thread = threading.Thread(
            target=run_server, daemon=True, name="pnl-dashboard"
        )
        dashboard_thread.start()
        # Give it a moment to start
        time.sleep(0.5)
        logging.info(f"PNL Dashboard server started at http://localhost:{port}")
    except Exception as e:
        logging.error(f"Could not start PNL dashboard server: {e}")
        import traceback
        traceback.print_exc()


def main() -> None:
    import sys
    
    # Check for dashboard-only flag
    dashboard_only = "--dashboard-only" in sys.argv or "-d" in sys.argv
    
    # Set up basic console logging for main process
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Suppress noisy third-party loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("solana.rpc").setLevel(logging.WARNING)

    # Start PNL dashboard server
    # Use port 5001 in dashboard-only mode, 5000 otherwise
    dashboard_port = 5001 if dashboard_only else 5000
    start_dashboard_server(port=dashboard_port)
    
    if dashboard_only:
        logging.info("Dashboard-only mode: Starting dashboard server only (no bots will be started)")
        logging.info(f"Dashboard available at http://localhost:{dashboard_port}")
        logging.info("Press Ctrl+C to stop the dashboard server")
        # Keep the process alive
        try:
            import time
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logging.info("Shutting down dashboard server...")
        return

    # Log supported platforms and listeners
    try:
        from platforms import platform_factory

        supported_platforms = platform_factory.get_supported_platforms()
        logging.info(f"Supported platforms: {[p.value for p in supported_platforms]}")

        # Log listener compatibility for each platform
        from config_loader import get_supported_listeners_for_platform

        for platform in supported_platforms:
            listeners = get_supported_listeners_for_platform(platform)
            logging.info(f"Platform {platform.value} supports listeners: {listeners}")

    except Exception as e:
        logging.warning(f"Could not load platform information: {e}")

    asyncio.run(run_all_bots())

def debug(sig, frame):
    import traceback
    traceback.print_stack(frame)
    namespace = frame.f_globals.copy()
    namespace.update(frame.f_locals)
    code.interact(local=namespace)

if __name__ == "__main__":
    import code, signal

    signal.signal(signal.SIGUSR1, debug)

    main()
