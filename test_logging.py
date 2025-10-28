#!/usr/bin/env python3
"""Test script to verify logging setup works correctly."""

import logging
import sys
from pathlib import Path

# Add src to path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_basic_logging():
    """Test basic logging setup."""
    print("=== Testing Basic Logging ===")
    
    # Test 1: Basic logging without any setup
    logging.info("This should NOT appear (no handlers)")
    
    # Test 2: Basic logging with basicConfig
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.info("This SHOULD appear (basicConfig setup)")
    
    # Test 3: Test our setup_file_logging function
    from utils.logger import setup_file_logging
    setup_file_logging("test.log")
    logging.info("This SHOULD appear in both console and file")

def test_bot_runner_logging():
    """Test the specific logging calls from bot_runner."""
    print("\n=== Testing Bot Runner Logging ===")
    
    # Simulate the bot_runner main function setup
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    # Test the specific log message that wasn't appearing
    bot_files = list(Path("bots").glob("*.yaml"))
    logging.info(f"Found {len(bot_files)} bot configuration files")
    
    # Test other logging calls
    logging.info("Supported platforms: ['pump_fun', 'lets_bonk']")
    logging.info("Platform pump_fun supports listeners: ['logs', 'blocks', 'geyser', 'pumpportal']")

if __name__ == "__main__":
    test_basic_logging()
    test_bot_runner_logging()
    print("\n=== Test Complete ===")
