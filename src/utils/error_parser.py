"""
Error parsing utilities for Solana transaction errors.

This module provides functionality to parse and classify Solana transaction errors,
particularly for identifying slippage errors and other platform-specific error types.
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from interfaces.core import Platform
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ParsedError:
    """Parsed error information from a transaction error string."""

    code: int | None = None  # Error code (e.g., 6002)
    name: str | None = None  # Error name (e.g., "TooMuchSolRequired")
    message: str | None = None  # Error message
    is_slippage: bool = False  # Whether this is a slippage error
    platform: Platform | None = None  # Platform this error is associated with


class ErrorParser:
    """Parser for Solana transaction errors using IDL error definitions."""

    def __init__(self):
        """Initialize error parser with IDL error definitions."""
        self._error_definitions: dict[Platform, dict[int, dict[str, Any]]] = {}
        self._load_idl_errors()

    def _load_idl_errors(self) -> None:
        """Load error definitions from IDL files."""
        # Load Pump.fun errors
        pumpfun_idl_path = Path(__file__).parent.parent.parent / "idl" / "pump_fun_idl.json"
        if pumpfun_idl_path.exists():
            try:
                with open(pumpfun_idl_path, "r") as f:
                    idl_data = json.load(f)
                    errors = idl_data.get("errors", [])
                    self._error_definitions[Platform.PUMP_FUN] = {
                        err["code"]: err for err in errors if "code" in err
                    }
                    logger.debug(f"Loaded {len(errors)} error definitions from pump_fun_idl.json")
            except Exception as e:
                logger.warning(f"Failed to load pump_fun_idl.json errors: {e}")

        # Load Raydium Launchlab (LetsBonk) errors
        launchlab_idl_path = Path(__file__).parent.parent.parent / "idl" / "raydium_launchlab_idl.json"
        if launchlab_idl_path.exists():
            try:
                with open(launchlab_idl_path, "r") as f:
                    idl_data = json.load(f)
                    errors = idl_data.get("errors", [])
                    self._error_definitions[Platform.LETS_BONK] = {
                        err["code"]: err for err in errors if "code" in err
                    }
                    logger.debug(f"Loaded {len(errors)} error definitions from raydium_launchlab_idl.json")
            except Exception as e:
                logger.warning(f"Failed to load raydium_launchlab_idl.json errors: {e}")

        # Load Raydium AMM errors (for reference, though we primarily use Pump.fun and LetsBonk)
        raydium_amm_idl_path = Path(__file__).parent.parent.parent / "idl" / "raydium_amm_idl.json"
        if raydium_amm_idl_path.exists():
            try:
                with open(raydium_amm_idl_path, "r") as f:
                    idl_data = json.load(f)
                    errors = idl_data.get("errors", [])
                    # Store as a separate dict, we can check it if needed
                    self._amm_errors = {err["code"]: err for err in errors if "code" in err}
                    logger.debug(f"Loaded {len(errors)} error definitions from raydium_amm_idl.json")
            except Exception as e:
                logger.warning(f"Failed to load raydium_amm_idl.json errors: {e}")

    def _extract_error_code(self, error_string: str) -> int | None:
        """Extract error code from error string.

        Examples:
            "InstructionError((3, Tagged(Custom(InstructionErrorCustom(6002)))))"
            -> 6002
            "Custom(6002)"
            -> 6002

        Args:
            error_string: Error string from transaction

        Returns:
            Error code if found, None otherwise
        """
        if not error_string:
            return None

        # Pattern to match InstructionErrorCustom(code) or Custom(code)
        patterns = [
            r"InstructionErrorCustom\((\d+)\)",  # InstructionErrorCustom(6002)
            r"Custom\((\d+)\)",  # Custom(6002)
            r"\((\d+)\)",  # Fallback: any number in parentheses
        ]

        for pattern in patterns:
            match = re.search(pattern, error_string)
            if match:
                try:
                    return int(match.group(1))
                except (ValueError, IndexError):
                    continue

        return None

    def parse_error(
        self, error_string: str | None, platform: Platform | None = None
    ) -> ParsedError:
        """Parse error string and classify error type.

        Args:
            error_string: Error string from transaction (can be None)
            platform: Platform to check errors for (if None, checks all platforms)

        Returns:
            ParsedError with error information and classification
        """
        if not error_string:
            return ParsedError()

        # Extract error code
        error_code = self._extract_error_code(error_string)

        if error_code is None:
            return ParsedError(message=error_string)

        # Check error definitions for each platform
        platforms_to_check = [platform] if platform else list(Platform)
        error_def = None
        found_platform = None

        for check_platform in platforms_to_check:
            if check_platform in self._error_definitions:
                if error_code in self._error_definitions[check_platform]:
                    error_def = self._error_definitions[check_platform][error_code]
                    found_platform = check_platform
                    break

        if not error_def:
            return ParsedError(code=error_code, message=error_string)

        # Determine if this is a slippage error
        is_slippage = self._is_slippage_error(error_code, found_platform)

        return ParsedError(
            code=error_code,
            name=error_def.get("name"),
            message=error_def.get("msg") or error_string,
            is_slippage=is_slippage,
            platform=found_platform,
        )

    def _is_slippage_error(self, error_code: int, platform: Platform | None) -> bool:
        """Check if an error code represents a slippage error.

        Args:
            error_code: Error code to check
            platform: Platform the error is from

        Returns:
            True if this is a slippage error
        """
        if not platform:
            return False

        # Pump.fun slippage errors
        if platform == Platform.PUMP_FUN:
            return error_code in [6002, 6003]  # TooMuchSolRequired, TooLittleSolReceived

        # LetsBonk/Raydium Launchlab slippage errors
        if platform == Platform.LETS_BONK:
            return error_code == 6004  # ExceededSlippage

        # Raydium AMM slippage errors (for reference)
        # Note: We don't use Raydium AMM directly, but errors might reference it
        if hasattr(self, "_amm_errors") and error_code in self._amm_errors:
            return error_code == 30  # ExceededSlippage

        return False


# Global error parser instance
_error_parser: ErrorParser | None = None


def get_error_parser() -> ErrorParser:
    """Get or create the global error parser instance.

    Returns:
        ErrorParser instance
    """
    global _error_parser
    if _error_parser is None:
        _error_parser = ErrorParser()
    return _error_parser


def parse_transaction_error(
    error_string: str | None, platform: Platform | None = None
) -> ParsedError:
    """Parse a transaction error string and return parsed error information.

    Convenience function that uses the global error parser.

    Args:
        error_string: Error string from transaction
        platform: Platform to check errors for (if None, checks all platforms)

    Returns:
        ParsedError with error information and classification
    """
    return get_error_parser().parse_error(error_string, platform)

