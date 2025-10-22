"""
Volatility calculation utilities for trading bot.

This module provides realized volatility calculation using sliding windows
to dynamically adjust trading parameters based on price movement patterns.
"""

import math
from collections import deque
from dataclasses import dataclass
from time import time
from typing import Optional

from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PriceData:
    """Price data point with timestamp."""
    
    price: float
    timestamp: float  # Unix timestamp in seconds


class VolatilityCalculator:
    """
    Calculate realized volatility using a sliding time window.
    
    Realized volatility measures the actual magnitude of price changes
    over a specified time period, providing insight into market stability.
    """
    
    def __init__(self, window_seconds: float = 5.0):
        """Initialize volatility calculator.
        
        Args:
            window_seconds: Time window for volatility calculation in seconds
        """
        self.window_seconds = window_seconds
        self.price_data: deque[PriceData] = deque()
        self._last_volatility: Optional[float] = None
        self._last_calculation_time: Optional[float] = None
        
    def _clean_old_data(self, current_timestamp: float) -> None:
        """Remove old data points outside the time window.
        
        Args:
            current_timestamp: Current timestamp (uses current time if None)
        """
        cutoff_time = current_timestamp - self.window_seconds
        while self.price_data and self.price_data[0].timestamp < cutoff_time:
            self.price_data.popleft()

    def add_price(self, price: float, timestamp: float) -> None:
        """Add a new price data point.
        
        Args:
            price: Current price
            timestamp: Unix timestamp (uses current time if None)
        """
        # Add new price data
        self.price_data.append(PriceData(price=price, timestamp=timestamp))
        
        # Remove old data outside the window
        self._clean_old_data(timestamp)
    
    def calculate_volatility(self, current_timestamp: float) -> Optional[float]:
        """Calculate realized volatility over the time window.
        
        Args:
            current_timestamp: Current timestamp (uses current time if None)
            
        Returns:
            Volatility as percentage per second, or None if insufficient data
        """            
        # Clean old data
        self._clean_old_data(current_timestamp)
            
        # Need at least 2 data points for volatility calculation
        if len(self.price_data) < 2:
            return None
            
        # Calculate percentage returns
        returns = []
        for i in range(1, len(self.price_data)):
            prev_price = self.price_data[i-1].price
            curr_price = self.price_data[i].price
            
            if prev_price > 0:
                # Calculate percentage return
                pct_return = (curr_price - prev_price) / prev_price
                returns.append(pct_return)
        
        if len(returns) < 2:
            return None
            
        # Calculate standard deviation of returns
        mean_return = sum(returns) / len(returns)
        variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
        std_dev = math.sqrt(variance)
        
        # Convert to percentage per second
        time_span = self.price_data[-1].timestamp - self.price_data[0].timestamp
        if time_span > 0:
            volatility_per_second = std_dev / time_span
        else:
            volatility_per_second = std_dev
            
        # Store for caching
        self._last_volatility = volatility_per_second
        self._last_calculation_time = current_timestamp
        
        return volatility_per_second
    
    def get_volatility_level(self, current_timestamp: float) -> str:
        """Get volatility level classification.
        
        Args:
            current_timestamp: Current timestamp (uses current time if None)
            
        Returns:
            Volatility level: 'low', 'medium', or 'high'
        """
        volatility = self.calculate_volatility(current_timestamp)
        
        if volatility is None:
            return 'unknown'
        elif volatility < 0.05:  # < 5% per 5 seconds
            return 'low'
        elif volatility < 0.1:  # 5-15% per 5 seconds
            return 'medium'
        else:  # > 15% per 5 seconds
            return 'high'
    
    def get_cached_volatility(self) -> Optional[float]:
        """Get the last calculated volatility value.
        
        Returns:
            Last calculated volatility or None
        """
        return self._last_volatility
    
    def has_sufficient_data(self, current_timestamp: float) -> bool:
        """Check if we have sufficient data for volatility calculation.
        
        Args:
            current_timestamp: Current timestamp (uses current time if None)
            
        Returns:
            True if we have enough data points in the time window
        """
            
        # Clean old data
        self._clean_old_data(current_timestamp)
            
        return len(self.price_data) >= 2
    
    def get_data_count(self) -> int:
        """Get number of data points in current window.
        
        Returns:
            Number of price data points
        """
        return len(self.price_data)
    
    def clear(self) -> None:
        """Clear all stored price data."""
        self.price_data.clear()
        self._last_volatility = None
        self._last_calculation_time = None


def calculate_take_profit_adjustment(
    original_tp_percentage: float,
    volatility_level: str,
    adjustment_config: Optional[dict] = None
) -> float:
    """Calculate adjusted take profit percentage based on volatility.
    
    Args:
        original_tp_percentage: Original take profit percentage (e.g., 0.4 for 40%)
        volatility_level: Volatility level ('low', 'medium', 'high')
        adjustment_config: Custom adjustment configuration
        
    Returns:
        Adjusted take profit percentage
    """
    if adjustment_config is None:
        # Default adjustments
        adjustments = {
            'low': 0.0,      # No adjustment for low volatility
            'medium': 0.25,  # Reduce by 25% for medium volatility
            'high': 0.45,    # Reduce by 45% for high volatility
        }
    else:
        adjustments = adjustment_config
        
    adjustment_factor = adjustments.get(volatility_level, 0.0)
    adjusted_percentage = original_tp_percentage * (1 - adjustment_factor)
    
    # Ensure we don't go below 5% take profit
    return max(adjusted_percentage, 0.05)
