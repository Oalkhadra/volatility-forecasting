"""
Strategy B: Historical Realized Volatility

This module calculates historical realized volatility using different methods:
- Close-to-close returns (standard)
- Parkinson estimator (uses high-low)
- Garman-Klass estimator (uses OHLC)

The hypothesis: Market systematically misprices options relative to what 
actually happens. Historical realized vol provides the "true" volatility.
"""

import numpy as np
import pandas as pd
from typing import Optional
from pandas.io.stata import datetime


def calculate_realized_volatility(
    prices: pd.Series,
    window: int = 30,
    annualize: bool = True
) -> pd.Series:
    """
    Calculate rolling realized volatility from price data.
    
    Args:
        prices: pandas Series of prices (typically 'close' prices)
        window: Rolling window size in days (default 30)
        annualize: If True, multiply by sqrt(252) to annualize
    
    Returns:
        pandas Series of rolling volatility estimates

    """
    log_returns = np.log(prices / prices.shift(1))
    rolling_stdev = log_returns.rolling(window=window, min_periods=2).std()
    
    if annualize:
        rolling_stdev = rolling_stdev * np.sqrt(252)

    return rolling_stdev


class RollingVolCalculator:
    """
    Class to maintain rolling volatility calculations.
    """
    
    def __init__(self,  window: int = 30, annualize: bool = True):
        """
        Initialize rolling vol calculator.
        
        Args:
            window: Lookback window in days
            annualize: Whether to annualize the volatility
        """
        self.window = window
        self.annualize = annualize

    def get_current_vol(self, prices_data: pd.DataFrame, current_date: datetime) -> Optional[float]:
        """
        Get current volatility estimate.
        
        Returns:
            Current vol, or None if insufficient data
        """
        prices_current = prices_data[prices_data['date'] < current_date]

        if len(prices_current) < self.window:
            return None
        
        return_window = prices_current.tail(self.window)
        returns = np.log(return_window['close'] / return_window['close'].shift(1)).dropna()

        stdev = np.std(returns, ddof=1)

        if self.annualize:
            return stdev * np.sqrt(252)
        
        return stdev