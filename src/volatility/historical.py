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
from typing import Optional, Literal


def calculate_realized_volatility(
    prices: pd.Series,
    window: int = 30,
    method: Literal['close', 'parkinson', 'garman_klass'] = 'close',
    annualize: bool = True
) -> pd.Series:
    """
    Calculate rolling realized volatility from price data.
    
    Args:
        prices: pandas Series of prices (typically 'close' prices)
        window: Rolling window size in days (default 30)
        method: Calculation method:
            - 'close': Standard close-to-close volatility
            - 'parkinson': Uses high-low range (more efficient)
            - 'garman_klass': Uses OHLC (most efficient)
        annualize: If True, multiply by sqrt(252) to annualize
    
    Returns:
        pandas Series of rolling volatility estimates
        
    Example:
        >>> prices = pd.Series([100, 101, 99, 102, 98])
        >>> rv = calculate_realized_volatility(prices, window=3)
        >>> print(rv)
        
    TODO: Implement realized volatility calculation
    
    Hints for 'close' method:
        1. Calculate log returns: ln(P_t / P_t-1)
        2. Calculate rolling standard deviation of returns
        3. Annualize: std * sqrt(252) for daily data
        4. Handle NaNs at the start (first 'window' values)
    
    Interview Question: "Why use log returns instead of simple returns?"
        Log returns are additive over time, symmetric, and more suitable
        for statistical analysis. Also, Black-Scholes assumes log-normal
        stock prices, which means log returns are normally distributed.
    """
    # TODO: Implement close-to-close method
    # For Phase 3, just implement 'close' method (ignore parkinson/GK for now)
    pass


def calculate_realized_volatility_ohlc(
    ohlc_df: pd.DataFrame,
    window: int = 30,
    method: Literal['parkinson', 'garman_klass'] = 'parkinson'
) -> pd.Series:
    """
    Calculate realized volatility using OHLC data (more efficient estimators).
    
    Args:
        ohlc_df: DataFrame with columns ['open', 'high', 'low', 'close']
        window: Rolling window size
        method: 'parkinson' (high-low) or 'garman_klass' (OHLC)
    
    Returns:
        pandas Series of volatility estimates
        
    Parkinson Formula:
        σ² = (1/4ln2) * (1/n) * Σ[ln(High/Low)]²
        
    Garman-Klass Formula (more complex, more efficient):
        σ² = 0.5*[ln(H/L)]² - (2ln2-1)*[ln(C/O)]²
    
    TODO: OPTIONAL - Implement for extra credit
    These estimators are more efficient (less variance) than close-to-close,
    but require OHLC data. For your project, close-to-close is sufficient.
    """
    pass


class RollingVolCalculator:
    """
    Class to maintain rolling volatility calculations.
    
    Useful for backtesting - maintains state as new data arrives.
    """
    
    def __init__(self, window: int = 30, annualize: bool = True):
        """
        Initialize rolling vol calculator.
        
        Args:
            window: Lookback window in days
            annualize: Whether to annualize the volatility
        """
        self.window = window
        self.annualize = annualize
        self.returns = []  # Store recent returns
        
    def update(self, new_price: float, prev_price: float) -> float:
        """
        Update calculator with new price and return current volatility estimate.
        
        Args:
            new_price: Current price
            prev_price: Previous price
            
        Returns:
            Current volatility estimate
            
        TODO: Implement incremental volatility update
        
        Hints:
            1. Calculate log return: ln(new_price / prev_price)
            2. Append to self.returns list
            3. Keep only last 'window' returns
            4. Calculate std of returns
            5. Annualize if needed: std * sqrt(252)
        """
        # TODO: Implement
        pass
    
    def get_current_vol(self) -> Optional[float]:
        """
        Get current volatility estimate.
        
        Returns:
            Current vol, or None if insufficient data
        """
        # TODO: Implement
        # Return None if len(self.returns) < self.window
        pass
    
    def reset(self):
        """Reset the calculator state."""
        self.returns = []


# Helper functions for volatility analysis

def vol_forecast_accuracy(
    realized_vols: pd.Series,
    forecasted_vols: pd.Series,
    horizon: int = 1
) -> dict:
    """
    Measure how accurate volatility forecasts are.
    
    Args:
        realized_vols: Actual realized volatility
        forecasted_vols: Forecasted volatility (shifted by horizon)
        horizon: Forecast horizon in periods
    
    Returns:
        Dictionary with accuracy metrics:
            - RMSE: Root mean squared error
            - MAE: Mean absolute error
            - Correlation: Correlation between forecast and realized
            
    TODO: OPTIONAL - Implement for analysis phase
    This will be useful in Phase 5 when comparing which vol model works best.
    """
    pass


if __name__ == "__main__":
    # Test your implementation
    print("="*60)
    print("REALIZED VOLATILITY TEST")
    print("="*60)
    
    # Create test price series
    np.random.seed(42)
    n_days = 100
    returns = np.random.normal(0, 0.02, n_days)  # 2% daily std
    prices = 100 * np.exp(np.cumsum(returns))
    prices_series = pd.Series(prices)
    
    # Calculate realized vol
    rv = calculate_realized_volatility(prices_series, window=30)
    
    if rv is not None:
        print(f"\nPrice range: ${prices_series.min():.2f} - ${prices_series.max():.2f}")
        print(f"Realized vol (30-day): {rv.iloc[-1]:.2%}")
        print(f"Expected ~20% annualized (2% daily * sqrt(252))")
        print("\n✓ Function implemented!")
    else:
        print("\n❌ Not implemented yet - complete the TODO")

