"""
Strategy C: Volatility Surface Fitting

This module fits parametric models to the implied volatility surface.
The hypothesis: Less liquid strikes/tenors deviate from the "fair" surface
implied by liquid options. We can find relative value by fitting a smooth
surface and comparing individual option IVs to the fitted value.

Common surface models:
- Polynomial (simple, good for learning)
- SVI (Stochastic Volatility Inspired - industry standard)
- SSVI (Surface SVI)
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize, curve_fit
from typing import Tuple, Literal


class VolatilitySurface:
    """
    Volatility surface model using polynomial fitting.
    
    Models IV as function of moneyness and time to expiration:
    IV(m, T) = a + b*m + c*m² + d*T + e*m*T
    
    Where m = ln(K/S) (log-moneyness)
    """
    
    def __init__(self, method: Literal['polynomial', 'svi'] = 'polynomial'):
        """
        Initialize volatility surface model.
        
        Args:
            method: Fitting method
                - 'polynomial': Simple polynomial fit (good for learning)
                - 'svi': Stochastic Volatility Inspired (industry standard)
        """
        self.method = method
        self.params = None
        self.is_fitted = False
        
    def fit(self, option_data: pd.DataFrame) -> None:
        """
        Fit volatility surface to option data.
        
        Args:
            option_data: DataFrame with columns:
                - 'strike': Strike price
                - 'underlying_price': Spot price
                - 'days_to_expiry': Days to expiration
                - 'vol': Implied volatility (from market or Dolt)
                - 'call_put': Option type
                
        The model fits: IV = f(moneyness, time_to_expiry)
        
        TODO: Implement surface fitting
        
        Hints for polynomial method:
            1. Calculate log-moneyness: m = ln(K/S)
            2. Convert DTE to years: T = days/365
            3. Define features: [1, m, m², T, m*T, ...]
            4. Use least squares to fit: IV = β₀ + β₁*m + β₂*m² + ...
            5. Store coefficients in self.params
            6. Set self.is_fitted = True
            
        Interview Question: "Why use log-moneyness instead of K/S?"
            Log-moneyness is more symmetric around ATM and better behaved
            for statistical modeling. It's also what most academic and 
            industry models use.
        """
        # TODO: Implement fitting
        pass
    
    def get_vol(
        self,
        strike: float,
        underlying_price: float,
        days_to_expiry: float,
        option_type: str = 'call'
    ) -> float:
        """
        Get fitted volatility for a specific option.
        
        Args:
            strike: Strike price
            underlying_price: Spot price
            days_to_expiry: Days to expiration
            option_type: 'call' or 'put' (may adjust for skew)
            
        Returns:
            Fitted implied volatility
            
        TODO: Implement volatility lookup from fitted surface
        
        Hints:
            1. Check if model is fitted
            2. Calculate log-moneyness: m = ln(K/S)
            3. Convert DTE to years: T = days/365
            4. Evaluate polynomial: IV = β₀ + β₁*m + β₂*m² + ...
            5. Bound result to [0.01, 5.0]
        """
        if not self.is_fitted:
            raise ValueError("Must call fit() before get_vol()")
        
        # TODO: Implement
        pass
    
    def plot_surface(self) -> None:
        """
        Plot the fitted volatility surface.
        
        TODO: OPTIONAL - For visualization in analysis phase
        Create 3D surface plot of IV vs (moneyness, time)
        """
        pass


def fit_volatility_surface(
    option_data: pd.DataFrame,
    method: str = 'polynomial',
    degree: int = 2
) -> VolatilitySurface:
    """
    Convenience function to fit and return a volatility surface.
    
    Args:
        option_data: DataFrame with option chain data
        method: Fitting method ('polynomial' or 'svi')
        degree: Polynomial degree for polynomial method
        
    Returns:
        Fitted VolatilitySurface object
        
    Example:
        >>> surface = fit_volatility_surface(spy_options)
        >>> vol = surface.get_vol(strike=295, underlying_price=300, days_to_expiry=30)
        >>> print(f"Fitted vol: {vol:.2%}")
        
    TODO: Implement convenience wrapper
    
    Hint: Create VolatilitySurface object, call fit(), return it
    """
    # TODO: Implement
    pass


# SVI Model (OPTIONAL - Advanced)

def svi_formula(k: np.ndarray, a: float, b: float, rho: float, m: float, sigma: float) -> np.ndarray:
    """
    SVI (Stochastic Volatility Inspired) volatility formula.
    
    Total variance w(k) = a + b[ρ(k-m) + sqrt((k-m)² + σ²)]
    
    Where:
        k = log-moneyness
        a = overall level
        b = slope
        ρ = skew (correlation)
        m = center (ATM location)
        σ = curvature
    
    Args:
        k: Log-moneyness array
        a, b, rho, m, sigma: SVI parameters
        
    Returns:
        Total variance (IV²*T) for each moneyness level
        
    TODO: OPTIONAL - Implement SVI formula
    This is more advanced and industry-standard, but polynomial is fine
    for your project. Implement this if you want extra polish.
    """
    pass


if __name__ == "__main__":
    # Test your implementation
    print("="*60)
    print("HISTORICAL VOLATILITY TEST")
    print("="*60)
    
    # Load SPY price data
    try:
        prices_df = pd.read_parquet('../data/processed/spy_prices.parquet')
        prices = prices_df['close']
        
        print(f"Loaded {len(prices)} days of SPY prices")
        print(f"Date range: {prices_df['date'].min()} to {prices_df['date'].max()}")
        
        # Calculate 30-day realized vol
        rv_30 = calculate_realized_volatility(prices, window=30)
        
        if rv_30 is not None:
            print(f"\n30-day Realized Volatility:")
            print(f"  Mean: {rv_30.mean():.2%}")
            print(f"  Std: {rv_30.std():.2%}")
            print(f"  Min: {rv_30.min():.2%}")
            print(f"  Max: {rv_30.max():.2%}")
            print("\n✓ Function implemented!")
        else:
            print("\n❌ Not implemented yet")
            
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Make sure spy_prices.parquet exists")

