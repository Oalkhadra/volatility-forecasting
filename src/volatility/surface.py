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
        # Extract variables
        S = option_data['underlying_price'].values
        K = option_data['strike'].values
        T = option_data['days_to_expiry'].values / 365 # Convert to years
        market_ivs = option_data['vol'].values

        m = np.log(K/S)
        # Build feature matrix [1, m, m², T, m*T]
        X = np.column_stack([
            np.ones(len(m)),
            m,
            m**2,
            T,
            m*T
        ]) 

        # Fit using LSE
        coeffs, residuals, rank, s = np.linalg.lstsq(X, market_ivs, rcond=None)

        # Store results
        self.params = coeffs
        self.is_fitted = True
        
        # Calculate fit quality
        fitted_ivs = X @ coeffs
        self.r_squared = 1 - np.sum((market_ivs - fitted_ivs)**2) / np.sum((market_ivs - market_ivs.mean())**2)
        self.rmse = np.sqrt(np.mean((market_ivs - fitted_ivs)**2))

    
    def get_vol(
        self,
        strike: float,
        underlying_price: float,
        days_to_expiry: float,
    ) -> float:
        """
        Get fitted volatility for a specific option.
        
        Args:
            strike: Strike price
            underlying_price: Spot price
            days_to_expiry: Days to expiration
            
        Returns:
            Fitted implied volatility
        """
        if not self.is_fitted:
            raise ValueError("Must call fit() before get_vol()")
        
        # Calculate log-moneyness
        m = np.log(strike / underlying_price)
        
        # Convert days to years
        T = days_to_expiry / 365
        
        # Evaluate polynomial: IV = β₀ + β₁*m + β₂*m² + β₃*T + β₄*m*T
        fitted_iv = (
            self.params[0] +           # β₀ (intercept)
            self.params[1] * m +       # β₁*m
            self.params[2] * m**2 +    # β₂*m²
            self.params[3] * T +       # β₃*T
            self.params[4] * m * T     # β₄*m*T
        )
        
        # Bound to reasonable range (1% to 500%)
        fitted_iv = np.clip(fitted_iv, 0.01, 5.0)
        
        return fitted_iv
        
    
    def plot_surface(self, S: float = 100) -> None:
        """
        Plot the fitted volatility surface.
        
        Args:
            S: Underlying price for plotting (default 100 for normalized view)
        """
        if not self.is_fitted:
            raise ValueError("Must call fit() before plot_surface()")
        
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
        except ImportError:
            print("matplotlib not installed. Install with: pip install matplotlib")
            return
        
        # Create grid of strikes and maturities
        moneyness_range = np.linspace(-0.15, 0.15, 30)  # -15% to +15%
        dte_range = np.linspace(7, 60, 20)  # 7 to 60 days
        
        M, T_days = np.meshgrid(moneyness_range, dte_range)
        
        # Calculate fitted IVs for each point
        IV = np.zeros_like(M)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                m = M[i, j]
                T = T_days[i, j] / 365
                
                # Evaluate polynomial
                IV[i, j] = (
                    self.params[0] +
                    self.params[1] * m +
                    self.params[2] * m**2 +
                    self.params[3] * T +
                    self.params[4] * m * T
                )
        
        # Create 3D plot
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot surface
        surf = ax.plot_surface(
            M, T_days, IV,
            cmap='viridis',
            alpha=0.8,
            edgecolor='none'
        )
        
        # Labels
        ax.set_xlabel('Log-Moneyness ln(K/S)', fontsize=10)
        ax.set_ylabel('Days to Expiry', fontsize=10)
        ax.set_zlabel('Implied Volatility', fontsize=10)
        ax.set_title('Fitted Volatility Surface', fontsize=12, fontweight='bold')
        
        # Color bar
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='IV')
        
        # Add fit quality text
        textstr = f'R² = {self.r_squared:.4f}\nRMSE = {self.rmse:.4f}'
        ax.text2D(0.02, 0.98, textstr, transform=ax.transAxes,
                  fontsize=10, verticalalignment='top',
                  bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.show()
        
        return fig


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
        
    """
    # Create surface object
    surface = VolatilitySurface(method=method)
    
    # Fit to data
    surface.fit(option_data)
    
    return surface


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
    print("VOLATILITY SURFACE TEST")
    print("="*60)
    
    # Load options data
    try:
        options_df = pd.read_parquet('././data/processed/spy_options.parquet')
        
        print(f"Loaded {len(options_df)} options")
        print(f"Date range: {options_df['date'].min()} to {options_df['date'].max()}")
        
        # Fit surface to one week of data
        test_date = options_df['date'].iloc[0]
        test_data = options_df[options_df['date'] == test_date].copy()
        
        print(f"\nFitting surface to {len(test_data)} options on {test_date}")
        print(f"  Calls: {(test_data['call_put'] == 'Call').sum()}")
        print(f"  Puts: {(test_data['call_put'] == 'Put').sum()}")
        
        # Get underlying price
        S = test_data['underlying_price'].iloc[0]
        print(f"  Underlying price: ${S:.2f}")
        
        # Create and fit surface
        surface = VolatilitySurface(method='polynomial')
        surface.fit(test_data)
        
        if surface.is_fitted:
            print("\n✓ Surface fitted successfully!")
            
            # Test get_vol on different strikes
            print("\nTesting fitted vols at different strikes:")
            test_strikes = [S * 0.95, S, S * 1.05]  # OTM, ATM, ITM for calls
            
            for K in test_strikes:
                fitted_vol = surface.get_vol(
                    strike=K,
                    underlying_price=S,
                    days_to_expiry=30
                )
                moneyness_pct = (K / S - 1) * 100
                print(f"  K=${K:.2f} ({moneyness_pct:+.1f}%): σ={fitted_vol:.2%}")
            
            # Compare fitted vs market for actual options
            print("\nComparing fitted vs market IVs:")
            sample_options = test_data.head(5)
            for idx, opt in sample_options.iterrows():
                fitted_vol = surface.get_vol(
                    opt['strike'],
                    opt['underlying_price'],
                    opt['days_to_expiry']
                )
                market_vol = opt['vol']
                diff = fitted_vol - market_vol
                print(f"  K=${opt['strike']:.0f} {opt['call_put']}: "
                      f"Market={market_vol:.2%}, Fitted={fitted_vol:.2%}, "
                      f"Diff={diff:+.2%}")
            
            print("\n✓ Volatility surface implementation complete!")
        else:
            print("\n❌ Surface fitting not implemented yet")
            
    except Exception as e:
        print(f"\nError: {e}")
        print("Make sure spy_options.parquet exists")
        import traceback
        traceback.print_exc()

