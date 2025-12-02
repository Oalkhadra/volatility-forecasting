"""
Strategy D: Volatility Forecasting Models

This module implements statistical models to forecast future volatility:
- GARCH(1,1): Industry standard for vol forecasting
- EWMA: Exponentially weighted moving average (simpler than GARCH)
- HAR: Heterogeneous Autoregressive (captures different horizons)

The hypothesis: Sophisticated statistical models predict realized vol better
than naive historical averages or the market's implied vol.
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple
from scipy.optimize import minimize


class GARCHForecaster:
    """
    GARCH(1,1) volatility forecasting model.
    
    The model:
        σ²_t = ω + α*r²_t-1 + β*σ²_t-1
        
    Where:
        ω = long-run variance level
        α = reaction to recent shocks (ARCH term)
        β = persistence of variance (GARCH term)
        α + β < 1 for stationarity
        
    GARCH captures:
    - Volatility clustering (high vol follows high vol)
    - Mean reversion (vol returns to long-run average)
    - Fat tails in return distribution
    """
    
    def __init__(self):
        """Initialize GARCH model."""
        self.omega = None  # Long-run variance
        self.alpha = None  # ARCH coefficient
        self.beta = None   # GARCH coefficient
        self.current_variance = None
        self.is_fitted = False
        
    def fit(self, returns: pd.Series) -> None:
        """
        Fit GARCH(1,1) model to return series.
        
        Args:
            returns: pandas Series of returns (typically log returns)
            
        Estimates parameters ω, α, β by maximum likelihood.
        
        TODO: Implement GARCH fitting
        
        Two approaches:
        
        APPROACH 1 (EASIER): Use arch library
            from arch import arch_model
            model = arch_model(returns, vol='Garch', p=1, q=1)
            result = model.fit(disp='off')
            self.omega = result.params['omega']
            self.alpha = result.params['alpha[1]']
            self.beta = result.params['beta[1]']
            
        APPROACH 2 (LEARNING): Implement from scratch
            1. Initialize: σ²_0 = var(returns)
            2. For each return r_t:
               σ²_t = ω + α*r²_t-1 + β*σ²_t-1
            3. Log-likelihood: L = Σ[-0.5*ln(2π*σ²_t) - r²_t/(2σ²_t)]
            4. Use scipy.optimize to maximize L over (ω, α, β)
            5. Constraints: ω>0, α>0, β>0, α+β<1
            
        Interview Question: "Explain GARCH parameters"
            ω controls long-run vol level. α controls how much recent
            shocks affect vol (ARCH effect). β controls persistence 
            (how long high vol lasts). α+β close to 1 means highly
            persistent volatility - typical for financial markets.
            
        Recommendation: Use arch library for Phase 3 to save time.
        Implement from scratch if you want deeper understanding.
        """
        # TODO: Implement GARCH fitting
        # Install: pip install arch
        pass
    
    def forecast(self, horizon: int = 1) -> float:
        """
        Forecast volatility h steps ahead.
        
        Args:
            horizon: Forecast horizon in days
            
        Returns:
            Forecasted annualized volatility
            
        GARCH(1,1) multi-step forecast:
            E[σ²_t+h] = E[σ²] + (α+β)^h * (σ²_t - E[σ²])
            
        Where:
            E[σ²] = ω / (1 - α - β) is unconditional variance
            σ²_t is current conditional variance
            
        TODO: Implement GARCH forecast
        
        Hints:
            1. Check if fitted
            2. Long-run var: lr_var = ω / (1 - α - β)
            3. Forecast: var_forecast = lr_var + (α+β)^h * (current_var - lr_var)
            4. Return: sqrt(var_forecast * 252) for annualized vol
        """
        if not self.is_fitted:
            raise ValueError("Must call fit() before forecast()")
        
        # TODO: Implement
        pass
    
    def update_and_forecast(self, new_return: float) -> float:
        """
        Update model with new return and generate forecast.
        
        Args:
            new_return: Most recent return
            
        Returns:
            Next period vol forecast
            
        TODO: Implement GARCH update
        
        Hints:
            1. Update variance: σ²_t = ω + α*r²_t-1 + β*σ²_t-1
            2. Store as self.current_variance
            3. Return forecast(horizon=1)
        """
        # TODO: Implement
        pass


class EWMAForecaster:
    """
    Exponentially Weighted Moving Average volatility model.
    
    Simpler than GARCH but captures persistence:
        σ²_t = λ*σ²_t-1 + (1-λ)*r²_t-1
        
    Where λ is the decay factor (typically 0.94 for daily data).
    RiskMetrics uses λ=0.94 as standard.
    """
    
    def __init__(self, lambda_decay: float = 0.94):
        """
        Initialize EWMA forecaster.
        
        Args:
            lambda_decay: Decay factor (0 < λ < 1)
                - λ = 0.94: RiskMetrics standard (roughly 17-day half-life)
                - Higher λ: More weight on past (slower decay)
                - Lower λ: More reactive to recent moves
        """
        self.lambda_decay = lambda_decay
        self.current_variance = None
        
    def fit(self, returns: pd.Series) -> None:
        """
        Initialize EWMA with return series.
        
        Args:
            returns: Historical returns
            
        TODO: Implement EWMA initialization
        
        Hints:
            1. Start with sample variance: var_0 = var(returns)
            2. Iterate through returns updating: 
               var_t = λ*var_t-1 + (1-λ)*r²_t-1
            3. Store final variance as self.current_variance
        """
        # TODO: Implement
        pass
    
    def update(self, new_return: float) -> float:
        """
        Update EWMA variance with new return and return vol forecast.
        
        Args:
            new_return: New return observation
            
        Returns:
            Current volatility estimate (annualized)
            
        TODO: Implement EWMA update
        
        Hints:
            1. If first observation: current_variance = return²
            2. Otherwise: variance = λ*variance_old + (1-λ)*return²
            3. Return sqrt(variance * 252) for annualized vol
            
        Interview Question: "Why EWMA instead of simple moving average?"
            EWMA gives exponentially declining weights to older data, which
            better captures the time-varying nature of volatility. Recent
            observations matter more, but all historical data contributes.
            Simple MA gives equal weight, which is unrealistic for vol.
        """
        # TODO: Implement
        pass
    
    def forecast(self, horizon: int = 1) -> float:
        """
        Forecast volatility h steps ahead.
        
        For EWMA, the forecast is constant (martingale property):
            E[σ²_t+h | Info_t] = σ²_t for all h > 0
        
        Args:
            horizon: Forecast horizon (doesn't matter for EWMA)
            
        Returns:
            Forecasted volatility (annualized)
            
        TODO: Implement EWMA forecast
        
        Hint: Just return sqrt(current_variance * 252)
        """
        # TODO: Implement
        pass


# Comparison function

def compare_forecasters(
    returns: pd.Series,
    realized_vols: pd.Series,
    train_size: int = 252
) -> pd.DataFrame:
    """
    Compare different volatility forecasting methods.
    
    Args:
        returns: Return series
        realized_vols: Actual realized volatilities
        train_size: Size of training window
        
    Returns:
        DataFrame comparing forecast accuracy of different methods
        
    TODO: OPTIONAL - For Phase 5 analysis
    This will be useful when comparing Strategy D variants.
    """
    pass


if __name__ == "__main__":
    # Test your implementation
    print("="*60)
    print("VOLATILITY FORECASTING TEST")
    print("="*60)
    
    # Load price data and calculate returns
    try:
        prices_df = pd.read_parquet('../data/processed/spy_prices.parquet')
        prices = prices_df['close']
        returns = np.log(prices / prices.shift(1)).dropna()
        
        print(f"Loaded {len(returns)} returns")
        print(f"Return stats:")
        print(f"  Mean: {returns.mean():.4f}")
        print(f"  Std: {returns.std():.4f}")
        print(f"  Annualized vol: {returns.std() * np.sqrt(252):.2%}")
        
        # Test EWMA (simpler)
        print("\n" + "-"*60)
        print("EWMA Forecaster")
        print("-"*60)
        ewma = EWMAForecaster(lambda_decay=0.94)
        ewma.fit(returns)
        
        if ewma.current_variance is not None:
            forecast_vol = ewma.forecast()
            print(f"EWMA forecast: {forecast_vol:.2%}")
            print("✓ EWMA implemented!")
        else:
            print("❌ EWMA not implemented yet")
        
        # Test GARCH
        print("\n" + "-"*60)
        print("GARCH Forecaster")
        print("-"*60)
        garch = GARCHForecaster()
        garch.fit(returns)
        
        if garch.is_fitted:
            forecast_vol = garch.forecast()
            print(f"GARCH forecast: {forecast_vol:.2%}")
            print(f"Parameters: ω={garch.omega:.6f}, α={garch.alpha:.4f}, β={garch.beta:.4f}")
            print(f"Persistence: α+β = {garch.alpha + garch.beta:.4f}")
            print("✓ GARCH implemented!")
        else:
            print("❌ GARCH not implemented yet")
            
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure spy_prices.parquet exists")
