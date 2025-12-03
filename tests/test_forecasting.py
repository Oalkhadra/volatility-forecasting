"""
Test script for volatility forecasting models.
This script benchmarks GARCH vs ML models against realized volatility.
"""

import numpy as np
import pandas as pd
import pytest
from src.volatility.forecasting import GARCHForecaster, MLForecaster
from src.volatility.historical import calculate_realized_volatility

# Set seed for reproducibility
np.random.seed(42)

@pytest.fixture
def mock_returns():
    """Generate synthetic returns for testing."""
    n_days = 500
    # GARCH(1,1) process simulation
    omega = 0.000001
    alpha = 0.1
    beta = 0.85
    
    returns = np.zeros(n_days)
    sigma2 = np.zeros(n_days)
    sigma2[0] = omega / (1 - alpha - beta)
    
    for t in range(1, n_days):
        sigma2[t] = omega + alpha * (returns[t-1]**2) + beta * sigma2[t-1]
        returns[t] = np.sqrt(sigma2[t]) * np.random.normal()
        
    return pd.Series(returns)

def calculate_rmse(predictions, actuals):
    """Calculate Root Mean Squared Error."""
    return np.sqrt(np.mean((predictions - actuals) ** 2))

def calculate_mae(predictions, actuals):
    """Calculate Mean Absolute Error."""
    return np.mean(np.abs(predictions - actuals))

class TestForecastingAccuracy:
    """Test suite for comparing forecasting models."""

    def test_garch_convergence(self, mock_returns):
        """Test that GARCH model converges and produces reasonable forecasts."""
        model = GARCHForecaster()
        model.fit(mock_returns)
        
        assert model.is_fitted
        forecast = model.forecast(horizon=7)
        
        # Sanity checks
        assert 0 < forecast < 1.0, f"GARCH forecast {forecast} out of reasonable bounds"
        assert model.alpha + model.beta < 1.0, "GARCH process should be stationary"

    def test_ml_training(self, mock_returns):
        """Test that ML model trains and produces forecasts."""
        model = MLForecaster(horizon=7)
        # Use first 400 days for training
        train_returns = mock_returns.iloc[:400]
        
        model.fit(train_returns)
        
        assert model.is_fitted
        forecast = model.forecast(train_returns)
        
        assert isinstance(forecast, float)
        assert 0 < forecast < 1.0, f"ML forecast {forecast} out of reasonable bounds"
        assert len(model.feature_names) > 0

    def test_expanding_window_backtest(self, mock_returns):
        """
        Run a mini backtest comparing GARCH and ML.
        This mimics the Phase 4 backtesting engine logic.
        """
        horizon = 7
        start_idx = 300  # Start backtest after 300 days
        end_idx = 400    # Run for 100 days (approx 14 weeks)
        
        garch_preds = []
        ml_preds = []
        actuals = []
        
        # Weekly stepping (every 7 days)
        for t in range(start_idx, end_idx, 7):
            # 1. Define training window (expanding)
            train_returns = mock_returns.iloc[:t]
            
            # 2. Define "Actual" target (future realized vol)
            # We look ahead 7 days to calculate what actually happened
            future_returns = mock_returns.iloc[t:t+horizon]
            if len(future_returns) < horizon:
                break
                
            # Actual realized vol for the NEXT 7 days
            realized_vol = future_returns.std() * np.sqrt(252)
            actuals.append(realized_vol)
            
            # 3. Forecast - GARCH
            garch = GARCHForecaster()
            garch.fit(train_returns)
            garch_preds.append(garch.forecast(horizon=horizon))
            
            # 4. Forecast - ML
            ml = MLForecaster(horizon=horizon)
            ml.fit(train_returns)
            ml_preds.append(ml.forecast(train_returns))

        # Convert to arrays
        garch_preds = np.array(garch_preds)
        ml_preds = np.array(ml_preds)
        actuals = np.array(actuals)
        
        # 5. Calculate Metrics
        garch_rmse = calculate_rmse(garch_preds, actuals)
        ml_rmse = calculate_rmse(ml_preds, actuals)
        
        print(f"\nBacktest Results ({len(actuals)} weeks):")
        print(f"GARCH RMSE: {garch_rmse:.4f}")
        print(f"ML RMSE:    {ml_rmse:.4f}")
        
        # Basic validity checks
        assert len(actuals) > 5, "Should have at least a few test points"
        assert not np.isnan(garch_rmse), "GARCH RMSE should be valid"
        assert not np.isnan(ml_rmse), "ML RMSE should be valid"

if __name__ == "__main__":
    # Run directly with verbose output
    pytest.main([__file__, "-v", "-s"])

