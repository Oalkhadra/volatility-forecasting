"""
Strategy D: Volatility Forecasting Models

This module implements two forecasting approaches:
- GARCH(1,1): Classical econometric forecasting (industry standard)
- XGBoost ML: Modern machine learning approach

Comparison: Traditional parametric (GARCH) vs non-parametric (ML)

The hypothesis: Statistical forecasting models predict realized vol better
than naive historical averages or the market's implied vol.
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from arch import arch_model
from .historical import calculate_realized_volatility

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

        """
        model = arch_model(returns, vol='GARCH', p=1, q=1)
        result = model.fit(disp='off')

        self.omega = result.params['omega']
        self.alpha = result.params['alpha[1]']
        self.beta = result.params['beta[1]']
        # Keep current_variance in percentage-squared scale to match omega
        # (returns are in percentage scale, so arch parameters are too)
        self.current_variance = result.conditional_volatility.iloc[-1] ** 2
        self.is_fitted = True

    def forecast(self, horizon: int = 1) -> float:
        """
        Forecast volatility h steps ahead.
        
        Args:
            horizon: Forecast horizon in days
            
        Returns:
            Forecasted annualized volatility (in decimal scale, e.g., 0.20 for 20%)
            
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
        
        omega = self.omega
        alpha = self.alpha
        beta = self.beta
        current_var = self.current_variance

        # Calculate forecasted variance (in percentage-squared scale)
        long_run_var = omega / (1 - alpha - beta)
        var_forecast = long_run_var + (alpha + beta) ** horizon * (current_var - long_run_var)

        # Convert to annualized volatility in decimal scale
        # sqrt(var * 252) gives percentage scale, divide by 100 for decimal
        return np.sqrt(var_forecast * 252) / 100


class MLForecaster:
    """
    Machine Learning volatility forecaster using XGBoost.
    
    Uses gradient boosting to predict future realized volatility based on:
    - Lagged realized volatilities (multiple windows)
    - Recent returns (momentum/reversal)
    - Higher moments (skew, kurtosis)
    - GARCH forecast (as a feature)
    
    This captures non-linear relationships and interactions that 
    parametric models like GARCH might miss.
    """
    
    def __init__(self, horizon: int = 7):
        """
        Initialize ML forecaster.
        
        Args:
            horizon: Forecast horizon in days (default 7 for weekly)
        """
        self.model = xgb.XGBRegressor(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            objective='reg:squarederror'
        )
        
        self.horizon = horizon
        self.is_fitted = False
        self.feature_names = []
        self.scaler = None  # For feature scaling if needed
        
    def _create_features(self, returns: pd.Series, lookback: int = 30) -> np.ndarray:
        """
        Create feature matrix from returns.
        
        Features:
        1. Lagged realized vols (5d, 10d, 20d, 30d)
        2. Recent returns (1d, 5d, 10d)
        3. Higher moments (30d skew, kurtosis)
        4. Return autocorrelation
        5. GARCH forecast (optional - as a feature)
        
        Args:
            returns: Return series
            lookback: How far back to look for features
            
        Returns:
            Feature array [rv_5d, rv_10d, rv_20d, rv_30d, ret_1d, ...]
            
        TODO: Implement feature engineering
        
        Hints:
            1. Calculate multiple rolling vols (different windows)
            2. Calculate recent cumulative returns
            3. Calculate rolling skew and kurtosis
            4. Make sure all features use ONLY past data (no leakage!)
            5. Handle NaNs (first few observations)
            
        Interview Question: "What features did you use for vol forecasting?"
            I used lagged realized volatilities at multiple horizons (5, 10, 20, 30 days)
            to capture short and long-term patterns. Recent returns capture momentum
            effects. Higher moments like skewness indicate tail risk, which relates
            to future vol. The multi-scale approach lets the model learn which
            horizon is most predictive.
        """
        n = len(returns)
        features_list = []
        
        # For each time point (after warmup period)
        for i in range(lookback, n):
            # Get returns UP TO time i
            past_returns = returns.iloc[:i]
            
            # Calculate features using ONLY past data
            rv_5 = calculate_realized_volatility(past_returns, window=5).iloc[-1]
            rv_10 = calculate_realized_volatility(past_returns, window=10).iloc[-1]
            rv_20 = calculate_realized_volatility(past_returns, window=20).iloc[-1]
            rv_30 = calculate_realized_volatility(past_returns, window=30).iloc[-1]
            
            # Recent returns
            ret_1d = past_returns.iloc[-1]
            ret_5d = past_returns.iloc[-5:].sum()
            ret_10d = past_returns.iloc[-10:].sum()
            
            # Higher moments (last 30 days)
            skew_30 = past_returns.iloc[-30:].skew()
            kurt_30 = past_returns.iloc[-30:].kurtosis()
            
            # Build feature dict for this time point
            features = {
                'rv_5d': rv_5,
                'rv_10d': rv_10,
                'rv_20d': rv_20,
                'rv_30d': rv_30,
                'ret_1d': ret_1d,
                'ret_5d': ret_5d,
                'ret_10d': ret_10d,
                'skew_30d': skew_30,
                'kurt_30d': kurt_30,
            }
            
            features_list.append(features)
        
        # Convert to DataFrame
        df = pd.DataFrame(features_list)
        
        # Store feature names
        self.feature_names = df.columns.tolist()
        
        return df

    def _create_targets(self, returns: pd.Series, lookback: int = 30, horizon: int = 7) -> np.ndarray:
        """
        Create target vector: forward-looking realized vol.
        
        Target = realized vol over next 'horizon' days
        
        Args:
            returns: Return series
            
        Returns:
            Array of forward realized vols
            
        """
        # 1. Calculate rolling volatility (annualized)
        rolling_vol = returns.rolling(window=horizon).std() * np.sqrt(252)
        
        # 2. Shift BACKWARDS so that the value at time t represents the vol
        # realized over the *next* 'horizon' days (t to t+horizon)
        targets = rolling_vol.shift(-horizon)
        
        # 3. Slice to match the feature generation
        # Features start at 'lookback', so targets must also start there
        # We drop the NaNs at the end created by the shift
        target_slice = targets.iloc[lookback : -horizon]
        
        return target_slice.values
    
    def fit(self, returns: pd.Series, min_train_size: int = 60) -> None:
        """
        Fit ML model to return series.
        
        Uses all available data to train. For walk-forward validation,
        call this method with expanding window in backtesting loop.
        
        Args:
            returns: Historical returns
            min_train_size: Minimum returns needed to fit
                    
        Hints:
            1. Create features for each time period
            2. Create targets (forward realized vol)
            3. Remove NaN rows
            4. Split into train/validation if desired (optional)
            5. Fit XGBoost model
            6. Set self.is_fitted = True
            
        Note: For backtesting, you'll call fit() with expanding window
        each week to ensure no lookahead bias.
        """
        if len(returns) < min_train_size:
            raise ValueError(f"Need at least {min_train_size} returns to fit ML model")
        
        # 1. Create Features and Targets
        # We use our helper methods. 
        # Note: _create_targets returns a numpy array aligned with features
        X = self._create_features(returns, lookback=30)
        y = self._create_targets(returns, lookback=30, horizon=self.horizon)

        # 2. Align lengths
        # _create_features usually returns df starting at index 'lookback'
        # _create_targets (the vectorized one) returns array of length (N - lookback - horizon)
        
        # We trim X to match y's length (drop the last 'horizon' rows of features
        # because we don't have a target for them yet!)
        X_aligned = X.iloc[:len(y)]
        
        # 3. Fit the model
        self.model.fit(X_aligned, y)
        self.is_fitted = True
        
        # Save feature names for later importance plotting
        self.feature_names = X.columns.tolist()

    def forecast(self, returns: pd.Series) -> float:
        """
        Generate volatility forecast using most recent data.
        
        Args:
            returns: Historical returns (to create features from)
            
        Returns:
            Forecasted annualized volatility for next 'horizon' days
            
        TODO: Implement ML prediction
        
        Hints:
            1. Check if model is fitted
            2. Create features from most recent returns
            3. Use self.model.predict() to generate forecast
            4. Return annualized vol
        """
        if not self.is_fitted:
            raise ValueError("Must call fit() before forecast()")
        
        # Create features for the most recent timestamp
        # We calculate features for the entire series and take the last row
        # In production, this would be optimized to only calc the last row
        all_features = self._create_features(returns, lookback=30)
        
        if len(all_features) == 0:
            raise ValueError("Not enough data to generate forecast features")
            
        current_features = all_features.iloc[-1:] # Keep as DataFrame to preserve names
        
        # Predict
        prediction = self.model.predict(current_features)[0]
        
        return float(prediction)
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importances from trained model.
        
        Returns:
            DataFrame with features and their importance scores
            
        TODO: OPTIONAL - For analysis phase
        This is useful for understanding what drives vol predictions.
        """
        if not self.is_fitted:
            raise ValueError("Must call fit() first")
        
        # XGBoost has built-in feature importance
        importance = self.model.feature_importances_
        
        return pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)


# Comparison and helper functions

def compare_forecasters(
    returns: pd.Series,
    realized_vols: pd.Series,
    train_size: int = 60
) -> pd.DataFrame:
    """
    Compare GARCH vs ML forecast accuracy.
    
    Args:
        returns: Return series
        realized_vols: Actual realized volatilities (to compare against)
        train_size: Minimum training window size
        
    Returns:
        DataFrame comparing forecast accuracy metrics:
            - RMSE: Root mean squared error
            - MAE: Mean absolute error
            - Correlation: Forecast vs realized
            - Hit rate: % of times forecast > realized when actual > avg
        
    TODO: OPTIONAL - For Phase 5 analysis
    This generates performance comparison of GARCH vs ML.
    Useful for understanding which forecasting method works better.
    """
    pass


def calculate_forecast_accuracy(
    forecasts: pd.Series,
    actuals: pd.Series
) -> dict:
    """
    Calculate accuracy metrics for vol forecasts.
    
    Args:
        forecasts: Forecasted volatilities
        actuals: Actual realized volatilities
        
    Returns:
        Dictionary with accuracy metrics
        
    TODO: OPTIONAL - For Phase 5 analysis
    """
    # Remove NaN values
    mask = ~(forecasts.isna() | actuals.isna())
    f = forecasts[mask]
    a = actuals[mask]
    
    # Calculate metrics
    rmse = np.sqrt(np.mean((f - a)**2))
    mae = np.mean(np.abs(f - a))
    correlation = np.corrcoef(f, a)[0, 1]
    
    # Directional accuracy (did we predict high/low correctly?)
    median_vol = a.median()
    correct_direction = ((f > median_vol) == (a > median_vol)).sum()
    hit_rate = correct_direction / len(a)
    
    return {
        'RMSE': rmse,
        'MAE': mae,
        'Correlation': correlation,
        'Hit_Rate': hit_rate,
        'N_Observations': len(a)
    }


if __name__ == "__main__":
    # Test your implementation
    print("="*60)
    print("VOLATILITY FORECASTING TEST")
    print("="*60)
    
    # Load price data and calculate returns
    try:
        prices_df = pd.read_parquet('././data/processed/spy_prices.parquet')
        prices = prices_df['close']
        returns = np.log(prices / prices.shift(1)).dropna()
        
        print(f"Loaded {len(returns)} returns")
        print(f"Return stats:")
        print(f"  Mean: {returns.mean():.4f}")
        print(f"  Std: {returns.std():.4f}")
        print(f"  Annualized vol: {returns.std() * np.sqrt(252):.2%}")
        
        # Test GARCH
        print("\n" + "-"*60)
        print("GARCH(1,1) Forecaster - Classical Econometric")
        print("-"*60)
        garch = GARCHForecaster()
        garch.fit(returns)
        
        if garch.is_fitted:
            forecast_vol = garch.forecast(horizon=7)
            print(f"✓ GARCH fitted successfully!")
            print(f"  Parameters: ω={garch.omega:.6f}, α={garch.alpha:.4f}, β={garch.beta:.4f}")
            print(f"  Persistence (α+β): {garch.alpha + garch.beta:.4f}")
            print(f"  Long-run vol: {np.sqrt((garch.omega/(1-garch.alpha-garch.beta))*252):.2%}")
            print(f"  7-day forecast: {forecast_vol:.2%}")
        else:
            print("❌ GARCH not implemented yet")
        
        # Test ML
        print("\n" + "-"*60)
        print("XGBoost ML Forecaster - Modern Machine Learning")
        print("-"*60)
        
        # Use first 80% for training, last 20% for testing
        train_size = int(len(returns) * 0.8)
        train_returns = returns.iloc[:train_size]
        test_returns = returns  # Use all for forecast
        
        ml = MLForecaster(horizon=7)
        ml.fit(train_returns)
        
        if ml.is_fitted:
            forecast_vol = ml.forecast(test_returns)
            print(f"✓ ML model fitted successfully!")
            print(f"  Training size: {len(train_returns)} days")
            print(f"  Number of features: {len(ml.feature_names)}")
            print(f"  7-day forecast: {forecast_vol:.2%}")
            
            # Show feature importance
            try:
                importance = ml.get_feature_importance()
                print(f"\n  Top 3 Important Features:")
                for i, row in importance.head(3).iterrows():
                    print(f"    {row['feature']}: {row['importance']:.4f}")
            except:
                pass
        else:
            print("❌ ML model not implemented yet")
        
        # Comparison
        if garch.is_fitted and ml.is_fitted:
            # Get fresh forecasts
            garch_fc = garch.forecast(horizon=7)
            ml_fc = ml.forecast(test_returns)
            
            # Calculate actual realized volatility for the next 7 days
            # We need the data following the test period to calculate "actual"
            # Note: In a real backtest, we wouldn't know this yet!
            # We take the last 7 days of the loaded data if available, or say "Unknown"
            # But here 'test_returns' IS the full dataset, so we can't calculate "next" 7 days
            # unless we held out data. 
            
            # Let's calculate the realized vol of the LAST 7 days to compare 
            # with what a forecast made 7 days ago WOULD have predicted (just for context)
            last_7d_realized = returns.iloc[-7:].std() * np.sqrt(252)
            
            print("\n" + "="*60)
            print("FORECAST SNAPSHOT (Annualized)")
            print("="*60)
            print(f"  GARCH forecast (Next 7d):     {garch_fc:.2%}")
            print(f"  ML forecast (Next 7d):        {ml_fc:.2%}")
            print(f"  Realized Vol (Last 7d):       {last_7d_realized:.2%} (for context)")
            
            print(f"\n  Both methods ready for backtesting!")
            print("="*60)
            
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure spy_prices.parquet exists")
