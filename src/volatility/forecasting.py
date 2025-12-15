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
        model = arch_model(returns, vol='GARCH', p=1, q=1, dist='t')
        result = model.fit(disp='off')

        self.omega = result.params['omega']
        self.alpha = result.params['alpha[1]']
        self.beta = result.params['beta[1]']
        self.current_variance = result.conditional_volatility.iloc[-1] ** 2
        self.is_fitted = True

    def forecast(self, horizon: int = 30) -> float:
        """
        Forecast volatility h steps ahead.
        
        Args:
            horizon: Forecast horizon in days
            
        Returns:
            Forecasted annualized volatility (in decimal scale, e.g., 0.20 for 20%)

        """
        if not self.is_fitted:
            raise ValueError("Must call fit() before forecast()")
        
        omega = self.omega
        alpha = self.alpha
        beta = self.beta
        current_var = self.current_variance

        # Calculate forecasted variance (in percentage-squared scale)
        persistence = alpha + beta
        
        # Handle IGARCH case (alpha + beta >= 1) where long-run variance is undefined
        if persistence >= 0.9999:
            # Use simplified forecast: current_var * persistence^horizon
            var_forecast = current_var * (persistence ** horizon)
        else:
            long_run_var = omega / (1 - persistence)
            var_forecast = long_run_var + (persistence ** horizon) * (current_var - long_run_var)

        # Convert to annualized volatility in decimal scale
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
    
    Supports pre-training on historical data for faster backtesting.
    """
    
    def __init__(self, pretrained: bool = False):
        """
        Initialize ML forecaster.
        
        Args:
            pretrained: If True, model expects to be pretrained before backtest.
                       This skips daily retraining in update_vol_model().
        """
        self.model = xgb.XGBRegressor(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.01,
            random_state=115,
            objective=asymmetric_squared_error
        )
        
        self.is_fitted = False
        self.pretrained = pretrained  # Skip daily retraining if True
        self.feature_names = []
        self.scaler = None  # For feature scaling if needed
        self._last_returns = None  # Store returns for forecasting
        self._last_vix = None  # Store VIX for forecasting
    
    def _create_features(self, returns: pd.Series, vix: pd.Series, lookback: int = 30, horizon: int = 30) -> pd.DataFrame:
        returns = returns.reset_index(drop=True)
        vix = vix.reset_index(drop=True)


        # ===========================================
        # Lagged VIX
        # ===========================================
        yesterday_vix = vix.shift(1)

        # ===========================================
        # REALIZED VOLATILITY FEATURES
        # ===========================================
        rv_1d = (np.abs(returns) * np.sqrt(252)).shift(1)            # Yesterday's |return|
        rv_5d = (returns.rolling(5).std() * np.sqrt(252)).shift(1)   # Past 5 days (t-5 to t-1)
        rv_22d = (returns.rolling(22).std() * np.sqrt(252)).shift(1) # Past 22 days (t-22 to t-1)
        rv_30d = (returns.rolling(30).std() * np.sqrt(252)).shift(1) # Past 30 days (t-30 to t-1)
        rv_60d = (returns.rolling(60).std() * np.sqrt(252)).shift(1) # Past 60 days (t-60 to t-1)
        rv_120d = (returns.rolling(120).std() * np.sqrt(252)).shift(1) # Past 120 days (t-120 to t-1)


        # ===========================================
        # LEVERAGE EFFECT (Black 1976)
        # ===========================================
        ret_1d = returns.shift(1)  # Yesterday's return
        ret_22d = returns.rolling(22).sum().shift(1) # Cumulative return t-22 to t-1
        ret_30d = returns.rolling(30).sum().shift(1) # Cumulative return t-30 to t-1
        ret_60d = returns.rolling(60).sum().shift(1) # Cumulative return t-60 to t-1
        ret_120d = returns.rolling(120).sum().shift(1) # Cumulative return t-120 to t-1
        
        # ===========================================
        # ASYMMETRIC/SIGNED VOLATILITY (Patton & Sheppard 2015)
        # ===========================================
        neg_returns = returns.clip(upper=0)  # Negative returns only
        
        # Realized semivariance: sqrt(sum(r^2)) * sqrt(252) / 100
        rsv_neg_5d = (np.sqrt((neg_returns**2).rolling(5).sum()) * np.sqrt(252)).shift(1)

        # ===========================================
        # BUILD DATAFRAME
        # ===========================================
        df = pd.DataFrame({
            'yesterday_vix': yesterday_vix,
            'rv_1d': rv_1d,
            'rv_5d': rv_5d,
            'rv_22d': rv_22d,       
            'rv_30d': rv_30d,
            'rv_60d': rv_60d,
            'rv_120d': rv_120d,
            'ret_1d': ret_1d,
            'ret_22d': ret_22d,
            'ret_30d': ret_30d,
            'ret_60d': ret_60d,
            'ret_120d': ret_120d,
            'rsv_neg_5d': rsv_neg_5d,
            'horizon': horizon
        })
        
        # Trim to start after lookback (ensures enough history for all features)
        df = df.iloc[lookback:].reset_index(drop=True)

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
    
    def fit(self, returns: pd.Series, vix: pd.Series, min_train_size = 30, horizons = [30]) -> None:
        """
        Fit ML model to return series.
        
        Uses all available data to train. For walk-forward validation,
        call this method with expanding window in backtesting loop.
        
        Args:
            returns: Historical returns
            vix: Historical VIX
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
        # If pretrained and already fitted, just update the returns cache
        if self.pretrained and self.is_fitted:
            self._last_returns = returns.copy()
            self._last_vix = vix.copy()
            return
            
        if len(returns) < min_train_size:
            raise ValueError(f"Need at least {min_train_size} returns to fit ML model")
        
        # Store returns and VIX for forecasting
        self._last_returns = returns.copy()
        self._last_vix = vix.copy()
        
        # 1. Create Features and Targets
        all_X, all_y = [], []
        for h in horizons:
            X = self._create_features(returns, vix, horizon=h)
            y = self._create_targets(returns, horizon=h)
            
            # Align X and y: targets are shorter because they need forward data
            # X has (n - lookback) rows, y has (n - lookback - horizon) rows
            # Trim X from the end to match y
            n_samples = len(y)
            if n_samples > 0:
                X = X.iloc[:n_samples]
                all_X.append(X)
                all_y.append(y)
        
        if len(all_X) == 0:
            raise ValueError("Not enough data to create training samples")
        
        X_combined = pd.concat(all_X, ignore_index=True)
        y_combined = np.concatenate(all_y)
        
        # Remove rows with NaN values
        mask = ~(X_combined.isna().any(axis=1) | np.isnan(y_combined))
        X_combined = X_combined[mask]
        y_combined = y_combined[mask]
        
        if len(X_combined) == 0:
            raise ValueError("All training samples contain NaN values")
        
        self.model.fit(X_combined, y_combined)
        self.is_fitted = True
        
        # Save feature names for later importance plotting
        self.feature_names = X_combined.columns.tolist()
        
        print(f"  MLForecaster trained on {len(X_combined)} samples across {len(horizons)} horizons")
    
    def pretrain(self, prices_df: pd.DataFrame, vix_df: pd.Series, cutoff_date, horizons = [30]) -> None:
        """
        Pre-train the model on historical data before a cutoff date.
        
        This avoids look-ahead bias by only using data before the backtest starts.
        After pretraining, daily calls to fit() will only update the returns cache
        for forecasting, not retrain the model.
        
        Args:
            prices_df: DataFrame with 'date' and 'log_ret' columns
            cutoff_date: Train only on data before this date (typically backtest start)
            horizons: List of forecast horizons to train on
        """
        print(f"Pre-training MLForecaster on data before {cutoff_date}...")
        
        # Filter to data before cutoff
        prices_df = prices_df.copy()
        prices_df['date'] = pd.to_datetime(prices_df['date'])
        historical = prices_df[prices_df['date'] < cutoff_date]
        
        if len(historical) < 100:
            raise ValueError(f"Not enough historical data for pre-training. Have {len(historical)}, need at least 100")
        
        returns = historical['log_ret'].dropna()
        vix = vix_df['vix_close'].dropna()
        print(f"  Training on {len(returns)} historical returns")
        
        # Mark as pretrained so future fit() calls just update returns cache
        self.pretrained = True
        self.is_fitted = False  # Temporarily unset so fit() actually trains
        
        # Train the model
        self.fit(returns, vix, horizons=horizons)
        
        # Now mark as pretrained
        self.pretrained = True
        print(f"  Pre-training complete!")

    def _create_current_features(self, returns: pd.Series, vix: pd.Series, horizon: int = 30) -> pd.DataFrame:
        """
        Create features for ONLY the most recent timestamp (optimized for forecasting).
        
        This matches the exact feature format from _create_features() but only
        computes for the current time point (much faster for forecasting).
        
        Args:
            returns: Return series (needs at least 120 values for all features)
            vix: VIX series
            horizon: Forecast horizon
            
        Returns:
            Single-row DataFrame with features for the current time point
        """
        if len(returns) < 120:
            raise ValueError("Need at least 120 returns to create features")
        
        # ===========================================
        # Lagged VIX (yesterday's VIX)
        # ===========================================
        yesterday_vix = vix.iloc[-1]
        
        # ===========================================
        # REALIZED VOLATILITY FEATURES
        # ===========================================
        rv_1d = np.abs(returns.iloc[-1]) * np.sqrt(252)  # Yesterday's |return|
        rv_5d = returns.iloc[-5:].std() * np.sqrt(252)   # Past 5 days
        rv_22d = returns.iloc[-22:].std() * np.sqrt(252) # Past 22 days
        rv_30d = returns.iloc[-30:].std() * np.sqrt(252) # Past 30 days
        rv_60d = returns.iloc[-60:].std() * np.sqrt(252) # Past 60 days
        rv_120d = returns.iloc[-120:].std() * np.sqrt(252) # Past 120 days
        
        # ===========================================
        # LEVERAGE EFFECT (Black 1976)
        # ===========================================
        ret_1d = returns.iloc[-1]  # Yesterday's return
        ret_22d = returns.iloc[-22:].sum()  # Cumulative return last 22 days
        ret_30d = returns.iloc[-30:].sum()  # Cumulative return last 30 days
        ret_60d = returns.iloc[-60:].sum()  # Cumulative return last 60 days
        ret_120d = returns.iloc[-120:].sum()  # Cumulative return last 120 days
        
        # ===========================================
        # ASYMMETRIC/SIGNED VOLATILITY (Patton & Sheppard 2015)
        # ===========================================
        neg_returns = returns.iloc[-5:].clip(upper=0)  # Negative returns only (last 5 days)
        rsv_neg_5d = np.sqrt((neg_returns**2).sum()) * np.sqrt(252)
        
        # ===========================================
        # BUILD DATAFRAME (same column order as _create_features)
        # ===========================================
        features = {
            'yesterday_vix': yesterday_vix,
            'rv_1d': rv_1d,
            'rv_5d': rv_5d,
            'rv_22d': rv_22d,
            'rv_30d': rv_30d,
            'rv_60d': rv_60d,
            'rv_120d': rv_120d,
            'ret_1d': ret_1d,
            'ret_22d': ret_22d,
            'ret_30d': ret_30d,
            'ret_60d': ret_60d,
            'ret_120d': ret_120d,
            'rsv_neg_5d': rsv_neg_5d,
            'horizon': horizon
        }
        
        return pd.DataFrame([features])

    def forecast(self, returns: pd.Series = None, vix: pd.Series = None, horizon: int = 30) -> float:
        """
        Generate volatility forecast using most recent data.
        
        Args:
            returns: Historical returns (to create features from).
                    If None, uses cached returns from last fit() call.
            vix: VIX series. If None, uses cached VIX from last fit() call.
            horizon: Forecast horizon in days (default 30)
            
        Returns:
            Forecasted annualized volatility for next 'horizon' days (decimal, e.g., 0.15 for 15%)

        """
        if not self.is_fitted:
            raise ValueError("Must call fit() before forecast()")
        
        # Use provided returns or fall back to cached returns
        if returns is None:
            if self._last_returns is None:
                raise ValueError("No returns provided and no cached returns available")
            returns = self._last_returns
        
        # Use provided VIX or fall back to cached VIX
        if vix is None:
            if self._last_vix is None:
                raise ValueError("No VIX provided and no cached VIX available")
            vix = self._last_vix
        
        # Create features for ONLY the current timestamp (fast!)
        current_features = self._create_current_features(returns, vix, horizon=horizon)
        
        # Predict
        prediction = self.model.predict(current_features)[0]
        
        return float(prediction) / 100  # Convert Vol estimate to decimal
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importances from trained model.
        
        Returns:
            DataFrame with features and their importance scores
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

def asymmetric_squared_error(y_true, y_pred):
    """
    Custom XGBoost objective that penalizes underprediction more.
    Underpredicting vol → theo too low → sell signals → dangerous short positions.
    
    Args:
        y_true: Actual forward RV
        y_pred: Model's prediction
        
    Returns:
        gradient, hessian for XGBoost
    """
    # Loss function weights
    under_weight = 1.2
    over_weight = 1.0
    
    residual = y_pred - y_true  # Positive = overprediction, Negative = underprediction
    
    weights = np.where(residual < 0, under_weight, over_weight)
    
    grad = weights * residual
    hess = weights * np.ones_like(residual)
    
    return grad, hess

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