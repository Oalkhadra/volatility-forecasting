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
            returns: pandas Series of returns
            
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

    """
    
    def __init__(self, pretrained: bool = False):
        """
        Initialize ML forecaster.
        
        Args:
            pretrained: If True, model expects to be pretrained before backtest.
                       This skips daily retraining in update_vol_model().
        """
        self.model = xgb.XGBRegressor( # Parameters have been fine-tuned through OOS train/test data in notebooks/xgboost_development.ipynb
            n_estimators=300,
            max_depth=3,
            learning_rate=0.01,
            random_state=115,
            objective=asymmetric_squared_error
        )
        
        self.is_fitted = False
        self.pretrained = pretrained  # Skip daily retraining if True for efficient backtesting
        self.feature_names = []
        self.scaler = None  # For feature scaling if needed
        self._last_returns = None  # Store returns for forecasting
        self._last_vix = None  # Store VIX for forecasting
    
    def _create_features(self, returns: pd.Series, vix: pd.Series, lookback: int = 30, horizon: int = 30) -> pd.DataFrame:
        returns = returns.reset_index(drop=True)
        vix = vix.reset_index(drop=True)

        # Lagged VIX
        yesterday_vix = vix.shift(1)

        # Historical realized volatility
        rv_1d = (np.abs(returns) * np.sqrt(252)).shift(1)            # Yesterday's |return|
        rv_5d = (returns.rolling(5).std() * np.sqrt(252)).shift(1)   # Past 5 days (t-5 to t-1)
        rv_22d = (returns.rolling(22).std() * np.sqrt(252)).shift(1) # Past 22 days (t-22 to t-1)
        rv_30d = (returns.rolling(30).std() * np.sqrt(252)).shift(1) # Past 30 days (t-30 to t-1)
        rv_60d = (returns.rolling(60).std() * np.sqrt(252)).shift(1) # Past 60 days (t-60 to t-1)
        rv_120d = (returns.rolling(120).std() * np.sqrt(252)).shift(1) # Past 120 days (t-120 to t-1)


        # Historical returns
        ret_1d = returns.shift(1)                       # Yesterday's return
        ret_22d = returns.rolling(22).sum().shift(1)    # Cumulative return t-22 to t-1
        ret_30d = returns.rolling(30).sum().shift(1)    # Cumulative return t-30 to t-1
        ret_60d = returns.rolling(60).sum().shift(1)    # Cumulative return t-60 to t-1
        ret_120d = returns.rolling(120).sum().shift(1)  # Cumulative return t-120 to t-1

        neg_returns = returns.clip(upper=0)
        rsv_neg_5d = (np.sqrt((neg_returns**2).rolling(5).sum()) * np.sqrt(252)).shift(1)

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

    def _create_targets(self, returns: pd.Series, lookback: int = 30, horizon: int = 30) -> np.ndarray:
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
        targets = rolling_vol.shift(-horizon)
        
        # 3. Slice to match the feature generation
        target_slice = targets.iloc[lookback : -horizon]
        
        return target_slice.values
    
    def create_feature_targets(self, returns: pd.Series, vix: pd.Series, lookback: int = 30, horizon: int = 30):
        # 1. Create Features and Targets for each horizon
        all_X, all_y = [], []
        
        X = self._create_features(returns, vix, lookback=lookback, horizon=horizon)
        y = self._create_targets(returns, lookback=lookback, horizon=horizon)
        
        # Align X and y: targets are shorter because they need forward data
        n_samples = len(y)
        if n_samples > 0:
            X = X.iloc[:n_samples]
            all_X.append(X)
            all_y.append(y)
        
        if len(all_X) == 0:
            raise ValueError("Not enough data to create training samples")
        
        X_combined = pd.concat(all_X, ignore_index=True)
        y_combined = np.concatenate(all_y)
        
        mask = ~(X_combined.isna().any(axis=1) | np.isnan(y_combined))
        X_combined = X_combined[mask]
        y_combined = y_combined[mask]

        return X_combined, y_combined

    def fit(self, returns: pd.Series, vix: pd.Series, lookback: int = 30, min_train_size = 30, horizon: int = 30) -> None:
        """
        Fit ML model to return series.

        Args:
            returns: Historical returns
            vix: Historical VIX
            min_train_size: Minimum returns needed to fit

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
        
        X_combined, y_combined = self.create_feature_targets(returns, vix, lookback=lookback, horizon=horizon)
        
        if len(X_combined) == 0:
            raise ValueError("Not enough data to create training samples")
        
        self.model.fit(X_combined, y_combined)
        self.is_fitted = True
        
        # Save feature names for later importance plotting
        self.feature_names = X_combined.columns.tolist()
        
        print(f"  MLForecaster trained on {len(X_combined)} samples")
    
    def pretrain(self, prices_df: pd.DataFrame, vix_df: pd.Series, cutoff_date, lookback: int = 30, horizon: int = 30) -> None:
        """
        Pre-train the model on historical data before a cutoff date.
        
        Avoiding look-ahead bias by only using data before the backtest, and training on a large pre-backtest dataset.
        After pretraining, daily calls to fit() will only update the returns cache for forecasting, not retrain the model.
        
        Args:
            prices_df: DataFrame with 'date' and 'log_ret' columns
            cutoff_date: Train only on data before this date (typically backtest start)
            horizon: Forecast horizon
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
        self.is_fitted = False
        
        # Train the model
        self.fit(returns, vix, lookback=lookback, horizon=horizon)
        
        # Now mark as pretrained
        self.pretrained = True
        print(f"  Pre-training complete!")

    def _create_current_features(self, returns: pd.Series, vix: pd.Series, horizon: int = 30) -> pd.DataFrame:
        """
        Create features for ONLY the most recent timestamp (optimized for forecasting).
        
        This matches the exact feature format from _create_features() but only
        computes for the current time point.
        
        Args:
            returns: Return series (needs at least 120 values for all features)
            vix: VIX series
            horizon: Forecast horizon
            
        Returns:
            Single-row DataFrame with features for the current time point
        """
        if len(returns) < 120:
            raise ValueError("Need at least 120 returns to create features")
        
        # Lagged VIX
        yesterday_vix = vix.iloc[-1]
        
        # Historical realized volatility
        rv_1d = np.abs(returns.iloc[-1]) * np.sqrt(252)  # Yesterday's |return|
        rv_5d = returns.iloc[-5:].std() * np.sqrt(252)   # Past 5 days
        rv_22d = returns.iloc[-22:].std() * np.sqrt(252) # Past 22 days
        rv_30d = returns.iloc[-30:].std() * np.sqrt(252) # Past 30 days
        rv_60d = returns.iloc[-60:].std() * np.sqrt(252) # Past 60 days
        rv_120d = returns.iloc[-120:].std() * np.sqrt(252) # Past 120 days
        
        # Historical returns
        ret_1d = returns.iloc[-1]  # Yesterday's return
        ret_22d = returns.iloc[-22:].sum()  # Cumulative return last 22 days
        ret_30d = returns.iloc[-30:].sum()  # Cumulative return last 30 days
        ret_60d = returns.iloc[-60:].sum()  # Cumulative return last 60 days
        ret_120d = returns.iloc[-120:].sum()  # Cumulative return last 120 days
        
        neg_returns = returns.iloc[-5:].clip(upper=0)
        rsv_neg_5d = np.sqrt((neg_returns**2).sum()) * np.sqrt(252)
        
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
            Forecasted annualized volatility 

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

        importance = self.model.feature_importances_
        
        return pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)


# Helper function (Custom Loss Function)
def asymmetric_squared_error(y_true, y_pred):
    """
    Custom XGBoost objective that penalizes underprediction slightly more.
    
    Args:
        y_true: Actual forward RV
        y_pred: Model's prediction
        
    Returns:
        gradient, hessian for XGBoost
    """
    # Loss function weights
    under_weight = 1.2
    over_weight = 1.0
    
    residual = y_pred - y_true
    weights = np.where(residual < 0, under_weight, over_weight)
    
    grad = weights * residual
    hess = weights * np.ones_like(residual)
    
    return grad, hess