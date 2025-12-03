# Strategy D: GARCH vs ML Forecasting

## 🎯 Updated Strategy D Structure

**Classical Econometrics vs Modern Machine Learning**

```
Strategy D: Statistical Forecasting
  ├── D1: GARCH(1,1) - Parametric time series model
  └── D2: XGBoost ML - Non-parametric machine learning
```

This comparison tests: **Can ML beat traditional econometrics at vol forecasting?**

---

## 📊 **Why This Comparison is Powerful**

### **GARCH (Classical)**
- ✅ Industry standard since 1980s
- ✅ Interpretable parameters (ω, α, β)
- ✅ Theoretical foundation
- ✅ Used by every bank/hedge fund
- ⚠️ Assumes specific functional form
- ⚠️ May miss non-linear patterns

### **XGBoost ML (Modern)**
- ✅ Can capture non-linear relationships
- ✅ Learns feature interactions
- ✅ No parametric assumptions
- ✅ Shows modern data science skills
- ⚠️ Black box (less interpretable)
- ⚠️ Needs more data than GARCH
- ⚠️ Risk of overfitting

---

## 🤖 **ML Feature Engineering (Most Important!)**

### **Good Features for Vol Prediction:**

```python
def create_features(returns):
    """
    Feature engineering is KEY to ML success!
    """
    features = {}
    
    # 1. Lagged Realized Vols (Multiple Windows)
    features['rv_5d'] = realized_vol(returns, window=5)
    features['rv_10d'] = realized_vol(returns, window=10)
    features['rv_20d'] = realized_vol(returns, window=20)
    features['rv_30d'] = realized_vol(returns, window=30)
    features['rv_60d'] = realized_vol(returns, window=60)
    
    # 2. Recent Returns (Momentum)
    features['ret_1d'] = returns.iloc[-1]
    features['ret_5d'] = returns.iloc[-5:].sum()
    features['ret_20d'] = returns.iloc[-20:].sum()
    
    # 3. Squared Returns (Direct vol proxy)
    features['ret_sq_1d'] = returns.iloc[-1]**2
    features['ret_sq_5d_avg'] = (returns.iloc[-5:]**2).mean()
    
    # 4. Higher Moments
    features['skew_30d'] = returns.iloc[-30:].skew()
    features['kurt_30d'] = returns.iloc[-30:].kurtosis()
    
    # 5. Vol of Vol (Second order)
    rv_series = realized_vol(returns, window=20)
    features['vol_of_vol'] = rv_series.iloc[-20:].std()
    
    # 6. Trend Features
    features['rv_trend'] = features['rv_5d'] - features['rv_30d']
    
    # 7. GARCH Forecast (Use as feature!)
    garch_temp = GARCHForecaster()
    garch_temp.fit(returns)
    features['garch_forecast'] = garch_temp.forecast()
    
    return features
```

**Why these features?**
- **Lagged vols:** Capture different time scales (HAR-like)
- **Recent returns:** Shocks that might increase vol
- **Squared returns:** Direct vol proxy
- **Higher moments:** Fat tails = high vol
- **Vol of vol:** Changing volatility regime
- **GARCH feature:** Incorporate classical model knowledge!

---

## 🎯 **Target Variable**

**What are you predicting?**

```python
# At time t, predict realized vol from t to t+7

# Features: Use data up to time t
X_t = create_features(returns[:t])

# Target: Actual realized vol over next 7 days
y_t = calculate_realized_volatility(returns[t:t+7], window=7).iloc[-1]
```

---

## 🏗️ **Implementation Structure**

### **GARCH (Use arch Library):**

```python
class GARCHForecaster:
    def fit(self, returns):
        from arch import arch_model
        
        # Fit GARCH(1,1)
        model = arch_model(returns*100, vol='Garch', p=1, q=1)  # Scale returns
        result = model.fit(disp='off', show_warning=False)
        
        # Extract parameters
        self.omega = result.params['omega']
        self.alpha = result.params['alpha[1]']
        self.beta = result.params['beta[1]']
        
        # Get current conditional variance
        self.current_variance = result.conditional_volatility.iloc[-1]**2 / (252*100)
        
        self.is_fitted = True
    
    def forecast(self, horizon=7):
        # Long-run variance
        lr_var = self.omega / (1 - self.alpha - self.beta)
        
        # Multi-step forecast
        forecast_var = lr_var + (self.alpha + self.beta)**horizon * (self.current_variance - lr_var)
        
        # Annualize
        return np.sqrt(forecast_var * 252)
```

---

### **ML (XGBoost):**

```python
class MLForecaster:
    def _create_features(self, returns, lookback=60):
        """Create lagged features."""
        n = len(returns)
        features_list = []
        
        for i in range(lookback, n):
            # Get past returns up to (but not including) time i
            past_returns = returns.iloc[i-lookback:i]
            
            # Calculate features
            feat = {
                'rv_5d': realized_vol(past_returns.iloc[-5:], window=5).iloc[-1],
                'rv_10d': realized_vol(past_returns.iloc[-10:], window=10).iloc[-1],
                'rv_20d': realized_vol(past_returns.iloc[-20:], window=20).iloc[-1],
                'rv_30d': realized_vol(past_returns.iloc[-30:], window=30).iloc[-1],
                'ret_1d': past_returns.iloc[-1],
                'ret_5d': past_returns.iloc[-5:].sum(),
                'skew_30d': past_returns.iloc[-30:].skew(),
                'kurt_30d': past_returns.iloc[-30:].kurtosis(),
            }
            
            features_list.append(feat)
        
        return pd.DataFrame(features_list)
    
    def _create_targets(self, returns, horizon=7):
        """Create forward-looking targets."""
        n = len(returns)
        targets = []
        
        for i in range(lookback, n - horizon):
            # Future returns (from i to i+horizon)
            future_returns = returns.iloc[i:i+horizon]
            
            # Calculate realized vol over horizon
            future_vol = future_returns.std() * np.sqrt(252)
            targets.append(future_vol)
        
        return np.array(targets)
    
    def fit(self, returns):
        X = self._create_features(returns)
        y = self._create_targets(returns)
        
        # Align X and y (targets are shorter)
        X = X.iloc[:len(y)]
        
        # Fit XGBoost
        self.model.fit(X, y)
        self.feature_names = X.columns.tolist()
        self.is_fitted = True
    
    def forecast(self, returns):
        # Create features from most recent data
        X_current = self._create_features(returns).iloc[[-1]]
        
        # Predict
        forecast = self.model.predict(X_current)[0]
        
        return forecast
```

---

## ⚠️ **Critical: Avoid Data Leakage!**

### **Expanding Window in Backtesting:**

```python
# For each trading week:
all_dates = sorted(options['date'].unique())

for i, current_date in enumerate(all_dates):
    # Get all returns BEFORE current date
    past_returns = returns[returns.index < current_date]
    
    # Strategy D1: GARCH
    garch = GARCHForecaster()
    garch.fit(past_returns)  # Only uses past!
    garch_vol = garch.forecast(horizon=7)
    
    # Strategy D2: ML
    ml = MLForecaster(horizon=7)
    if len(past_returns) >= 60:  # Minimum for ML
        ml.fit(past_returns)  # Only uses past!
        ml_vol = ml.forecast(past_returns)
    else:
        ml_vol = None  # Skip first few weeks (insufficient data)
    
    # Use forecasts to generate theo prices and signals
    # ...
```

**This ensures:** No future data used ✓

---

## 🎓 **Interview Talking Points**

**Q: "Why compare GARCH with ML?"**

**Your Answer:**
> "I wanted to test whether machine learning could beat the industry-standard GARCH model at volatility forecasting. GARCH has strong theoretical foundations and captures vol clustering and mean reversion through its parametric form. However, it assumes a specific structure. XGBoost can learn non-linear relationships and feature interactions from data without parametric assumptions. I engineered features including lagged vols at multiple horizons, recent returns, and higher moments. Both models used expanding window training with strict temporal validation. The comparison tests whether flexible ML can extract patterns that structured econometric models miss."

**Follow-up Q: "What did you find?"**

**Be ready to discuss actual results:**
- Which performed better?
- In what market regimes?
- Feature importance from ML
- GARCH parameter interpretation

---

## 📊 **Expected Results (Hypotheses)**

### **GARCH Might Win If:**
- Markets are efficient
- Vol follows clean mean-reverting process
- Limited data (GARCH needs less)
- Parameters are stable

### **ML Might Win If:**
- Non-linear patterns exist
- Feature interactions matter
- Regime changes occur
- More data available

**You won't know until you test!** That's the research question.

---

## ⏱️ **Implementation Time:**

**GARCH:** 2-3 hours
- Use arch library (don't implement from scratch)
- Test on SPY returns
- Validate parameters make sense

**ML:** 4-5 hours
- Feature engineering: 2 hours (most important!)
- Model training: 1 hour
- Validation: 1 hour
- Feature importance analysis: 1 hour

**Total Strategy D:** 6-8 hours

---

## 🚀 **Next Steps:**

1. **Install dependencies:**
```bash
pip install arch xgboost scikit-learn
```

2. **Implement GARCH first** (easier, foundation)
3. **Then implement ML** (can use GARCH as feature!)
4. **Test both** with `python3 src/volatility/forecasting.py`

---

## ✅ **Phase 3 Summary When Done:**

```
Strategy B: Historical 30-day realized vol ✓
Strategy C: Polynomial vol surface ✓
Strategy D:
  - D1: GARCH(1,1) forecast ✓
  - D2: XGBoost ML forecast ✓
```

**Four different vol estimation methods** = Extremely strong project!

---

**Ready to implement GARCH and ML?** Start with GARCH (using arch library to save time), then tackle the ML feature engineering! 🚀

