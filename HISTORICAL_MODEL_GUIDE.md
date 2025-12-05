# Historical Volatility Model - Implementation Guide

## Key Concept: Historical Model is Different

Unlike GARCH, ML, and VolatilitySurface models that get "fitted" once, the **Historical Volatility Model** is **stateful** and gets **updated incrementally** as the backtest progresses.

## How It Works

### 1. Initialization (in `setup_historical_model`)
```python
# Create an empty RollingVolCalculator
rolling_vol = RollingVolCalculator(window=30, annualize=True)
return rolling_vol
```

**Important**: We return the calculator object itself, not a volatility value!

### 2. During Backtest (walk-forward updates)

The calculator maintains a rolling window of returns and updates with each new price:

```
Day 1:  [price1] -> Not enough data yet (returns < 30)
Day 2:  [price1, price2] -> Calculate 1 return, still not enough
...
Day 30: [price1...price30] -> Calculate 29 returns -> vol estimate ready!
Day 31: [price2...price31] -> Roll forward, drop oldest price
```

### 3. How the Strategy Uses It

#### A. `update_vol_model(current_date)`
Called every trading day to update the calculator:

```python
if isinstance(self.vol_model, RollingVolCalculator):
    # Get two most recent prices
    prev_price = historical_prices.iloc[-2]['close']
    new_price = historical_prices.iloc[-1]['close']
    
    # Update calculator (adds new return to rolling window)
    self.vol_model.update(new_price, prev_price)
```

#### B. `calculate_theo_price(S, K, T, r, option_type)`
Gets current volatility estimate:

```python
elif isinstance(self.vol_model, RollingVolCalculator):
    sigma = self.vol_model.get_current_vol()  # Get current vol estimate
    if sigma is None:
        sigma = 0.2  # Default if not enough data yet
```

#### C. `generate_signals()`
Uses the same logic to get vol for all options:

```python
elif isinstance(self.vol_model, RollingVolCalculator):
    current_vol = self.vol_model.get_current_vol()
    sigmas = np.full(len(df), current_vol)  # Broadcast to all options
```

## What I Fixed

### Issue 1: `setup_historical_model()` in `main.py`
**Before:**
```python
rolling_vol = RollingVolCalculator(window=30, annualize=np.True)
return realized_vol.iloc[-1]  # ❌ realized_vol doesn't exist!
```

**After:**
```python
rolling_vol = RollingVolCalculator(window=window, annualize=True)
return rolling_vol  # ✅ Return the calculator object
```

### Issue 2: `calculate_theo_price()` in `strategy.py`
**Before:**
```python
else:
    sigma = self.vol_model  # ❌ Treats it as a float!
```

**After:**
```python
elif isinstance(self.vol_model, RollingVolCalculator):
    sigma = self.vol_model.get_current_vol()  # ✅ Call method
    if sigma is None:
        sigma = 0.2
```

### Issue 3: `update_vol_model()` in `strategy.py`
**Before:**
```python
# Only handled VolatilitySurface, ignored historical model
```

**After:**
```python
if isinstance(self.vol_model, RollingVolCalculator):
    # Get two most recent prices
    prev_price = float(historical_prices.iloc[-2]['close'])
    new_price = float(historical_prices.iloc[-1]['close'])
    # Update the calculator
    self.vol_model.update(new_price, prev_price)
```

### Issue 4: `generate_signals()` in `strategy.py`
**Before:**
```python
else:
    sigmas = np.full(len(df), self.vol_model)  # ❌ Won't work for objects
```

**After:**
```python
elif isinstance(self.vol_model, RollingVolCalculator):
    current_vol = self.vol_model.get_current_vol()
    sigmas = np.full(len(df), current_vol)  # ✅ Broadcast vol value
```

### Issue 5: Strategy needs prices data
**Added to `MispricingStrategy.__init__()`:**
```python
prices_data: Optional[pd.DataFrame] = None  # New parameter
```

**Updated `run_single_backtest()` in `main.py`:**
```python
# Historical model now gets prices_data
results_df, engine = run_single_backtest(
    'historical', hist_model, 
    returns_data=None, 
    options_data=None, 
    prices_data=prices  # ✅ Pass prices for calculator updates
)
```

## Comparison with Other Models

| Model Type | Initialization | Updates During Backtest | How Strategy Uses It |
|------------|----------------|-------------------------|---------------------|
| **Historical** | Create empty calculator | `update(new_price, prev_price)` each day | `get_current_vol()` |
| **VolatilitySurface** | Create empty surface | `fit(options_data)` weekly | `get_vol(K, S, dte)` |
| **GARCH** | Fit on initial data | Optional: refit with expanding window | `forecast(horizon)` |
| **ML** | Train on initial data | Optional: retrain with expanding window | `forecast(returns)` |

## Testing Your Implementation

Run a simple test:
```bash
cd /home/oalkhadra/options-trading
python -m src.backtest.main
```

You should see:
- ✅ Historical model backtest completes
- ✅ Vol surface backtest completes
- ✅ Equity curves generated
- ✅ IV comparison plot shows both models

## Key Takeaways

1. **Historical model is stateful**: It maintains a rolling window of returns
2. **Walk-forward updates**: Updated incrementally, not refitted from scratch
3. **No "training"**: Just needs enough data points (window size) to calculate std dev
4. **Most realistic**: Uses only past data, no lookahead bias
5. **Simplest baseline**: Good benchmark for comparing more sophisticated models

## Interview Talking Points

When discussing this:
- "Historical vol is a naive baseline that assumes vol is constant over the rolling window"
- "It's walk-forward and doesn't require any training/fitting"
- "We update it incrementally with each new price, maintaining a 30-day rolling window"
- "Unlike GARCH or ML, it has no predictive power - just uses recent realized vol"
- "If historical vol beats GARCH/ML, it suggests either (a) vol is persistent enough that recent past predicts near future, or (b) the complex models are overfitting"

