# Phase 3: Volatility Models - The Core of Your Project

## 🎯 Goal

Implement three competing volatility estimation methods. This is **THE** contribution of your project - comparing which approach to volatility modeling produces the best trading signals.

---

## 📊 The Three Strategies

### **Strategy B: Historical Realized Volatility**
**Hypothesis:** Market systematically misprices options relative to actual historical behavior

**File:** `src/volatility/historical.py`

**What to implement:**
1. `calculate_realized_volatility()` - Rolling historical vol
2. `RollingVolCalculator` class - For incremental updates

**Estimated time:** 3-4 hours

---

### **Strategy C: Volatility Surface Fitting**
**Hypothesis:** Illiquid strikes deviate from the "fair" surface implied by liquid options

**File:** `src/volatility/surface.py`

**What to implement:**
1. `VolatilitySurface` class with `.fit()` and `.get_vol()` methods
2. `fit_polynomial_surface()` - Polynomial fitting function

**Estimated time:** 4-6 hours

---

### **Strategy D: GARCH Forecasting**
**Hypothesis:** Statistical models predict vol better than naive methods or market IV

**File:** `src/volatility/forecasting.py`

**What to implement:**
1. `GARCHForecaster` class - GARCH(1,1) model
2. `EWMAForecaster` class - Simpler alternative

**Estimated time:** 5-8 hours (GARCH is complex!)

---

## 🚀 Suggested Order

### **Week 1: Strategy B (Historical Vol)**

**Why start here:**
- Easiest to implement
- Validates your data pipeline
- Provides baseline for comparison

**Steps:**
1. Implement `calculate_realized_volatility()` using close-to-close
2. Test on SPY price data (you already have this!)
3. Visualize: plot realized vol over time
4. Compare to Dolt's HV (from `spy_volatility.parquet`)

**Expected output:**
- 30-day realized vol time series
- Mean vol around 15-20% for SPY
- Vol spikes during market stress

---

### **Week 2: Strategy C (Vol Surface)**

**Why second:**
- Moderate complexity
- Teaches you about volatility smile/skew
- Core derivatives concept

**Steps:**
1. Implement polynomial surface fitting
2. For each date, fit surface to all available options
3. For each option, compare market IV to fitted IV
4. Visualize: 3D surface plot (optional but cool!)

**Expected output:**
- Fitted surface parameters for each week
- Residuals (market IV - fitted IV)
- Some options above surface, some below (trading signals!)

---

### **Week 3: Strategy D (GARCH)**

**Why last:**
- Most complex mathematically
- Requires understanding of MLE, time series
- Can use library (arch package) to save time

**Steps:**
1. Start with EWMA (easier warmup)
2. Implement or use GARCH library
3. Generate rolling forecasts
4. Compare forecasts to realized vol

**Expected output:**
- One-step-ahead vol forecasts
- GARCH parameters (ω, α, β)
- Forecast vs realized comparison

---

## 📝 Deliverables

### **For Each Strategy:**

1. **Implementation** - Working code in respective file
2. **Validation** - Test on SPY data
3. **Visualization** - Plot time series of vol estimates
4. **Comparison** - How does it compare to market IV?

### **Combined:**

5. **Comparison notebook** - Side-by-side comparison
6. **Signal generation** - Trading signals from each method

---

## 🧪 Testing Strategy

### **Strategy B Test:**
```python
# Load SPY prices
prices = pd.read_parquet('data/processed/spy_prices.parquet')['close']

# Calculate 30-day realized vol
rv = calculate_realized_volatility(prices, window=30)

# Should see:
# - ~20 valid vol estimates (after 30-day warmup)
# - Vol between 10% and 30% (typical for SPY)
# - Higher during Aug 2019 (tariff concerns)
```

### **Strategy C Test:**
```python
# Load options for one date
options = pd.read_parquet('data/processed/spy_options.parquet')
date_options = options[options['date'] == '2019-06-08']

# Fit surface
surface = VolatilitySurface()
surface.fit(date_options)

# Get fitted vol for ATM option
S = date_options['underlying_price'].iloc[0]
fitted_vol = surface.get_vol(S, S, 30)

# Compare to market
market_vol = date_options[date_options['strike'] == S]['vol'].iloc[0]
print(f"Market: {market_vol:.2%}, Fitted: {fitted_vol:.2%}")
```

### **Strategy D Test:**
```python
# Load returns
prices = pd.read_parquet('data/processed/spy_prices.parquet')['close']
returns = np.log(prices / prices.shift(1)).dropna()

# Fit GARCH
garch = GARCHForecaster()
garch.fit(returns)

# Forecast
forecast = garch.forecast(horizon=7)  # 7-day ahead
print(f"GARCH forecast: {forecast:.2%}")
```

---

## 💡 Key Concepts to Understand

### **Strategy B: Historical Vol**

**Math:**
```
Return: r_t = ln(P_t / P_t-1)
Variance: σ² = (1/n)Σ(r_t - μ)²
Volatility: σ = sqrt(σ² * 252)  # Annualize
```

**Pros:**
- Simple, transparent
- Based on actual realized outcomes
- Establishes baseline

**Cons:**
- Backward-looking (uses past data)
- Assumes vol is constant over window
- Can be slow to react to regime changes

---

### **Strategy C: Vol Surface**

**Math:**
```
Model: IV(m,T) = β₀ + β₁*m + β₂*m² + β₃*T + β₄*m*T + ...
Where: m = ln(K/S), T = days/365
Fit: minimize Σ[IV_market - IV_model]²
```

**Pros:**
- Captures volatility smile/skew
- Finds relative value across strikes
- Smooths out noise in illiquid options

**Cons:**
- Assumes smooth relationship (may not hold during stress)
- Requires sufficient options across strikes
- Model risk (wrong functional form)

---

### **Strategy D: GARCH**

**Math:**
```
GARCH(1,1): σ²_t = ω + α*r²_t-1 + β*σ²_t-1
Forecast: E[σ²_t+h] = long_run_var + (α+β)^h * (σ²_t - long_run_var)
```

**Pros:**
- Forward-looking (generates forecasts)
- Captures vol clustering
- Mean-reversion to long-run average

**Cons:**
- Complex to estimate (MLE, constraints)
- Can be unstable with limited data
- Assumes specific functional form

---

## 📊 What Success Looks Like

After Phase 3, you should have:

1. **Three vol time series** for SPY (June - Dec 2019):
   - Historical 30-day realized vol
   - Fitted surface vol (for each option)
   - GARCH forecasted vol

2. **Three theo price series** for each option:
   - Theo using historical vol
   - Theo using surface vol
   - Theo using GARCH vol

3. **Three signal series**:
   - Buy/sell signals from Strategy B
   - Buy/sell signals from Strategy C
   - Buy/sell signals from Strategy D

---

## 🎓 Interview Talking Points

**Q: "Why compare three different vol models?"**

**Your Answer:**
> "I wanted to understand whether options mispricings come from systematic vol risk premium, relative value across the surface, or forecasting errors. Strategy B tests if historical vol beats implied vol. Strategy C tests if surface smoothing finds arbitrage in illiquid strikes. Strategy D tests if statistical forecasts beat the market. By comparing their trading performance, I can identify which source of edge is most reliable for options strategies."

---

## ⏱️ Timeline

- **Week 1**: Strategy B (historical) - 3-4 hours
- **Week 2**: Strategy C (surface) - 4-6 hours
- **Week 3**: Strategy D (GARCH) - 5-8 hours
- **Total**: 12-18 hours over 3 weeks

**You're on track for end of January!**

---

## 🚀 Getting Started

### **Start with Strategy B:**

1. Open `src/volatility/historical.py`
2. Implement `calculate_realized_volatility()`
3. Test with: `python3 src/volatility/historical.py`
4. Visualize in notebook

### **Then Strategy C:**

1. Open `src/volatility/surface.py`
2. Implement `VolatilitySurface.fit()` and `.get_vol()`
3. Test on one date's options
4. Visualize the surface (optional but impressive!)

### **Finally Strategy D:**

1. Install arch package: `pip install arch`
2. Open `src/volatility/forecasting.py`
3. Implement EWMA first (easier)
4. Then GARCH (use arch library recommended)
5. Generate rolling forecasts

---

## 📚 Resources

### **For Strategy B (Historical Vol):**
- Simple calculation, numpy/pandas sufficient
- Reference: any quant finance textbook

### **For Strategy C (Vol Surface):**
- Polynomial fitting: `numpy.polyfit` or `scipy.optimize`
- Paper: "Arbitrage-free SVI volatility surfaces" (Gatheral)
- Simpler: Just use polynomial for your project

### **For Strategy D (GARCH):**
- **Recommended**: Use `arch` library (saves time)
  - Install: `pip install arch`
  - Docs: https://arch.readthedocs.io/
- Alternative: Implement from scratch (learning experience)
- Reference: "Time Series Analysis" by Hamilton

---

**Ready to start?** Focus on Strategy B first - get that working, then build up to C and D!

Good luck! 🎯

