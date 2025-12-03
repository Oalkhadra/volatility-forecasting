# 🚀 Phase 3 Started: Volatility Models

## 🎉 Phase 2 Complete - All Tests Passing!

You've successfully built a complete options pricing system:
- ✅ Black-Scholes pricer (15/15 tests)
- ✅ All Greeks (30/30 tests)
- ✅ Implied volatility solver
- ✅ Validated on real SPY data

**Total: 45/45 tests passing** 🎯

---

## 📦 New Files Created for Phase 3

### **Module Structure:**
```
src/volatility/
  ├── __init__.py          # Package exports
  ├── historical.py        # Strategy B: Historical realized vol
  ├── surface.py           # Strategy C: Vol surface fitting  
  └── forecasting.py       # Strategy D: GARCH forecasting
```

### **Documentation:**
- `PHASE_3_ASSIGNMENT.md` - Full assignment guide
- `PHASE_3_FORMULAS.md` - Formula reference
- `PHASE_2_COMPLETE.md` - What you just finished
- `PROJECT_STATUS.md` - Overall project tracker

---

## 🎯 Your Phase 3 Assignment

### **Strategy B: Historical Vol** (Start Here!)
**File:** `src/volatility/historical.py`

**Functions to implement:**
1. `calculate_realized_volatility()` - Main function
2. `RollingVolCalculator` class - For incremental updates

**Test:**
```bash
python3 src/volatility/historical.py
```

**Estimated time:** 3-4 hours

---

### **Strategy C: Vol Surface** (Second)
**File:** `src/volatility/surface.py`

**Classes to implement:**
1. `VolatilitySurface` class:
   - `.fit()` method
   - `.get_vol()` method
2. `fit_polynomial_surface()` helper

**Test:**
```bash
python3 src/volatility/surface.py
```

**Estimated time:** 4-6 hours

---

### **Strategy D: GARCH** (Third)
**File:** `src/volatility/forecasting.py`

**Classes to implement:**
1. `GARCHForecaster` class (can use arch library!)
2. `EWMAForecaster` class (simpler warmup)

**Test:**
```bash
python3 src/volatility/forecasting.py
```

**Estimated time:** 5-8 hours

---

## 📚 Resources Provided

### **In Each File:**
- Comprehensive docstrings
- TODO comments with hints
- Example code snippets
- Built-in test functions
- Interview questions

### **In Formula Reference:**
- All equations you need
- Python implementation examples
- Expected behaviors
- Validation criteria

---

## 🧪 How to Test Each Strategy

### **Strategy B:**
```python
import pandas as pd
from src.volatility.historical import calculate_realized_volatility

# Load SPY prices
prices = pd.read_parquet('data/processed/spy_prices.parquet')['close']

# Calculate 30-day realized vol
rv = calculate_realized_volatility(prices, window=30)

# Check output
print(f"Mean vol: {rv.mean():.2%}")  # Should be 15-25%
print(f"Vol range: {rv.min():.2%} to {rv.max():.2%}")
```

### **Strategy C:**
```python
from src.volatility.surface import VolatilitySurface

# Load options
options = pd.read_parquet('data/processed/spy_options.parquet')
test_data = options[options['date'] == '2019-06-08']

# Fit surface
surface = VolatilitySurface()
surface.fit(test_data)

# Get fitted vol
vol = surface.get_vol(strike=290, underlying_price=287.65, days_to_expiry=30)
print(f"Fitted vol: {vol:.2%}")
```

### **Strategy D:**
```python
from src.volatility.forecasting import GARCHForecaster
import numpy as np

# Calculate returns
prices = pd.read_parquet('data/processed/spy_prices.parquet')['close']
returns = np.log(prices / prices.shift(1)).dropna()

# Fit GARCH
garch = GARCHForecaster()
garch.fit(returns)

# Forecast
forecast = garch.forecast(horizon=7)
print(f"7-day vol forecast: {forecast:.2%}")
```

---

## 🎓 What You'll Learn in Phase 3

### **Strategy B:**
- Time series analysis
- Rolling window calculations
- Pandas operations
- Annualization conventions

### **Strategy C:**
- Polynomial fitting
- Least squares regression
- Volatility smile/skew
- Cross-sectional analysis

### **Strategy D:**
- GARCH modeling (industry standard!)
- Maximum likelihood estimation
- Time series forecasting
- Model persistence and mean reversion

---

## 💼 Interview Prep

By completing Phase 3, you'll be able to discuss:

**"How would you model volatility?"**
> "I implemented three approaches in my project: historical realized vol using rolling windows, polynomial surface fitting across strikes and maturities, and GARCH(1,1) for statistical forecasting. Each has tradeoffs - historical is simple but backward-looking, surface captures cross-sectional patterns, and GARCH provides forward-looking forecasts with mean reversion. I backtested all three to compare which produces the best risk-adjusted returns."

---

## 📁 File Organization

```
options-trading/
├── src/
│   ├── preprocess.py        ✓ Complete
│   ├── risk_free_rate.py    ✓ Complete
│   ├── pricer.py            ✓ Complete (45 tests passing)
│   ├── greeks.py            ✓ Complete (30 tests passing)
│   └── volatility/
│       ├── __init__.py      📝 TODO
│       ├── historical.py    📝 TODO (Strategy B)
│       ├── surface.py       📝 TODO (Strategy C)
│       └── forecasting.py   📝 TODO (Strategy D)
├── tests/
│   ├── test_pricer.py       ✓ Complete
│   └── test_greeks.py       ✓ Complete
├── data/
│   └── processed/           ✓ Ready
└── notebooks/
    └── exploration.ipynb    ✓ Active

```

---

## ⏱️ Timeline

**Completed (4 weeks):**
- Week 1-2: Data pipeline
- Week 3-4: Pricing & Greeks

**Remaining (9 weeks to end of January):**
- Week 5-7: Vol models (← You are here)
- Week 8-10: Backtesting
- Week 11-12: Analysis
- Week 13: Documentation

**Status:** ✅ ON SCHEDULE

---

## 🚀 Next Steps

### **This Week:**
1. Read `PHASE_3_ASSIGNMENT.md`
2. Review `PHASE_3_FORMULAS.md`
3. Implement Strategy B (historical vol)
4. Test on SPY data

### **Getting Started:**
```bash
# Read the assignment
cat PHASE_3_ASSIGNMENT.md

# Open the first file
code src/volatility/historical.py

# Start with calculate_realized_volatility()
```

---

## 💡 Tips for Success

1. **Start simple** - Get Strategy B working before moving on
2. **Test frequently** - Run the built-in tests after each function
3. **Visualize** - Plot your vol estimates to sanity check
4. **Compare** - Check against Dolt's HV to validate
5. **Ask questions** - GARCH is complex, don't hesitate to ask!

---

## 🎯 Success Criteria for Phase 3

You're done when you can:

1. ✅ Calculate 30-day realized vol from SPY prices
2. ✅ Fit vol surface to one week of options
3. ✅ Generate GARCH vol forecast from returns
4. ✅ All three produce reasonable vol estimates (10-40% range)
5. ✅ Can use these vols to generate theo prices

---

## 📊 What Phase 4 Will Look Like

Once Phase 3 is done, you'll have three vol estimates. In Phase 4, you'll:

```python
# For each option, each week:

# Strategy B
historical_vol = get_historical_vol(date, window=30)
theo_B = black_scholes_price(S, K, T, r, historical_vol, type)
signal_B = "BUY" if theo_B > market_price else "SELL"

# Strategy C  
surface_vol = surface.get_vol(K, S, DTE)
theo_C = black_scholes_price(S, K, T, r, surface_vol, type)
signal_C = "BUY" if theo_C > market_price else "SELL"

# Strategy D
garch_vol = garch.forecast()
theo_D = black_scholes_price(S, K, T, r, garch_vol, type)
signal_D = "BUY" if theo_D > market_price else "SELL"

# Execute trades, delta hedge, track PnL
```

**But first, build the vol models!**

---

**Ready to dive into Phase 3?** 🚀

Start with Strategy B - it's the most straightforward and gets you comfortable with the workflow!

