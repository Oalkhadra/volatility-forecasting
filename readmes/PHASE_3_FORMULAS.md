# Phase 3: Volatility Formulas Quick Reference

## 📐 Strategy B: Historical Realized Volatility

### **Close-to-Close Volatility**

**Log Returns:**
```
r_t = ln(P_t / P_t-1)
```

**Sample Variance:**
```
σ² = (1/(n-1)) * Σ(r_t - μ)²
```

**Annualized Volatility:**
```
σ_annual = σ_daily * sqrt(252)
```

**Python Implementation:**
```python
import numpy as np
import pandas as pd

def calculate_realized_volatility(prices, window=30):
    # Calculate log returns
    returns = np.log(prices / prices.shift(1))
    
    # Rolling standard deviation
    rolling_std = returns.rolling(window=window).std()
    
    # Annualize (252 trading days)
    annualized_vol = rolling_std * np.sqrt(252)
    
    return annualized_vol
```

---

## 📐 Strategy C: Volatility Surface (Polynomial)

### **Log-Moneyness:**
```
m = ln(K/S)

m < 0: ITM call, OTM put
m = 0: ATM
m > 0: OTM call, ITM put
```

### **Time to Maturity:**
```
T = days_to_expiry / 365
```

### **Polynomial Model (Degree 2):**
```
IV(m,T) = β₀ + β₁*m + β₂*m² + β₃*T + β₄*m*T
```

**Python Implementation:**
```python
import numpy as np

def fit_polynomial_surface(options_df):
    # Calculate features
    m = np.log(options_df['strike'] / options_df['underlying_price'])
    T = options_df['days_to_expiry'] / 365
    
    # Feature matrix [1, m, m², T, m*T]
    X = np.column_stack([
        np.ones(len(m)),  # Intercept
        m,                # Linear moneyness
        m**2,             # Quadratic moneyness
        T,                # Linear time
        m*T               # Interaction
    ])
    
    # Target: market IVs
    y = options_df['vol'].values
    
    # Fit using least squares
    coeffs, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
    
    return coeffs

def get_fitted_vol(strike, spot, dte, coeffs):
    m = np.log(strike / spot)
    T = dte / 365
    
    # Evaluate polynomial
    fitted_iv = (
        coeffs[0] +           # β₀
        coeffs[1]*m +         # β₁*m
        coeffs[2]*m**2 +      # β₂*m²
        coeffs[3]*T +         # β₃*T
        coeffs[4]*m*T         # β₄*m*T
    )
    
    # Bound to reasonable range
    return np.clip(fitted_iv, 0.01, 5.0)
```

---

## 📐 Strategy D: GARCH(1,1)

### **GARCH Model:**
```
σ²_t = ω + α*r²_t-1 + β*σ²_t-1

Where:
  ω = long-run variance level
  α = ARCH coefficient (shock sensitivity)
  β = GARCH coefficient (persistence)
  
Constraints:
  ω > 0
  α ≥ 0
  β ≥ 0
  α + β < 1 (stationarity)
```

### **Long-Run Variance:**
```
E[σ²] = ω / (1 - α - β)
```

### **Multi-Step Forecast:**
```
E[σ²_t+h] = E[σ²] + (α+β)^h * (σ²_t - E[σ²])

As h → ∞: E[σ²_t+h] → E[σ²]  (mean reversion)
```

### **Python Implementation (Using arch library):**
```python
from arch import arch_model
import numpy as np

def fit_garch(returns):
    # Fit GARCH(1,1) model
    model = arch_model(returns, vol='Garch', p=1, q=1)
    result = model.fit(disp='off')
    
    # Extract parameters
    omega = result.params['omega']
    alpha = result.params['alpha[1]']
    beta = result.params['beta[1]']
    
    return omega, alpha, beta

def forecast_garch(omega, alpha, beta, current_var, horizon=1):
    # Long-run variance
    long_run_var = omega / (1 - alpha - beta)
    
    # Forecast
    forecast_var = long_run_var + (alpha + beta)**horizon * (current_var - long_run_var)
    
    # Annualized vol
    forecast_vol = np.sqrt(forecast_var * 252)
    
    return forecast_vol
```

---

## 📐 EWMA (Simpler Alternative to GARCH)

### **EWMA Model:**
```
σ²_t = λ*σ²_t-1 + (1-λ)*r²_t-1

Where:
  λ = decay factor (typically 0.94 for daily returns)
  
RiskMetrics uses λ = 0.94 (corresponds to ~17-day half-life)
```

### **Python Implementation:**
```python
def ewma_volatility(returns, lambda_decay=0.94):
    # Initialize
    variance = returns.iloc[0]**2
    variances = [variance]
    
    # Update recursively
    for ret in returns.iloc[1:]:
        variance = lambda_decay * variance + (1 - lambda_decay) * ret**2
        variances.append(variance)
    
    # Annualize
    volatilities = np.sqrt(np.array(variances) * 252)
    
    return pd.Series(volatilities, index=returns.index)
```

---

## 🧮 Example Calculations

### **Strategy B Example:**
```
SPY prices: [287.65, 290.12, 288.43, 291.56, ...]

Returns: [0.0085, -0.0058, 0.0108, ...]

30-day std: 0.0125 (daily)

Annualized: 0.0125 * sqrt(252) = 0.1984 = 19.84%

Use 19.84% as your vol input to BS!
```

### **Strategy C Example:**
```
Options on 2019-06-08:
  K=$285, IV=15.2%, m=-0.0089
  K=$290, IV=14.5%, m=0.0085  
  K=$295, IV=15.8%, m=0.0250

Fit: IV = 0.145 + 0.05*m + 0.2*m² + 0.01*T

For K=$292.50:
  m = ln(292.50/287.65) = 0.0167
  IV_fitted = 0.145 + 0.05*0.0167 + 0.2*0.0167² + ... = 15.1%

Compare to market IV to get signal!
```

### **Strategy D Example:**
```
GARCH parameters (fitted):
  ω = 0.000015
  α = 0.10
  β = 0.85
  
Current variance: σ²_t = 0.0004

Long-run var: E[σ²] = 0.000015/(1-0.10-0.85) = 0.0003

7-day forecast:
  σ²_t+7 = 0.0003 + 0.95^7 * (0.0004 - 0.0003)
         = 0.0003 + 0.698 * 0.0001
         = 0.00037
         
  σ_annual = sqrt(0.00037 * 252) = 30.5%
```

---

## 📊 Expected Behavior

### **Strategy B (Historical):**
- Vol is **backward-looking**
- Smooth, slow to react
- Spikes after market events (but delayed)
- Typically 10-25% for SPY

### **Strategy C (Surface):**
- Vol is **cross-sectional** (across strikes)
- Captures smile/skew
- Should be smooth across strikes
- May differ from market for illiquid options

### **Strategy D (GARCH):**
- Vol is **forward-looking** (forecast)
- Quick to react to shocks
- Mean-reverting
- Captures clustering (high vol follows high vol)

---

## 🎓 Interview Questions for Phase 3

### **Q: "How do these three strategies differ?"**

**Answer:**
> "Strategy B uses historical realized vol - what actually happened in the past 30 days. It's simple but backward-looking. Strategy C fits a smooth surface to the entire vol surface, finding relative value where individual options deviate from the fitted curve. Strategy D uses GARCH to forecast future vol based on time-series dynamics like clustering and mean reversion. Each tests a different hypothesis about option mispricing: systematic premium, cross-sectional arbitrage, or forecasting skill."

### **Q: "Which do you think works best?"**

**Answer (honest!):**
> "I don't know yet - that's what my backtesting will reveal! Academically, there's evidence that implied vol exceeds realized vol on average (vol risk premium), which would favor Strategy B. But GARCH might capture time-varying dynamics better. And Strategy C might find value in less liquid strikes. The comparison is the whole point of my project."

---

## 📁 Files Created for Phase 3

- `src/volatility/__init__.py` - Package init
- `src/volatility/historical.py` - Strategy B (3 functions)
- `src/volatility/surface.py` - Strategy C (2 classes)
- `src/volatility/forecasting.py` - Strategy D (2 classes)
- `PHASE_3_ASSIGNMENT.md` - Your assignment guide
- `PHASE_3_FORMULAS.md` - This reference

---

## ⏱️ Timeline Check

**Completed:**
- Week 1-2: Phase 1 (Data) ✓
- Week 3-4: Phase 2 (Pricing) ✓

**Remaining:**
- Week 5-7: Phase 3 (Vol Models) ← You are here
- Week 8-10: Phase 4 (Backtesting)
- Week 11-12: Phase 5 (Analysis)
- Week 13: Phase 6 (Documentation)

**Target: End of January** ✓ Still on track!

---

**Ready for Phase 3?** Start with Strategy B (historical vol) - it's the easiest and gets you warmed up! 🚀

