# Black-Scholes Formula Quick Reference

## 📐 The Formulas

### Core Parameters
- **S** = Current stock price
- **K** = Strike price  
- **T** = Time to expiration (in years)
- **r** = Risk-free rate (annual, as decimal)
- **σ** = Volatility (annual, as decimal)
- **N(x)** = Cumulative standard normal distribution

### d1 and d2
```
d₁ = [ln(S/K) + (r + σ²/2)T] / (σ√T)

d₂ = d₁ - σ√T
```

### Option Prices
```
Call = S·N(d₁) - K·e^(-rT)·N(d₂)

Put = K·e^(-rT)·N(-d₂) - S·N(-d₁)
```

---

## 🐍 Python Implementation

```python
import numpy as np
from scipy.stats import norm

def black_scholes_call(S, K, T, r, sigma):
    """Calculate call option price."""
    
    # Calculate d1
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    
    # Calculate d2
    d2 = d1 - sigma*np.sqrt(T)
    
    # Calculate call price
    call = S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    
    return call


def black_scholes_put(S, K, T, r, sigma):
    """Calculate put option price."""
    
    # Calculate d1 and d2 (same as call)
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    
    # Calculate put price
    put = K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)
    
    return put
```

---

## 🧮 Example Calculation (By Hand)

**Given:**
- S = 100
- K = 100  
- T = 1.0 year
- r = 0.05 (5%)
- σ = 0.20 (20%)

**Step 1: Calculate d₁**
```
d₁ = [ln(100/100) + (0.05 + 0.20²/2)×1] / (0.20×√1)
   = [0 + (0.05 + 0.02)×1] / 0.20
   = 0.07 / 0.20
   = 0.35
```

**Step 2: Calculate d₂**
```
d₂ = 0.35 - 0.20×√1
   = 0.35 - 0.20
   = 0.15
```

**Step 3: Look up N(d₁) and N(d₂)**
```
N(0.35) = 0.6368  (from normal table)
N(0.15) = 0.5596
```

**Step 4: Calculate Call Price**
```
Call = 100×0.6368 - 100×e^(-0.05×1)×0.5596
     = 63.68 - 100×0.9512×0.5596
     = 63.68 - 53.23
     = $10.45
```

**Step 5: Calculate Put Price (using same d₁, d₂)**
```
N(-0.15) = 0.4404
N(-0.35) = 0.3632

Put = 100×e^(-0.05×1)×0.4404 - 100×0.3632
    = 100×0.9512×0.4404 - 36.32
    = 41.89 - 36.32
    = $5.57
```

**Verify Put-Call Parity:**
```
C - P = S - K·e^(-rT)
10.45 - 5.57 = 100 - 100×e^(-0.05)
4.88 ≈ 4.88 ✓
```

---

## 🔍 Intuition Behind the Formula

### What is N(d₁)?
- **Risk-neutral probability** that option expires in the money
- For calls: probability S > K at expiration
- Adjusted for stock's drift

### What is N(d₂)?  
- **Actual probability** that option expires in the money
- Used to discount the strike payment

### Why e^(-rT)?
- **Present value** of strike price
- Money paid/received at expiration is worth less today

### Why S·N(d₁) - K·e^(-rT)·N(d₂)?
- **Expected present value** of payoff
- N(d₁) weights the stock price
- N(d₂) weights the strike payment

---

## 📊 Properties to Remember

### Vega (∂V/∂σ)
- Higher volatility → Higher option price
- True for both calls and puts
- ATM options have highest vega

### Theta (∂V/∂T)
- Time decay
- Options lose value as expiration approaches
- ATM options have highest theta (in absolute terms)

### Delta (∂V/∂S)
- Call delta: 0 to 1
- Put delta: -1 to 0
- ATM options have delta ≈ 0.5 (calls) or ≈ -0.5 (puts)

### Gamma (∂²V/∂S²)
- Rate of change of delta
- Highest for ATM options
- Same for calls and puts

---

## 🎯 Common Mistakes

1. **Wrong units**
   - T must be in YEARS (divide days by 365)
   - r and σ must be ANNUALIZED

2. **Forgetting negative signs**
   - Put formula uses N(-d₁) and N(-d₂)

3. **Using log10 instead of ln**
   - Use `np.log()` not `np.log10()`

4. **Not handling T=0**
   - Division by zero when calculating d₁
   - Return intrinsic value instead

5. **Wrong normal distribution**
   - Use `norm.cdf()` not `norm.pdf()`
   - cdf = cumulative, pdf = probability density

---

## 📱 Quick Test Values

| S | K | T | r | σ | Call | Put |
|---|---|---|---|---|------|-----|
| 100 | 100 | 1.0 | 0.05 | 0.20 | 10.45 | 5.57 |
| 100 | 100 | 0.5 | 0.05 | 0.20 | 7.97 | 4.08 |
| 100 | 100 | 1.0 | 0.05 | 0.30 | 13.27 | 8.34 |
| 110 | 100 | 1.0 | 0.05 | 0.20 | 16.73 | 1.81 |
| 90 | 100 | 1.0 | 0.05 | 0.20 | 4.68 | 9.76 |

Use these to validate your implementation!

---

## 🎓 For Interviews

**Q: "Walk me through the Black-Scholes formula."**

**Good Answer:**
> "Black-Scholes prices European options using five inputs: spot price, strike, time to expiration, risk-free rate, and volatility. The formula calculates d₁ and d₂, which represent risk-neutral probabilities. For a call, we take the expected stock value at expiration (S times N of d₁) minus the present value of the strike payment (K times e to the minus rT times N of d₂). The key insight is that we're calculating the expected present value of the payoff under risk-neutral dynamics."

**Follow-up Q: "What are the assumptions?"**
- European exercise only
- Constant volatility
- Constant risk-free rate  
- No dividends
- Frictionless markets (no transaction costs)
- Continuous trading
- Log-normal stock returns

