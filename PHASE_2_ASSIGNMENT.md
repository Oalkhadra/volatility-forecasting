# Phase 2: Black-Scholes Pricer - Your First Assignment

## 🎯 Goal

Implement the Black-Scholes option pricing model from scratch. This is the foundation of your entire project - every strategy will use this to calculate theoretical prices.

---

## 📝 Assignment: Complete `src/pricer.py`

You have **5 functions to implement**:

### 1. `black_scholes_price()` ⭐ (Most Important)

**What it does**: Calculates theoretical option price given volatility

**The Math**:
```
d1 = [ln(S/K) + (r + σ²/2)T] / (σ√T)
d2 = d1 - σ√T

Call Price = S·N(d1) - K·e^(-rT)·N(d2)
Put Price = K·e^(-rT)·N(-d2) - S·N(-d1)
```

**Edge Cases to Handle**:
- T = 0 (expiration) → Return intrinsic value
- σ = 0 (no volatility) → Return discounted intrinsic value

**Hints**:
- Use `np.log()` for natural logarithm
- Use `np.sqrt()` for square root
- Use `np.exp()` for exponential
- Use `norm.cdf()` for cumulative normal distribution N(x)

---

### 2. `intrinsic_value()` (Helper Function)

**What it does**: Calculates what option would be worth if exercised now

**The Math**:
```
Call Intrinsic = max(S - K, 0)
Put Intrinsic = max(K - S, 0)
```

**Easy!** Just one line per option type.

---

### 3. `time_value()` (Helper Function)

**What it does**: Calculates premium above intrinsic value

**The Math**:
```
Time Value = Option Price - Intrinsic Value
```

**Hint**: Call `intrinsic_value()` that you already wrote!

---

### 4. `moneyness()` (Classification Function)

**What it does**: Determines if option is ITM/ATM/OTM

**The Logic**:
```
ATM: |S - K| / S < 0.005 (within 0.5%)
ITM Call: S > K
OTM Call: S < K
ITM Put: K > S
OTM Put: K < S
```

---

### 5. Edge Cases (Part of `black_scholes_price`)

You must handle:
- **At expiration (T = 0)**: Return intrinsic value
- **Zero volatility (σ = 0)**: Return `max(S - K*e^(-rT), 0)` for call

---

## ✅ How to Test Your Work

### Option 1: Built-in Validation

```bash
cd /home/oalkhadra/options-trading
python3 src/pricer.py
```

This runs 5 test cases against known academic values. You should see:
```
✓ PASS - ATM Call (Hull Example 13.6)
✓ PASS - ATM Put (Hull Example 13.6)
✓ PASS - OTM Call
✓ PASS - ITM Put
✓ PASS - Deep ITM Call

✓ ALL TESTS PASSED - Pricer is working correctly!
```

### Option 2: Unit Tests (More Rigorous)

```bash
pytest tests/test_pricer.py -v
```

This runs 20+ unit tests including:
- Basic pricing tests
- Put-call parity verification
- Edge case handling
- Input validation
- Sensitivity tests (vol, time)

---

## 📚 Resources

### Black-Scholes Formula Explanation
- [Investopedia - Black-Scholes](https://www.investopedia.com/terms/b/blackscholes.asp)
- Hull's "Options, Futures, and Other Derivatives" - Chapter 15

### Python Libraries
- `numpy`: Mathematical operations
- `scipy.stats.norm`: Normal distribution functions

### Example Implementation (Don't peek until you try!)
```python
# d1 calculation example:
d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
```

---

## 🎓 What You'll Learn

1. **Options pricing theory** - Black-Scholes is THE fundamental model
2. **Numerical methods** - Handling edge cases, precision
3. **Python scientific computing** - numpy, scipy
4. **Test-driven development** - Write code that passes tests
5. **Professional coding** - Type hints, docstrings, validation

---

## 💡 Interview Questions to Prepare

While implementing, think about:

1. **"What are the assumptions of Black-Scholes?"**
   - European options (no early exercise)
   - Constant volatility
   - Constant risk-free rate
   - No dividends
   - Log-normal stock prices

2. **"Why does higher volatility increase option prices?"**
   - Higher vol = higher probability of large moves
   - Asymmetric payoff: unlimited upside, limited downside
   - More optionality = more valuable

3. **"What happens to time value as expiration approaches?"**
   - Time decay (theta)
   - At expiration: time value = 0, price = intrinsic value

4. **"How would you handle American options?"**
   - Black-Scholes only works for European
   - Need binomial tree or finite difference methods
   - Early exercise complicates things

---

## ⏱️ Time Estimate

- **Reading/Understanding**: 30-45 minutes
- **Implementation**: 2-3 hours
- **Testing/Debugging**: 1-2 hours
- **Total**: 4-6 hours

Don't rush! Understanding this deeply is crucial for the rest of the project.

---

## 🚀 Next Steps After Completion

Once your pricer passes all tests:

1. **Validate on real data** - Use notebooks to test on SPY options
2. **Compare to Dolt Greeks** - Check if your prices match market
3. **Move to Phase 2b** - Implement Greeks calculator

---

## 🆘 Getting Stuck?

**Common Issues**:

1. **"My prices are way off"**
   - Check units: T in years? sigma annualized?
   - Check d1/d2 formula carefully
   - Print intermediate values to debug

2. **"Tests pass but validation fails"**
   - Might be rounding differences
   - Check if you're using norm.cdf() correctly

3. **"Put-call parity doesn't hold"**
   - Usually means d1 or d2 calculation is wrong
   - Double-check the formulas

**Debug Strategy**:
```python
# Add prints to see intermediate values
print(f"d1: {d1}, d2: {d2}")
print(f"N(d1): {norm.cdf(d1)}, N(d2): {norm.cdf(d2)}")
print(f"Call: {call_price}, Put: {put_price}")
```

---

## 📊 Expected Output

When done, you should be able to:

```python
from src.pricer import black_scholes_price

# Price an ATM call
price = black_scholes_price(
    S=100,      # SPY at $100
    K=100,      # Strike $100
    T=30/365,   # 30 days to expiration
    r=0.02,     # 2% risk-free rate
    sigma=0.20, # 20% implied vol
    option_type='call'
)

print(f"Theoretical call price: ${price:.2f}")
# Should output something like: $2.35
```

---

## ✨ Bonus Challenge (Optional)

If you finish early and want more practice:

1. **Add dividend support**: Modify formula to include dividend yield
2. **Vectorize the function**: Make it work on arrays for speed
3. **Add Greeks**: delta, gamma, vega (we'll do this in Phase 2b)

---

**Ready to start?** Open `src/pricer.py` and start filling in the TODOs! 

Good luck! 🎯

