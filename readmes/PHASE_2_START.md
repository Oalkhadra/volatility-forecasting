# 🚀 Phase 2 Started: Black-Scholes Pricer

## 📦 Files Created

### Implementation File
**`src/pricer.py`** - Your assignment!
- 5 functions with TODO comments
- Input validation included
- Built-in validation function
- Comprehensive docstrings

### Test File  
**`tests/test_pricer.py`** - 20+ unit tests
- Test basic pricing
- Test put-call parity
- Test edge cases
- Test input validation
- Run with: `pytest tests/test_pricer.py -v`

### Documentation
**`PHASE_2_ASSIGNMENT.md`** - Full assignment guide
- What to implement
- How to test
- Common mistakes
- Interview prep questions
- Estimated time: 4-6 hours

**`BS_FORMULA_REFERENCE.md`** - Formula reference
- Black-Scholes equations
- Python implementation examples
- Hand calculation walkthrough
- Quick test values
- Interview talking points

---

## ✅ Your Task

### Step 1: Read the Assignment
```bash
cat PHASE_2_ASSIGNMENT.md
```

### Step 2: Implement the Functions

Open `src/pricer.py` and fill in the TODOs:

1. ✏️ **`black_scholes_price()`** - Main pricing function
2. ✏️ **`intrinsic_value()`** - Helper function  
3. ✏️ **`time_value()`** - Helper function
4. ✏️ **`moneyness()`** - Classification function
5. ✏️ **Handle edge cases** - T=0, sigma=0

### Step 3: Test Your Implementation

**Quick validation:**
```bash
python3 src/pricer.py
```

**Full test suite:**
```bash
pytest tests/test_pricer.py -v
```

### Step 4: Verify on Real Data

Once tests pass, we'll validate on SPY options in a notebook.

---

## 📚 Resources Available

### Formula Reference
The `BS_FORMULA_REFERENCE.md` file has:
- Complete formulas with explanations
- Example calculation by hand
- Python code snippets
- Test values to validate against

### Key Formula
```
d₁ = [ln(S/K) + (r + σ²/2)T] / (σ√T)
d₂ = d₁ - σ√T

Call = S·N(d₁) - K·e^(-rT)·N(d₂)
Put = K·e^(-rT)·N(-d₂) - S·N(-d₁)
```

### Python Functions You'll Need
- `np.log()` - Natural logarithm
- `np.sqrt()` - Square root
- `np.exp()` - Exponential
- `norm.cdf()` - Cumulative normal distribution

---

## 🎯 Success Criteria

Your implementation is correct when:

1. ✅ Built-in validation passes all 5 test cases
2. ✅ Pytest passes all 20+ unit tests
3. ✅ Put-call parity holds (tested automatically)
4. ✅ Edge cases handled (T=0, sigma=0)
5. ✅ Prices match academic benchmarks within $0.10

---

## 💡 Tips

### Start Simple
1. Implement `intrinsic_value()` first (easiest)
2. Then `time_value()` (uses intrinsic_value)
3. Then `moneyness()` (just classification)
4. Finally `black_scholes_price()` (the main one)

### Debug Strategy
If your prices are wrong:
```python
# Add prints to debug
print(f"d1: {d1:.4f}, d2: {d2:.4f}")
print(f"N(d1): {norm.cdf(d1):.4f}")
print(f"N(d2): {norm.cdf(d2):.4f}")
```

### Common Issues
- **Prices way off?** Check if T is in years (divide days by 365)
- **Negative prices?** Check the formulas carefully
- **Tests fail at expiration?** Handle T=0 edge case

---

## 📊 What Your Data Looks Like

You have one SPY option in your data:
```json
{
  "date": "2019-06-08",
  "strike": 245,
  "call_put": "Call",
  "mid_price": 43.04,
  "days_to_expiry": 13,
  "underlying_price": 260.60,
  "risk_free_rate": 0.02,
  "vol": 0.5503  // Market implied vol from Dolt
}
```

After implementing, you could price this option:
```python
theo_price = black_scholes_price(
    S=260.60,
    K=245,
    T=13/365,
    r=0.02,
    sigma=0.5503,  # Using market IV
    option_type='call'
)
# Should get ~43.04 (very close to market price)
```

---

## ⏱️ Time Estimate

- **Understanding formulas**: 30-45 min
- **Implementation**: 2-3 hours
- **Testing & debugging**: 1-2 hours
- **Total**: 4-6 hours

Don't rush! This is foundational for everything else.

---

## 🆘 If You Get Stuck

1. **Check the formula reference** - Step-by-step example
2. **Run built-in validation** - See which test cases fail
3. **Add debug prints** - See intermediate values
4. **Compare to reference** - Hand-calculate a simple case

---

## 🎓 Why This Matters

This isn't just coding practice - you're building the core of a real options pricing system. When you interview:

**They'll ask**: "Have you implemented Black-Scholes?"

**You'll say**: "Yes, I built a complete Black-Scholes pricer from scratch with unit tests. I validated it against academic benchmarks and then used it to price thousands of SPY options for my volatility trading strategies. The implementation handles edge cases like options at expiration and supports both calls and puts with proper put-call parity."

**That's impressive.** 🎯

---

## 🚀 Next Phase Preview

After you complete this, Phase 2b will be:
- **Greeks calculator** (delta, gamma, vega, theta, rho)
- **Implied volatility solver** (Newton-Raphson)
- **Validation notebook** (compare to real SPY data)

But first, master the pricer!

---

## ✨ Ready to Start?

```bash
# Open the file and start coding!
code src/pricer.py

# Or use your favorite editor
nano src/pricer.py
vim src/pricer.py
```

**Good luck!** Take your time, understand the math, and build something you're proud of. 💪

