# Phase 4 & 5: Complete Strategy Implementation & Comparison

## Quick Start

### Phase 4: Complete Strategy (2-3 hours)

**Goal**: Implement 3 methods in `src/backtest/strategy.py`

```bash
# 1. Open strategy.py and implement TODOs
code src/backtest/strategy.py

# 2. Test your implementation
python tests/test_strategy_manual.py

# 3. If tests pass, move to Phase 5
```

**Files to Edit**:
- `src/backtest/strategy.py` - Implement 3 methods with TODOs

**Methods to Implement**:
1. ✅ `calculate_theo_price()` - Get vol forecast, price with BS
2. ✅ `generate_signals()` - Find mispricings, create signals
3. ✅ `create_trade_from_signal()` - Convert signal to trade

---

### Phase 5: Compare Models (4-5 hours)

**Goal**: Run 4 backtests, compare results

```bash
# 1. Open comparison script and implement TODOs
code scripts/run_backtest_comparison.py

# 2. Run comparison
python scripts/run_backtest_comparison.py

# 3. Review results
ls -lh results/phase5/
```

**Files to Edit**:
- `scripts/run_backtest_comparison.py` - Implement all TODOs

**Functions to Implement**:
1. ✅ Model setup (4 functions)
2. ✅ Backtest execution
3. ✅ Performance metrics
4. ✅ Visualization
5. ✅ Results saving

---

## Project Structure

```
options-trading/
├── src/
│   └── backtest/
│       ├── engine.py          # ✅ Complete (Portfolio, BacktestEngine)
│       └── strategy.py        # ⚠️ TODO: Implement 3 methods
├── scripts/
│   └── run_backtest_comparison.py  # ⚠️ TODO: Implement full script
├── tests/
│   └── test_strategy_manual.py     # ✅ Use for testing
├── results/
│   └── phase5/                     # Output directory
├── PHASE_4_GUIDE.md           # Legacy guide
├── PHASE_5_GUIDE.md           # 📖 Detailed implementation guide
└── PHASE_4_5_README.md        # 📖 This file (quick reference)
```

---

## Implementation Checklist

### Phase 4: Strategy Implementation

#### Task 1: `calculate_theo_price()`
- [ ] Handle 4 model types (GARCH, ML, Surface, Historical)
- [ ] Get vol forecast from model
- [ ] Price with Black-Scholes
- [ ] Return theo price

**Hint**: Use `isinstance()` to check model type

```python
if isinstance(self.vol_model, GARCHForecaster):
    sigma = self.vol_model.forecast(horizon=int(T*365))
elif isinstance(self.vol_model, MLForecaster):
    sigma = self.vol_model.forecast(self.returns_data)
# ... etc
```

---

#### Task 2: `generate_signals()`
- [ ] Count current positions (use `count_option_positions()`)
- [ ] Loop through options snapshot
- [ ] Calculate theo price for each option
- [ ] Calculate mispricing: `(market - theo) / theo`
- [ ] If `|mispricing| > threshold`: Create TradeSignal
- [ ] Calculate delta and hedge quantity
- [ ] Sort signals by mispricing magnitude
- [ ] Return top N signals

**Key Logic**:
```python
mispricing_pct = (market_price - theo_price) / theo_price

if mispricing_pct > 0:
    action = 'sell'  # Overpriced
else:
    action = 'buy'   # Underpriced
```

---

#### Task 3: `create_trade_from_signal()`
- [ ] If `trade_type == 'option'`: Create option trade
- [ ] If `trade_type == 'stock'`: Create hedge trade
- [ ] Return Trade object

**Structure**:
```python
if trade_type == 'option':
    return Trade(
        symbol=signal.symbol,
        quantity=signal.quantity,  # Signed correctly
        price=signal.market_price,
        trade_type='option',
        timestamp=signal.timestamp,
        strike=signal.strike,
        expiry=signal.expiry,
        option_type=signal.option_type
    )
```

---

### Phase 5: Comparison Implementation

#### Task 1: Model Setup (4 functions)
- [ ] `setup_historical_model()` - Calculate rolling vol
- [ ] `setup_surface_model()` - Fit volatility surface
- [ ] `setup_garch_model()` - Fit GARCH(1,1)
- [ ] `setup_ml_model()` - Fit XGBoost

---

#### Task 2: Backtest Execution
- [ ] `run_single_backtest()` - Run one backtest
- [ ] `run_all_backtests()` - Run all 4 models

**Pattern**:
```python
# Create engine
engine = BacktestEngine(start_date, end_date, initial_capital)

# Load data
engine.load_data(options_path, prices_path, rates_path)

# Create strategy
strategy = MispricingStrategy(vol_model, ...)

# Run
engine.set_strategy(strategy)
results = engine.run()
```

---

#### Task 3: Performance Metrics
- [ ] `calculate_performance_metrics()` - Sharpe, drawdown, etc.
- [ ] `create_metrics_comparison_table()` - Compare all models

**Metrics to Calculate**:
- Total return
- Annualized return
- Sharpe ratio: `(mean_return / std_return) * sqrt(252)`
- Max drawdown: Worst peak-to-trough decline
- Win rate: % of positive return days
- Volatility: Annualized std dev

---

#### Task 4: Visualization
- [ ] `plot_equity_curves()` - Show all models
- [ ] `plot_drawdown_comparison()` - OPTIONAL
- [ ] `plot_metrics_bar_chart()` - OPTIONAL

---

#### Task 5: Main Execution
- [ ] `main()` - Orchestrate everything
- [ ] Print configuration
- [ ] Run backtests
- [ ] Calculate metrics
- [ ] Generate plots
- [ ] Save results
- [ ] Announce winner

---

## Testing Your Work

### Phase 4 Tests

```bash
# Run manual test suite
python tests/test_strategy_manual.py
```

**Expected Output**:
```
✓ PASS - calculate_theo_price
✓ PASS - generate_signals
✓ PASS - create_trade_from_signal
✓ PASS - mini_backtest

🎉 All tests passed! Ready for Phase 5!
```

---

### Phase 5 Tests

```bash
# Run full comparison
python scripts/run_backtest_comparison.py
```

**Expected Output**:
```
Running backtests...
  Historical... Complete!
  Surface... Complete!
  GARCH... Complete!
  ML... Complete!

PERFORMANCE COMPARISON
======================
              Sharpe Ratio  Total Return (%)  Max Drawdown (%)
Model                                                          
GARCH                 0.85              5.68             -3.21
ML                    0.62              3.46             -4.12
Historical            0.45              2.35             -5.67
Surface              -0.23             -1.23             -6.89

🏆 WINNER: GARCH (Sharpe: 0.85)
```

---

## Common Issues & Solutions

### Issue 1: "Method not implemented"
**Cause**: Forgot to remove `raise NotImplementedError()`
**Fix**: Replace with actual implementation code

### Issue 2: "No signals generated"
**Cause**: Mispricing threshold too high or vol model broken
**Fix**: 
- Check vol model returns reasonable values (0.1 - 0.5)
- Lower threshold to 0.05 for testing
- Add debug prints in `generate_signals()`

### Issue 3: "Negative option prices"
**Cause**: Vol forecast is invalid (negative or too high)
**Fix**: Add bounds checking:
```python
sigma = max(0.05, min(2.0, sigma))  # Clip to [5%, 200%]
```

### Issue 4: "Index out of bounds"
**Cause**: Not enough historical data for ML/GARCH
**Fix**: Ensure returns series has 60+ days before first backtest week

### Issue 5: "Module not found"
**Cause**: Running from wrong directory
**Fix**: Always run from project root:
```bash
cd /home/oalkhadra/options-trading
python scripts/run_backtest_comparison.py
```

---

## Expected Results

### Typical Performance Ranges

**Sharpe Ratios** (2019 data):
- Excellent: > 1.0
- Good: 0.5 - 1.0
- Okay: 0.0 - 0.5
- Poor: < 0.0

**Total Returns** (1 year):
- With 10% threshold: 0% - 8%
- Transaction costs eat into returns
- Higher threshold = fewer trades = lower returns but better risk-adjusted

**Max Drawdowns**:
- Typical: -5% to -10%
- Concerning: > -15%

---

## Interview Preparation

### 2-Minute Project Summary

"I built an options trading strategy comparison framework in Python. The strategy trades SPY weekly options based on mispricing between theoretical prices (from volatility models) and market prices, maintaining delta-neutrality through daily stock hedging.

I implemented 4 volatility forecasting methods: historical baseline, volatility surface fitting, GARCH(1,1) econometric modeling, and XGBoost machine learning. Each model forecasts future volatility, which I use to price options with Black-Scholes.

The strategy identifies mispricings > 10%, trades the most mispriced options (up to 5 positions), and rebalances delta hedges daily. I backtested on 2019 SPY data with realistic transaction costs.

Results showed [YOUR RESULTS]. The project has ~2,500 lines of code with end-to-end implementation: data pipeline, pricing engine, Greeks calculator, volatility models, backtesting framework, and performance analysis."

---

### Key Questions to Prepare

1. **"Why delta-neutral?"**
   - Isolates vol bet from directional exposure
   - PnL comes from vol forecast accuracy, not market direction
   - Market-neutral strategy suitable for all environments

2. **"Which model performed best?"**
   - [Your answer based on results]
   - Explain why (e.g., GARCH captures vol clustering)

3. **"What are limitations?"**
   - No bid-ask spreads modeled
   - Weekly trading only (miss intraweek opportunities)
   - Single underlying (concentration risk)
   - Historical backtest (may not generalize)

4. **"How would you improve?"**
   - Ensemble forecasts (combine models)
   - Dynamic position sizing by conviction
   - Add liquidity filters (volume, spreads)
   - Multi-asset diversification

5. **"What did you learn?"**
   - Vol forecasting is hard but valuable
   - Transaction costs matter enormously
   - Model sophistication doesn't guarantee better results
   - Risk management is critical

---

## Next Steps After Phase 5

### Short-term (1-2 days)
1. Parameter sensitivity analysis
   - Test different thresholds (5%, 15%, 20%)
   - Test different max_positions (3, 5, 10)
   - Document results

2. Write findings report
   - Which model won and why
   - Economic interpretation
   - Challenges encountered

### Medium-term (1 week)
1. Add Phase 6 enhancements
   - Bid-ask spread modeling
   - Implied vol error tracking
   - Trade-by-trade PnL attribution

2. Test on different data
   - 2020 (COVID volatility)
   - Different underlyings (QQQ, IWM)

### Long-term (Optional)
1. Live trading preparation
   - Real-time data integration
   - Order management system
   - Risk monitoring

2. Advanced strategies
   - Volatility carry
   - Dispersion trading
   - Term structure arbitrage

---

## Resources

### Documentation
- `PHASE_5_GUIDE.md` - Detailed implementation guide with code examples
- `PHASE_4_GUIDE.md` - Legacy backtest engine documentation
- `README.md` - Project overview and setup

### Code References
- `src/backtest/engine.py` - Review for Trade, Position, Portfolio classes
- `src/pricing/` - Black-Scholes and Greeks implementations
- `src/volatility/` - All vol model implementations

### Learning Resources
- Hull's textbook - Options fundamentals
- Quantopian lectures - Algorithmic trading
- QuantConnect tutorials - Strategy backtesting

---

## Success Criteria

### Phase 4 Complete ✅
- All 3 methods implemented
- Manual tests pass
- Mini-backtest runs without errors
- Can generate signals and execute trades

### Phase 5 Complete ✅
- All 4 backtests run successfully
- Performance metrics calculated correctly
- Equity curves plotted
- Results saved to CSV
- Winner identified

### Interview Ready ✅
- Can explain project in 2 minutes
- Understand each model's strengths/weaknesses
- Can discuss results and economic intuition
- Prepared for technical deep-dives
- Can explain design decisions

---

## File Summary

| File | Status | Purpose |
|------|--------|---------|
| `src/backtest/strategy.py` | ⚠️ TODO | Implement 3 methods |
| `scripts/run_backtest_comparison.py` | ⚠️ TODO | Implement comparison |
| `tests/test_strategy_manual.py` | ✅ Ready | Test your implementation |
| `PHASE_5_GUIDE.md` | ✅ Ready | Detailed guide with examples |
| `PHASE_4_5_README.md` | ✅ Ready | Quick reference (this file) |

---

## Getting Help

If stuck:
1. Check `PHASE_5_GUIDE.md` for detailed examples
2. Review error messages carefully
3. Add debug prints to narrow down issues
4. Test each function individually before integration
5. Check data schema matches expectations

---

**Good luck! 🚀**

Remember: The goal is learning and demonstrating skills. Focus on understanding the concepts and being able to explain your design decisions.

