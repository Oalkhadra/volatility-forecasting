# Phase 4 & 5 Workflow

## Visual Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         PHASE 4: STRATEGY                           │
└─────────────────────────────────────────────────────────────────────┘

Step 1: Implement calculate_theo_price()
   ┌──────────────┐
   │  Vol Model   │ → forecast() → σ → Black-Scholes → Theo Price
   └──────────────┘
   Types: GARCH, ML, Surface, Historical

Step 2: Implement generate_signals()
   ┌────────────────┐
   │ Options Market │ → Compare → Mispricing → TradeSignal
   └────────────────┘              ↑
                              Theo Price
   
   Logic: market > theo → SELL (overpriced)
          market < theo → BUY (underpriced)

Step 3: Implement create_trade_from_signal()
   ┌──────────────┐
   │ TradeSignal  │ → Convert → Trade (Option + Stock Hedge)
   └──────────────┘

Test:
   python tests/test_strategy_manual.py

┌─────────────────────────────────────────────────────────────────────┐
│                      PHASE 5: COMPARISON                            │
└─────────────────────────────────────────────────────────────────────┘

Step 1: Setup Models
   Historical Vol  → Rolling window → σ_hist
   Surface Fit     → Polynomial     → σ_surf(K, T)
   GARCH(1,1)      → Time series    → σ_garch(h)
   XGBoost ML      → Features       → σ_ml

Step 2: Run Backtests (4x)
   For each model:
      ┌──────────┐    ┌──────────┐    ┌──────────┐
      │ Vol Model│ → │ Strategy │ → │  Engine  │ → Results
      └──────────┘    └──────────┘    └──────────┘

Step 3: Compare Results
   ┌───────────┐
   │ Results 1 │ ──┐
   │ Results 2 │   ├─→ Metrics → Sharpe, Drawdown, Returns
   │ Results 3 │   │
   │ Results 4 │ ──┘
   └───────────┘

Step 4: Visualize
   Equity Curves → plot_equity_curves()
   Drawdowns     → plot_drawdown_comparison()
   Metrics Table → create_metrics_comparison_table()

Step 5: Save & Report
   results/phase5/
      ├── equity_curves.png
      ├── drawdowns.png
      ├── metrics_comparison.csv
      └── *_equity.csv

Run:
   python scripts/run_backtest_comparison.py
```

---

## Detailed Flow: Weekly Trading Loop

```
BACKTEST PERIOD: Jan 2019 - Dec 2019
═══════════════════════════════════════════════════════

Week 1 (Saturday):
   ┌─────────────────────────────────────┐
   │ 1. Handle Expiries                  │ → Settle expired options
   │ 2. Get Options Snapshot             │ → All available options
   │ 3. Strategy.generate_signals()      │ → Find mispricings
   │ 4. Execute Option Trades            │ → Trade top N signals
   │ 5. Execute Delta Hedges             │ → Initial hedges
   └─────────────────────────────────────┘
        │
        ├─→ Monday:    Mark-to-market → Rebalance delta
        ├─→ Tuesday:   Mark-to-market → Rebalance delta
        ├─→ Wednesday: Mark-to-market → Rebalance delta
        ├─→ Thursday:  Mark-to-market → Rebalance delta
        └─→ Friday:    Mark-to-market → Rebalance delta
        
Week 2 (Saturday):
   (Repeat...)

Week N:
   Final settlements → Calculate PnL → Return results
```

---

## Data Flow: generate_signals()

```
INPUT:
   ┌─────────────────────────────────────────┐
   │ options_snapshot (DataFrame)            │
   │ ├─ strike                               │
   │ ├─ expiration                           │
   │ ├─ call_put                             │
   │ ├─ mid_price        ← Market price      │
   │ ├─ underlying_price                     │
   │ └─ days_to_expiry                       │
   └─────────────────────────────────────────┘
            ↓
   ┌─────────────────────────────────────────┐
   │ FOR EACH OPTION:                        │
   │                                         │
   │ 1. Calculate Theo Price                 │
   │    vol_model.forecast() → σ             │
   │    black_scholes(σ) → theo_price        │
   │                                         │
   │ 2. Calculate Mispricing                 │
   │    mispricing = (market - theo) / theo  │
   │                                         │
   │ 3. If |mispricing| > threshold:         │
   │    ├─ Determine action (buy/sell)       │
   │    ├─ Calculate delta (Greeks)          │
   │    ├─ Calculate hedge_qty               │
   │    └─ Create TradeSignal                │
   └─────────────────────────────────────────┘
            ↓
   ┌─────────────────────────────────────────┐
   │ Sort by |mispricing| (best first)       │
   └─────────────────────────────────────────┘
            ↓
OUTPUT:
   ┌─────────────────────────────────────────┐
   │ List[TradeSignal]                       │
   │ ├─ Top N signals (max_positions limit) │
   │ └─ Ready for execution                  │
   └─────────────────────────────────────────┘
```

---

## Vol Model Interface

```
┌─────────────────────────────────────────────────────────────────┐
│                    VOLATILITY MODELS                            │
└─────────────────────────────────────────────────────────────────┘

1. GARCH(1,1):
   garch = GARCHForecaster()
   garch.fit(returns)              # Fit on historical returns
   σ = garch.forecast(horizon=7)   # Forecast 7 days ahead
   
2. XGBoost ML:
   ml = MLForecaster(horizon=7)
   ml.fit(returns)                 # Train on historical returns
   σ = ml.forecast(returns)        # Forecast using recent data
   
3. Volatility Surface:
   surface = VolatilitySurface()
   surface.fit(options_df)         # Fit to implied vols
   σ = surface.get_vol(K, S, dte, type)  # Interpolate
   
4. Historical:
   hist_vol = 0.20                 # Simple float
   # OR
   hist_vol = calculate_realized_volatility(returns, window=30).iloc[-1]
```

---

## Trade Execution Flow

```
TradeSignal → create_trade_from_signal() → Trade → execute_trade()

EXAMPLE:
   Signal: BUY 1 call @ $5.00, strike=$300, delta=0.55
   
   ↓
   
   Trade 1 (Option):
      symbol='SPY', quantity=+1, price=5.00, type='option'
      strike=300, expiry=2019-06-21, option_type='call'
   
   Trade 2 (Hedge):
      symbol='SPY', quantity=-55, price=300.00, type='stock'
      (Hedge: -delta * contracts * 100 = -0.55 * 1 * 100 = -55 shares)
   
   ↓
   
   Portfolio:
      Cash: -$500 - $16,500 - $0.50 - $1.65 = -$17,002.15
      Positions:
         SPY_300C_20190621: +1 @ $5.00
         SPY: -55 @ $300.00
      Delta: +55 - 55 = 0 (NEUTRAL!)
```

---

## Performance Metrics Calculation

```
EQUITY CURVE:
   [100000, 100150, 99980, 100320, 100500, ...]
   
   ↓ pct_change()
   
RETURNS:
   [NaN, 0.0015, -0.0017, 0.0034, 0.0018, ...]
   
   ↓ Calculate Metrics
   
SHARPE RATIO:
   mean_return = 0.0012 (daily)
   std_return = 0.0089 (daily)
   sharpe = (0.0012 / 0.0089) * sqrt(252) = 2.14
   
MAX DRAWDOWN:
   running_max = [100000, 100150, 100150, 100320, 100500, ...]
   drawdown = equity - running_max = [0, 0, -170, 0, 0, ...]
   max_dd = -170 / 100150 = -0.17% = -0.17%
   
TOTAL RETURN:
   (final - initial) / initial = (105000 - 100000) / 100000 = 5.0%
```

---

## File Dependencies

```
scripts/run_backtest_comparison.py
   │
   ├── imports src.backtest.engine
   │   └── BacktestEngine, Portfolio, Trade
   │
   ├── imports src.backtest.strategy
   │   └── MispricingStrategy, TradeSignal
   │
   ├── imports src.volatility.forecasting
   │   └── GARCHForecaster, MLForecaster
   │
   ├── imports src.volatility.surface
   │   └── VolatilitySurface
   │
   └── imports src.volatility.historical
       └── calculate_realized_volatility

src/backtest/strategy.py
   │
   ├── imports src.pricing.pricer
   │   └── black_scholes_price
   │
   ├── imports src.pricing.greeks
   │   └── calculate_all_greeks
   │
   └── imports src.backtest.engine
       └── Trade
```

---

## Quick Reference: Key Functions

### Strategy Methods (TO IMPLEMENT)

```python
# 1. Theoretical pricing
theo_price = strategy.calculate_theo_price(S, K, T, r, option_type)

# 2. Signal generation
signals = strategy.generate_signals(
    portfolio, options_snapshot, underlying_price, 
    current_date, risk_free_rate
)

# 3. Trade creation
option_trade = strategy.create_trade_from_signal(signal, 'option')
hedge_trade = strategy.create_trade_from_signal(signal, 'stock')
```

### Engine Methods (ALREADY COMPLETE)

```python
# Portfolio
portfolio.execute_trade(trade, apply_costs=True)
portfolio.mark_to_market(prices_dict, date)
portfolio.handle_expiries(date, underlying_price)
delta = portfolio.get_portfolio_delta(S, r, date)

# Backtest
engine.load_data(options_path, prices_path, rates_path)
engine.set_strategy(strategy)
results = engine.run()
```

### Comparison Functions (TO IMPLEMENT)

```python
# Model setup
hist_model = setup_historical_model(prices)
garch_model = setup_garch_model(returns)
ml_model = setup_ml_model(returns)
surf_model = setup_surface_model(options)

# Execution
results, engine = run_single_backtest(model_name, vol_model)
all_results = run_all_backtests()

# Analysis
metrics = calculate_performance_metrics(equity_curve)
comparison_df = create_metrics_comparison_table(all_results)

# Visualization
plot_equity_curves(all_results, save_path)
plot_drawdown_comparison(all_results, save_path)
```

---

## Common Patterns

### Pattern 1: Vol Model Dispatch
```python
if isinstance(vol_model, GARCHForecaster):
    sigma = vol_model.forecast(horizon=int(T*365))
elif isinstance(vol_model, MLForecaster):
    sigma = vol_model.forecast(returns_data)
elif isinstance(vol_model, VolatilitySurface):
    sigma = vol_model.get_vol(K, S, int(T*365), option_type)
else:
    sigma = vol_model if not callable(vol_model) else vol_model(S, K, T)
```

### Pattern 2: Signal Filtering
```python
signals = []
for idx, row in options_snapshot.iterrows():
    # Extract params
    # Calculate theo
    # Calculate mispricing
    if abs(mispricing_pct) > threshold:
        signals.append(create_signal(...))

# Sort and limit
signals.sort(key=lambda s: abs(s.mispricing_pct), reverse=True)
return signals[:max_positions - current_positions]
```

### Pattern 3: Performance Calculation
```python
returns = equity.pct_change().dropna()
sharpe = (returns.mean() / returns.std()) * np.sqrt(252)
running_max = equity.expanding().max()
max_dd = ((equity - running_max) / running_max).min()
```

---

## Debugging Checklist

If backtest fails:

1. ✅ Check data loaded correctly
   ```python
   print(engine.options_data.shape)
   print(engine.prices_data.head())
   ```

2. ✅ Check vol model returns valid values
   ```python
   sigma = vol_model.forecast(...)
   print(f"Vol: {sigma:.2%}")  # Should be 5% - 50%
   ```

3. ✅ Check signals generated
   ```python
   signals = strategy.generate_signals(...)
   print(f"Signals: {len(signals)}")
   for s in signals:
       print(f"  {s.action} {s.option_type} {s.mispricing_pct:.2%}")
   ```

4. ✅ Check trades executed
   ```python
   print(f"Trades: {len(portfolio.trade_log)}")
   print(f"Positions: {len(portfolio.positions)}")
   ```

5. ✅ Check portfolio delta
   ```python
   delta = portfolio.get_portfolio_delta(S, r, date)
   print(f"Delta: {delta:.0f} shares")  # Should be near 0
   ```

---

## Timeline

### Day 1: Phase 4
- Morning: Implement `calculate_theo_price()` (1 hour)
- Afternoon: Implement `generate_signals()` (2 hours)
- Evening: Implement `create_trade_from_signal()` + test (1 hour)

### Day 2: Phase 5 Setup
- Morning: Model setup functions (2 hours)
- Afternoon: Backtest execution functions (2 hours)

### Day 3: Phase 5 Complete
- Morning: Performance metrics + visualization (2 hours)
- Afternoon: Run backtests + analyze results (2 hours)
- Evening: Write findings + prepare interview answers (2 hours)

**Total: ~14-16 hours** (Spread over 3 days for reflection time)

---

## Success Indicators

### Phase 4 Working
- ✅ Test suite passes (4/4 tests)
- ✅ Mini-backtest completes without errors
- ✅ Generates 5-20 signals per week
- ✅ Executes both option and hedge trades
- ✅ Portfolio delta stays near zero

### Phase 5 Working
- ✅ All 4 backtests complete
- ✅ Sharpe ratios in reasonable range (-0.5 to 2.0)
- ✅ Equity curves show variation between models
- ✅ Plots generated and saved
- ✅ CSV files created with results

### Interview Ready
- ✅ Can explain project in 2 minutes
- ✅ Understand why each model performed as it did
- ✅ Can discuss design decisions and trade-offs
- ✅ Prepared for technical deep-dives
- ✅ Have specific examples and numbers ready

---

**You're now ready to begin Phase 4! Start with `src/backtest/strategy.py`** 🚀

