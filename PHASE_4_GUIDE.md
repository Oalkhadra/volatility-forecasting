# Phase 4: Backtesting Engine - Updated with Daily Rebalancing

## 🎯 Strategy Overview

**Weekly Options Trading + Daily Delta Rebalancing**

### Weekly (Saturdays/Sundays):
1. Handle expiries from previous week
2. Calculate theoretical prices using vol model
3. Find mispriced options (theo vs. market)
4. Execute option trades
5. Execute initial delta hedges

### Daily (Mon-Fri):
1. Mark all positions to market
2. Recalculate option deltas
3. Rebalance stock hedge to maintain delta-neutrality
4. Log PnL

## 💰 Transaction Costs

**Options:** $0.50 per contract  
**Stock:** 1 basis point (0.01%) of notional

These are realistic retail execution costs and will help determine if the strategy is economically viable.

## 🏗 Implementation Status

### ✅ Portfolio Class (COMPLETE)
- [x] `execute_trade()` - Handles trades with transaction costs
- [x] `mark_to_market()` - Updates position values daily
- [x] `handle_expiries()` - Settles ITM/OTM options
- [x] `get_portfolio_delta()` - Calculates total delta

### 📝 BacktestEngine Class (TODO)
- [ ] `load_data()` - Load options and daily prices
- [ ] `rebalance_delta()` - Daily hedge rebalancing
- [ ] `run()` - Main weekly/daily loop

### 📝 MispricingStrategy Class (TODO)
- [ ] `calculate_theo_price()` - Get vol from model, price with BS
- [ ] `generate_signals()` - Find mispricings, create signals

## 🔄 Backtest Loop Structure

```python
for week_date in trading_saturdays:
    # WEEKLY: New option positions
    handle_expiries()
    signals = strategy.generate_signals()
    execute_option_trades(signals)
    execute_initial_hedges(signals)
    
    # DAILY: Rebalance loop
    for day in get_weekdays(week_date):
        mark_to_market()
        rebalance_delta()  # <-- NEW
        log_pnl()
```

## 📊 Expected Output

The backtest will produce a DataFrame with columns:
- `date`: Daily timestamps
- `equity`: Total portfolio value
- `cash`: Cash balance
- `num_positions`: Number of open positions
- (Optional) `delta`: Portfolio delta after rebalancing

## 🚀 Next Steps

1. **Implement `BacktestEngine.load_data()`**
   - Load `spy_options.parquet` and `spy_prices.parquet`
   - Filter by date range
   - Prepare for iteration

2. **Implement `BacktestEngine.rebalance_delta()`**
   - Calculate portfolio delta
   - Determine required stock hedge
   - Execute rebalance trade

3. **Implement `BacktestEngine.run()`**
   - Outer loop: Weekly trading
   - Inner loop: Daily rebalancing
   - Return results DataFrame

4. **Implement `MispricingStrategy`**
   - Calculate theo prices
   - Generate signals
   - Create delta-hedge trades

## 💡 Key Design Decisions

### Why Daily Rebalancing?
- More realistic (market makers rebalance continuously)
- Isolates vol edge from delta drift
- You have daily data (no extra burden)

### Why Transaction Costs?
- Tests economic viability
- Prevents overfitting to noise
- Realistic for interviews

### Why Weekly Options Trading?
- Matches your data frequency (Saturday snapshots)
- Reduces noise from intraweek volatility
- Focuses on persistent mispricings

## 📂 Files

- `src/backtest/engine.py` - Portfolio and BacktestEngine ✅
- `src/backtest/strategy.py` - MispricingStrategy 📝
- `PHASE_4_GUIDE.md` - This file

---

**You're ready to start implementing!** The Portfolio class is complete and tested. Now build the engine loop and strategy logic.
