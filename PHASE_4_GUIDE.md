# Phase 4: Backtesting Engine - Mispricing Strategy

## 🎯 Your Strategy (Clarified)

You are implementing a **Relative Value / Statistical Arbitrage** strategy:

1. **Weekly Rebalancing** (Sunday): Review all available options
2. **Theoretical Pricing**: Calculate "fair" price using your vol model (GARCH/ML/Historical/Surface)
3. **Mispricing Detection**: Compare theo price vs. market price
   - If `market_price > theo_price * (1 + threshold)` → **SELL** (overpriced)
   - If `market_price < theo_price * (1 - threshold)` → **BUY** (underpriced)
4. **Delta Hedging**: For each option position, trade stock to neutralize delta
5. **Hold to Expiry**: No intra-week rebalancing
6. **Settlement**:
   - ITM options: Realize intrinsic value
   - OTM options: Expire worthless

## 🏗 Architecture

### Key Components

1. **`Portfolio`**: 
   - Tracks cash and positions (options + stock hedge)
   - Handles trade execution
   - Manages option expiries

2. **`MispricingStrategy`**:
   - Takes a vol model as input
   - Calculates theo prices
   - Generates buy/sell signals based on mispricing
   - Generates delta-hedge trades

3. **`BacktestEngine`**:
   - Steps through weeks
   - Handles expiries
   - Executes strategy signals
   - Logs PnL

### Position Keying

Options need unique identifiers:
```python
"SPY_290C_20190621"  # SPY, $290 Call, June 21 2019
```

This prevents collisions between different strikes/expiries.

## 📊 Phase 5 Preview

You'll run **4 separate backtests**:
1. Strategy B: Historical Vol model
2. Strategy C: Vol Surface model
3. Strategy D (GARCH): GARCH forecasting
4. Strategy D (ML): XGBoost forecasting

Each produces an equity curve. Phase 5 compares them.

## 🛠 Implementation Tasks

### 1. `Portfolio` Class
- [x] `execute_trade()` - Process trades and update positions
- [ ] `mark_to_market()` - Update position values
- [ ] `handle_expiries()` - Settle expired options

### 2. `MispricingStrategy` Class
- [ ] `calculate_theo_price()` - Get vol from model, price with BS
- [ ] `generate_signals()` - Find mispricings, create signals

### 3. `BacktestEngine` Class
- [ ] `load_data()` - Load options and prices
- [ ] `run()` - Main weekly loop

## 💡 Key Concepts

### Delta Hedging Example
- Sell 1 SPY $290 Call (delta = 0.60)
- Hedge: Buy 60 shares of SPY
- Net delta ≈ 0 (market-neutral)

### Expiry PnL
**Long Call Example:**
- Buy 1 SPY $290 Call @ $5.00 (cost = $500)
- At expiry, SPY = $295
- Intrinsic = max(0, 295 - 290) = $5
- Settlement: Receive $500
- Net PnL = $0 (breakeven)

**Short Put Example:**
- Sell 1 SPY $290 Put @ $3.00 (credit = $300)
- At expiry, SPY = $285
- Intrinsic = max(0, 290 - 285) = $5
- Settlement: Pay $500
- Net PnL = -$200 (loss)

## 📂 File Structure

```
src/backtest/
├── __init__.py
├── engine.py       # Portfolio, BacktestEngine
└── strategy.py     # MispricingStrategy
```

## 🚀 Next Steps

Start implementing the TODOs in order:
1. `Portfolio.execute_trade()` ✅ (Done)
2. `Portfolio.mark_to_market()`
3. `Portfolio.handle_expiries()`
4. `MispricingStrategy.calculate_theo_price()`
5. `MispricingStrategy.generate_signals()`
6. `BacktestEngine.load_data()`
7. `BacktestEngine.run()`

