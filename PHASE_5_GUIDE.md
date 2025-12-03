# Phase 5: Strategy Comparison & Analysis

## Overview

**Goal**: Compare 4 volatility forecasting methods by backtesting a delta-neutral options mispricing strategy on SPY weekly options (2019 data).

**What You'll Build**:
1. Complete `src/backtest/strategy.py` implementation (3 methods)
2. Run 4 backtests (one per vol model)
3. Generate performance comparison
4. Create visualizations
5. Write findings report

**Time Estimate**: 6-8 hours
- Phase 4 completion: 2-3 hours
- Phase 5 implementation: 4-5 hours

---

## Phase 4: Complete `strategy.py` (DO THIS FIRST)

### File: `src/backtest/strategy.py`

You need to implement **3 core methods** in the `MispricingStrategy` class:

#### 1. `calculate_theo_price()` 

**Purpose**: Get vol forecast from model and price option with Black-Scholes.

**Steps**:
```python
def calculate_theo_price(self, S, K, T, r, option_type, current_date=None):
    # Step 1: Get volatility forecast from self.vol_model
    # Handle 4 different model types:
    
    if isinstance(self.vol_model, GARCHForecaster):
        # GARCH: forecast(horizon) where horizon = days to expiry
        horizon = int(T * 365)  # Convert years to days
        sigma = self.vol_model.forecast(horizon)
        
    elif isinstance(self.vol_model, MLForecaster):
        # ML: forecast(returns) with historical data up to current_date
        returns = self.returns_data  # Filter up to current_date if needed
        sigma = self.vol_model.forecast(returns)
        
    elif isinstance(self.vol_model, VolatilitySurface):
        # Surface: get_vol(strike, spot, dte, option_type)
        dte = int(T * 365)
        sigma = self.vol_model.get_vol(K, S, dte, option_type)
        
    else:
        # Simple float or callable (historical vol)
        if callable(self.vol_model):
            sigma = self.vol_model(S, K, T)
        else:
            sigma = self.vol_model
    
    # Step 2: Price with Black-Scholes
    theo_price = black_scholes_price(S, K, T, r, sigma, option_type)
    
    return theo_price
```

**Edge Cases to Handle**:
- `T = 0` (expiration day): Return intrinsic value
- Invalid vol (`sigma < 0` or `sigma > 2`): Skip option or use default vol
- Model errors: Catch exceptions, log warning, skip option

**Test It**:
```python
# Quick test in notebook
strategy = MispricingStrategy(vol_model=0.20)  # Simple float
price = strategy.calculate_theo_price(S=300, K=300, T=7/365, r=0.02, option_type='call')
print(f"Theo price: ${price:.2f}")  # Should be ~$3-4
```

---

#### 2. `generate_signals()`

**Purpose**: Find mispriced options and create TradeSignal objects.

**Steps**:
```python
def generate_signals(self, portfolio, options_snapshot, underlying_price, 
                    current_date, risk_free_rate):
    signals = []
    
    # Step 1: Check how many positions we can add
    option_positions = count_option_positions(portfolio)
    available_slots = self.max_positions - option_positions
    
    if available_slots <= 0:
        return signals  # Already at max positions
    
    # Step 2: Loop through options
    for idx, row in options_snapshot.iterrows():
        # Extract parameters
        K = row['strike']
        expiry = row['expiration']  # datetime
        option_type = row['call_put']  # 'call' or 'put'
        market_price = row['mid_price']
        dte = row['days_to_expiry']
        T = dte / 365.0
        
        # Skip if too close to expiry
        if dte < 2:
            continue
        
        # Calculate theo price
        try:
            theo_price = self.calculate_theo_price(
                S=underlying_price,
                K=K,
                T=T,
                r=risk_free_rate,
                option_type=option_type,
                current_date=current_date
            )
        except Exception as e:
            # Skip on error
            continue
        
        # Calculate mispricing
        if theo_price < 0.01:  # Skip if theo price too low
            continue
            
        mispricing_pct = (market_price - theo_price) / theo_price
        
        # Check if mispricing exceeds threshold
        if abs(mispricing_pct) > self.threshold:
            # Determine action
            if mispricing_pct > 0:
                action = 'sell'  # Market overpriced
                quantity = -self.contracts_per_trade
            else:
                action = 'buy'  # Market underpriced
                quantity = self.contracts_per_trade
            
            # Calculate delta for hedge
            greeks = calculate_all_greeks(
                S=underlying_price,
                K=K,
                T=T,
                r=risk_free_rate,
                sigma=theo_price_vol,  # Use forecasted vol
                option_type=option_type
            )
            delta = greeks['delta']
            
            # Hedge quantity = -delta * contracts * 100
            hedge_quantity = -delta * quantity * 100
            
            # Create signal
            signal = TradeSignal(
                symbol='SPY',
                action=action,
                quantity=quantity,
                market_price=market_price,
                theo_price=theo_price,
                mispricing_pct=mispricing_pct,
                strike=K,
                expiry=expiry,
                option_type=option_type,
                delta=delta,
                hedge_quantity=hedge_quantity,
                timestamp=current_date,
                underlying_price=underlying_price,
                dte=dte
            )
            
            signals.append(signal)
    
    # Step 3: Sort by mispricing magnitude
    signals.sort(key=lambda s: abs(s.mispricing_pct), reverse=True)
    
    # Step 4: Return top N signals
    return signals[:available_slots]
```

**Key Decisions**:
- **Mispricing calculation**: `(market - theo) / theo` (percentage)
- **Action logic**: 
  - `market > theo` → SELL (overpriced)
  - `market < theo` → BUY (underpriced)
- **Hedge sign**: Long call → Short stock (negative hedge quantity)
- **Filtering**: Skip low DTE, low price, or error-prone options

**Test It**:
```python
# In notebook, create mock snapshot
snapshot = pd.DataFrame({
    'strike': [300, 305, 310],
    'expiration': [pd.Timestamp('2019-06-21')] * 3,
    'call_put': ['call', 'call', 'call'],
    'mid_price': [5.0, 3.0, 1.5],
    'days_to_expiry': [7, 7, 7]
})

signals = strategy.generate_signals(
    portfolio=portfolio,
    options_snapshot=snapshot,
    underlying_price=300,
    current_date=pd.Timestamp('2019-06-14'),
    risk_free_rate=0.02
)

print(f"Generated {len(signals)} signals")
for s in signals:
    print(f"  {s.action} {s.option_type} ${s.strike} | Mispricing: {s.mispricing_pct:.2%}")
```

---

#### 3. `create_trade_from_signal()`

**Purpose**: Convert TradeSignal to Trade object for execution.

**Steps**:
```python
def create_trade_from_signal(self, signal, trade_type):
    if trade_type == 'option':
        # Create option trade
        trade = Trade(
            symbol=signal.symbol,
            quantity=signal.quantity,  # Already signed correctly
            price=signal.market_price,
            trade_type='option',
            timestamp=signal.timestamp,
            strike=signal.strike,
            expiry=signal.expiry,
            option_type=signal.option_type
        )
        
    elif trade_type == 'stock':
        # Create delta hedge trade
        trade = Trade(
            symbol=signal.symbol,
            quantity=signal.hedge_quantity,  # Already signed correctly
            price=signal.underlying_price,
            trade_type='stock',
            timestamp=signal.timestamp,
            strike=None,
            expiry=None,
            option_type=None
        )
    else:
        raise ValueError(f"Unknown trade_type: {trade_type}")
    
    return trade
```

**Test It**:
```python
# From previous signal
signal = signals[0]

option_trade = strategy.create_trade_from_signal(signal, 'option')
print(f"Option trade: {option_trade.quantity} contracts @ ${option_trade.price:.2f}")

hedge_trade = strategy.create_trade_from_signal(signal, 'stock')
print(f"Hedge trade: {hedge_trade.quantity:.0f} shares @ ${hedge_trade.price:.2f}")
```

---

### Helper Function: `count_option_positions()`

```python
def count_option_positions(portfolio) -> int:
    """Count number of option positions."""
    count = 0
    for pos_key, pos in portfolio.positions.items():
        if pos.position_type == 'option':
            count += 1
    return count
```

---

## Testing Phase 4

Before moving to Phase 5, test your strategy implementation:

### Test Script: `tests/test_strategy_manual.py`

```python
"""Manual test of strategy implementation."""

import sys
sys.path.append('..')

from src.backtest.engine import BacktestEngine, Portfolio
from src.backtest.strategy import MispricingStrategy
from datetime import datetime

# Test 1: Simple historical vol model
print("Test 1: Strategy with fixed vol")
strategy = MispricingStrategy(vol_model=0.20, threshold=0.10, max_positions=5)

# Test 2: Run mini-backtest (1 week)
print("\nTest 2: Mini backtest")
engine = BacktestEngine(
    start_date=datetime(2019, 6, 1),
    end_date=datetime(2019, 6, 30),
    initial_capital=100000
)

engine.load_data(
    options_path='../data/processed/spy_options.parquet',
    prices_path='../data/processed/spy_prices.parquet',
    rates_path='../data/processed/risk_free_rate.parquet'
)

engine.set_strategy(strategy)
results = engine.run()

print(f"\nResults shape: {results.shape}")
print(f"Final equity: ${results.iloc[-1]['equity']:,.2f}")
print(f"Trades executed: {len(engine.portfolio.trade_log)}")

# Test 3: Check positions
print(f"\nOpen positions: {len(engine.portfolio.positions)}")
for key, pos in engine.portfolio.positions.items():
    print(f"  {key}: {pos.quantity} @ ${pos.current_price:.2f}")

print("\n✓ Strategy tests complete!")
```

Run it:
```bash
cd tests
python test_strategy_manual.py
```

**Expected Output**:
- No errors
- Some trades executed
- Final equity near $100,000 (short test period)
- Some open positions at end

---

## Phase 5: Strategy Comparison

Once Phase 4 is complete, move to comparing models.

### File: `scripts/run_backtest_comparison.py`

Implement the TODOs in order:

### 1. Model Setup Functions

#### `setup_historical_model()`
```python
def setup_historical_model(prices, window=30):
    # Calculate returns
    returns = np.log(prices / prices.shift(1)).dropna()
    
    # Calculate rolling vol
    rolling_vol = returns.rolling(window=window).std() * np.sqrt(252)
    
    # Get latest vol
    latest_vol = rolling_vol.iloc[-1]
    
    # Return as simple float (strategy will use it directly)
    return latest_vol
```

#### `setup_garch_model()`
```python
def setup_garch_model(returns, horizon=7):
    garch = GARCHForecaster()
    
    # Use first 80% of data for initial fit
    train_size = int(len(returns) * 0.8)
    train_returns = returns.iloc[:train_size]
    
    garch.fit(train_returns)
    
    return garch
```

#### `setup_ml_model()`
```python
def setup_ml_model(returns, horizon=7):
    ml = MLForecaster(horizon=horizon)
    
    # Use first 80% of data for initial fit
    train_size = int(len(returns) * 0.8)
    train_returns = returns.iloc[:train_size]
    
    ml.fit(train_returns)
    
    return ml
```

#### `setup_surface_model()`
```python
def setup_surface_model(options_df):
    surface = VolatilitySurface()
    
    # Fit to options data
    # Need: strike, spot, dte, option_type, implied_vol
    surface.fit(options_df)
    
    return surface
```

---

### 2. Backtest Execution

#### `run_single_backtest()`
```python
def run_single_backtest(model_name, vol_model, returns_data=None):
    print(f"\nRunning {model_name}...")
    
    # Create engine
    engine = BacktestEngine(
        start_date=CONFIG['start_date'],
        end_date=CONFIG['end_date'],
        initial_capital=CONFIG['initial_capital']
    )
    
    # Load data
    engine.load_data(
        options_path=CONFIG['options_path'],
        prices_path=CONFIG['prices_path'],
        rates_path=CONFIG['rates_path']
    )
    
    # Create strategy
    strategy = MispricingStrategy(
        vol_model=vol_model,
        returns_data=returns_data,
        threshold=CONFIG['mispricing_threshold'],
        max_positions=CONFIG['max_positions'],
        contracts_per_trade=CONFIG['contracts_per_trade']
    )
    
    # Attach and run
    engine.set_strategy(strategy)
    results = engine.run()
    
    print(f"  Complete! Final equity: ${results.iloc[-1]['equity']:,.2f}")
    
    return results, engine
```

#### `run_all_backtests()`
```python
def run_all_backtests():
    # Load data
    prices = pd.read_parquet(CONFIG['prices_path'])
    options = pd.read_parquet(CONFIG['options_path'])
    
    # Calculate returns
    returns = np.log(prices['close'] / prices['close'].shift(1)).dropna()
    
    results = {}
    
    # 1. Historical
    hist_model = setup_historical_model(prices['close'])
    results['Historical'], _ = run_single_backtest('Historical', hist_model)
    
    # 2. Surface
    surf_model = setup_surface_model(options)
    results['Surface'], _ = run_single_backtest('Surface', surf_model)
    
    # 3. GARCH
    garch_model = setup_garch_model(returns)
    results['GARCH'], _ = run_single_backtest('GARCH', garch_model, returns)
    
    # 4. ML
    ml_model = setup_ml_model(returns)
    results['ML'], _ = run_single_backtest('ML', ml_model, returns)
    
    return results
```

---

### 3. Performance Metrics

#### `calculate_performance_metrics()`
```python
def calculate_performance_metrics(equity_curve, initial_capital=100000):
    # Calculate returns
    returns = equity_curve.pct_change().dropna()
    
    # Total return
    total_return = (equity_curve.iloc[-1] - initial_capital) / initial_capital
    
    # Annualized return (assuming 252 trading days)
    num_days = len(equity_curve)
    years = num_days / 252
    annualized_return = (1 + total_return) ** (1 / years) - 1
    
    # Sharpe ratio
    mean_daily_return = returns.mean()
    std_daily_return = returns.std()
    sharpe_ratio = (mean_daily_return / std_daily_return) * np.sqrt(252)
    
    # Max drawdown
    running_max = equity_curve.expanding().max()
    drawdown = (equity_curve - running_max) / running_max
    max_drawdown = drawdown.min()
    
    # Win rate
    win_rate = (returns > 0).sum() / len(returns)
    
    # Volatility
    volatility = std_daily_return * np.sqrt(252)
    
    # Calmar ratio
    calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
    
    return {
        'Total Return (%)': total_return * 100,
        'Annualized Return (%)': annualized_return * 100,
        'Sharpe Ratio': sharpe_ratio,
        'Max Drawdown (%)': max_drawdown * 100,
        'Win Rate (%)': win_rate * 100,
        'Volatility (%)': volatility * 100,
        'Calmar Ratio': calmar_ratio
    }
```

#### `create_metrics_comparison_table()`
```python
def create_metrics_comparison_table(all_results):
    metrics_list = []
    
    for model_name, results_df in all_results.items():
        equity = results_df['equity']
        metrics = calculate_performance_metrics(equity)
        metrics['Model'] = model_name
        metrics_list.append(metrics)
    
    # Create DataFrame
    df = pd.DataFrame(metrics_list)
    df = df.set_index('Model')
    
    # Sort by Sharpe ratio
    df = df.sort_values('Sharpe Ratio', ascending=False)
    
    return df
```

---

### 4. Visualization

#### `plot_equity_curves()`
```python
def plot_equity_curves(all_results, save_path=None):
    plt.figure(figsize=(12, 6))
    
    for model_name, results_df in all_results.items():
        plt.plot(results_df['date'], results_df['equity'], 
                label=model_name, linewidth=2)
    
    plt.axhline(y=100000, color='gray', linestyle='--', 
                alpha=0.5, label='Initial Capital')
    
    plt.title('Equity Curves: All Models', fontsize=14, fontweight='bold')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Equity ($)')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
```

#### `plot_drawdown_comparison()` (OPTIONAL)
```python
def plot_drawdown_comparison(all_results, save_path=None):
    plt.figure(figsize=(12, 6))
    
    for model_name, results_df in all_results.items():
        equity = results_df['equity']
        running_max = equity.expanding().max()
        drawdown = (equity - running_max) / running_max * 100
        
        plt.plot(results_df['date'], drawdown, label=model_name, linewidth=2)
    
    plt.title('Drawdown Comparison', fontsize=14, fontweight='bold')
    plt.xlabel('Date')
    plt.ylabel('Drawdown (%)')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
    
    plt.show()
```

---

### 5. Main Execution

#### `main()`
```python
def main():
    print("="*80)
    print("PHASE 5: STRATEGY COMPARISON")
    print("="*80)
    
    # Create output directory
    os.makedirs(CONFIG['results_dir'], exist_ok=True)
    
    # Run backtests
    print("\nStep 1: Running backtests...")
    all_results = run_all_backtests()
    
    # Calculate metrics
    print("\nStep 2: Calculating metrics...")
    metrics_df = create_metrics_comparison_table(all_results)
    print("\n" + "="*60)
    print("PERFORMANCE COMPARISON")
    print("="*60)
    print(metrics_df.round(2))
    
    # Generate plots
    print("\nStep 3: Generating visualizations...")
    plot_equity_curves(all_results, 
                      save_path=f"{CONFIG['results_dir']}/equity_curves.png")
    plot_drawdown_comparison(all_results,
                            save_path=f"{CONFIG['results_dir']}/drawdowns.png")
    
    # Save results
    print("\nStep 4: Saving results...")
    save_results_to_csv(all_results, metrics_df, CONFIG['results_dir'])
    
    # Print winner
    winner = metrics_df.index[0]
    print("\n" + "="*80)
    print(f"🏆 WINNER: {winner} (Best Sharpe Ratio: {metrics_df.iloc[0]['Sharpe Ratio']:.2f})")
    print("="*80)
```

---

## Running Phase 5

```bash
# From project root
python scripts/run_backtest_comparison.py
```

**Expected Runtime**: 5-15 minutes (depends on data size and models)

**Expected Output**:
```
================================================================================
PHASE 5: STRATEGY COMPARISON
================================================================================

Step 1: Running backtests...
Running Historical...
  Complete! Final equity: $102,345.67

Running Surface...
  Complete! Final equity: $98,765.43

Running GARCH...
  Complete! Final equity: $105,678.90

Running ML...
  Complete! Final equity: $103,456.78

Step 2: Calculating metrics...
============================================================
PERFORMANCE COMPARISON
============================================================
              Total Return (%)  Annualized Return (%)  Sharpe Ratio  ...
Model                                                                ...
GARCH                    5.68                   5.82          0.85  ...
ML                       3.46                   3.55          0.62  ...
Historical               2.35                   2.40          0.45  ...
Surface                 -1.23                  -1.25         -0.23  ...

Step 3: Generating visualizations...
[Plots displayed]

Step 4: Saving results...
Saved: results/phase5/metrics_comparison.csv
Saved: results/phase5/equity_curves.csv

================================================================================
🏆 WINNER: GARCH (Best Sharpe Ratio: 0.85)
================================================================================
```

---

## Analysis & Findings

After running Phase 5, answer these questions:

### 1. Model Performance
- Which model had the best Sharpe ratio?
- Which had the worst max drawdown?
- Did any model consistently outperform others?

### 2. Economic Intuition
- Why did GARCH/ML perform better/worse than surface?
- What does it mean if Surface model loses money? (Variance risk premium)
- How do transaction costs affect results?

### 3. Strategy Behavior
- How many trades per week on average?
- What was the average mispricing captured?
- Did delta hedging work (check position deltas)?

### 4. Robustness
- Are results stable or sensitive to parameters?
- What if you change `mispricing_threshold` to 0.05 or 0.15?
- What if you change `max_positions` to 3 or 10?

---

## Deliverables Checklist

### Phase 4 ✅
- [ ] `calculate_theo_price()` implemented
- [ ] `generate_signals()` implemented
- [ ] `create_trade_from_signal()` implemented
- [ ] Manual test passes (1-week backtest runs)
- [ ] No errors when running mini-backtest

### Phase 5 ✅
- [ ] All 4 model setup functions implemented
- [ ] `run_single_backtest()` working
- [ ] `run_all_backtests()` completes successfully
- [ ] Performance metrics calculated
- [ ] Equity curves plotted
- [ ] Results saved to CSV
- [ ] Comparison table generated

### Analysis ✅
- [ ] Metrics comparison table reviewed
- [ ] Winner identified (best Sharpe)
- [ ] Findings documented
- [ ] Interview questions prepared

---

## Interview Prep: Key Talking Points

### Project Overview (2 min)
"I built an options trading strategy comparison framework to evaluate different volatility forecasting methods. The strategy trades SPY weekly options based on mispricing between theoretical and market prices, maintaining delta-neutrality through stock hedging. I compared 4 vol models: historical baseline, volatility surface, GARCH(1,1), and XGBoost ML."

### Technical Depth (5 min)
- **Strategy**: "Delta-neutral mispricing arbitrage. If my model predicts vol=20% but market prices at 25%, I sell the option and hedge with stock. PnL comes from vol forecast accuracy, not directional bets."

- **Models**: "GARCH captures vol clustering and mean reversion—standard in industry. XGBoost learns non-linear patterns from multiple features like recent returns, realized vol at different horizons, and higher moments. Surface is the market benchmark—what is implied vol pricing?"

- **Implementation**: "Built end-to-end: data pipeline, Black-Scholes pricer with Greeks, vol forecasting models, backtesting engine with transaction costs and daily rebalancing. ~2,000 lines of Python, 45 unit tests."

### Results (3 min)
"[Your results here] GARCH outperformed with Sharpe ratio of 0.85. ML was close behind at 0.62. Surface model actually lost money, which is interesting—suggests variance risk premium exists (implied vol > realized vol). Historical vol was the weakest baseline."

### Challenges (2 min)
- "Walk-forward validation: Had to refit models weekly with expanding window to avoid lookahead bias."
- "Transaction costs matter: Options are expensive ($0.50/contract). Had to filter for significant mispricings (>10%)."
- "Delta hedging: Daily rebalancing added costs but maintained neutrality. Less frequent = more delta risk."

### Extensions (2 min)
- "Ensemble model: Combine GARCH + ML forecasts"
- "Dynamic position sizing: Scale by forecast confidence"
- "Bid-ask spread modeling: Current backtest assumes mid-price fills"
- "Alternative strategies: Volatility carry, dispersion trading"

---

## Common Errors & Solutions

### Error 1: "Module not found: src.backtest"
**Solution**: Run from project root, not `scripts/` dir
```bash
cd /home/oalkhadra/options-trading
python scripts/run_backtest_comparison.py
```

### Error 2: "calculate_theo_price() not implemented"
**Solution**: Check you've removed `raise NotImplementedError()` and added actual code

### Error 3: "Index out of bounds in forecast()"
**Solution**: Check that returns data has enough history (need 60+ days for ML)

### Error 4: "Negative option prices"
**Solution**: Check vol forecast is reasonable (0.05 < vol < 2.0). Add bounds checking.

### Error 5: "No signals generated"
**Solution**: 
- Check mispricing threshold isn't too high
- Verify vol model is returning sensible values
- Add debug prints in `generate_signals()`

---

## Next Steps (Post-Phase 5)

1. **Parameter Optimization**
   - Grid search: threshold, max_positions, contracts_per_trade
   - Find optimal combination

2. **Robustness Testing**
   - Test on 2020 data (COVID volatility)
   - Different underlyings (QQQ, IWM)
   - Different option maturities (monthly vs weekly)

3. **Advanced Features**
   - Vega-neutral portfolio (hedge gamma too)
   - Dynamic rebalancing frequency
   - Risk-based position sizing

4. **Live Trading Preparation**
   - Real-time data feed integration
   - Order management system
   - Risk monitoring dashboard

---

## Resources

### Documentation
- `PHASE_4_GUIDE.md`: Backtest engine details
- `README.md`: Project overview
- `src/` docstrings: Function-level docs

### Code References
- `src/pricing/pricer.py`: Black-Scholes implementation
- `src/pricing/greeks.py`: Delta, gamma, vega calculations
- `src/volatility/`: All vol models
- `src/backtest/engine.py`: Portfolio and execution

### Theory
- Hull's "Options, Futures, and Other Derivatives" - Black-Scholes, Greeks
- Tsay's "Analysis of Financial Time Series" - GARCH models
- Gatheral's "The Volatility Surface" - Implied vol dynamics

---

## Final Checklist

Before calling Phase 5 complete:

- [ ] All functions implemented (no `NotImplementedError`)
- [ ] Full backtest runs without errors
- [ ] Results saved to `results/phase5/`
- [ ] Equity curve plot looks reasonable
- [ ] Metrics table makes sense (Sharpe ratios in [-1, 2] range typically)
- [ ] Winner identified and can explain why
- [ ] README updated with findings
- [ ] Ready to present project in interview

**Good luck! 🚀**

---

## Questions?

If you get stuck:

1. Check error message carefully
2. Add debug prints to narrow down issue
3. Test each function individually before full backtest
4. Review `engine.py` to understand data flow
5. Check that data files exist and have expected schema

Remember: The goal is learning and demonstrating skills. If something doesn't work perfectly, that's okay—being able to explain challenges and solutions is valuable too.

