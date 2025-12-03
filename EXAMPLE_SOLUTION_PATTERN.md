# Example Solution Patterns

> **Note**: These are EXAMPLE patterns to help you understand the approach.
> Do NOT copy-paste blindly. Understand the logic and adapt to your specific needs.

---

## Pattern 1: `calculate_theo_price()`

### Example Implementation

```python
def calculate_theo_price(self, S, K, T, r, option_type, current_date=None):
    """Calculate theoretical option price using vol model."""
    
    # Edge case: Already expired
    if T <= 0:
        return intrinsic_value(S, K, option_type)
    
    # Step 1: Get volatility forecast from model
    # Handle different model types
    
    if isinstance(self.vol_model, GARCHForecaster):
        # GARCH model: forecast(horizon in days)
        horizon = max(1, int(T * 365))  # Convert years to days
        sigma = self.vol_model.forecast(horizon=horizon)
        
    elif isinstance(self.vol_model, MLForecaster):
        # ML model: forecast(returns series)
        # Use historical returns up to current date
        if self.returns_data is None:
            raise ValueError("ML model requires returns_data")
        
        # Filter returns up to current_date if provided
        if current_date is not None:
            returns = self.returns_data[self.returns_data.index < current_date]
        else:
            returns = self.returns_data
            
        sigma = self.vol_model.forecast(returns)
        
    elif isinstance(self.vol_model, VolatilitySurface):
        # Surface model: get_vol(strike, spot, dte, option_type)
        dte = max(1, int(T * 365))
        sigma = self.vol_model.get_vol(
            strike=K,
            spot=S,
            dte=dte,
            option_type=option_type
        )
        
    else:
        # Simple vol: float or callable
        if callable(self.vol_model):
            sigma = self.vol_model(S, K, T)
        else:
            sigma = float(self.vol_model)
    
    # Step 2: Validate volatility
    # Clip to reasonable range to avoid pricing errors
    sigma = max(0.01, min(2.0, sigma))  # 1% to 200%
    
    # Step 3: Calculate theoretical price using Black-Scholes
    theo_price = black_scholes_price(
        S=S,
        K=K,
        T=T,
        r=r,
        sigma=sigma,
        option_type=option_type
    )
    
    return theo_price
```

### Key Points

1. **Type checking**: Use `isinstance()` to determine model type
2. **Edge cases**: Handle T=0, invalid sigma, missing data
3. **Data filtering**: For ML/GARCH, may need to filter returns to current_date
4. **Validation**: Clip sigma to reasonable range (avoid negative or extreme values)
5. **Consistent interface**: All models return annualized volatility

---

## Pattern 2: `generate_signals()` - Partial Example

```python
def generate_signals(self, portfolio, options_snapshot, underlying_price, 
                    current_date, risk_free_rate):
    """Generate trading signals based on mispricing."""
    
    signals = []
    
    # Step 1: Check current position count
    option_positions = count_option_positions(portfolio)
    available_slots = self.max_positions - option_positions
    
    if available_slots <= 0:
        return signals  # At capacity
    
    # Step 2: Loop through available options
    for idx, row in options_snapshot.iterrows():
        # Extract option parameters
        K = float(row['strike'])
        expiry = pd.to_datetime(row['expiration'])
        option_type = str(row['call_put'])  # 'call' or 'put'
        market_price = float(row['mid_price'])
        dte = int(row['days_to_expiry'])
        T = dte / 365.0
        
        # Filter: Skip if too close to expiry
        if dte < 2:
            continue
        
        # Filter: Skip if market price too low (illiquid)
        if market_price < 0.10:
            continue
        
        # Step 3: Calculate theoretical price
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
            # Skip on error (log in production)
            continue
        
        # Validate theo price
        if theo_price <= 0.01:
            continue
        
        # Step 4: Calculate mispricing
        mispricing_pct = (market_price - theo_price) / theo_price
        
        # Check if exceeds threshold
        if abs(mispricing_pct) > self.threshold:
            # Step 5: Determine trade direction
            if mispricing_pct > 0:
                action = 'sell'  # Market overpriced
                quantity = -self.contracts_per_trade
            else:
                action = 'buy'  # Market underpriced
                quantity = self.contracts_per_trade
            
            # Step 6: Calculate delta for hedging
            # Need to recalculate with the vol we used for pricing
            # (Store this or recalculate - here we recalculate)
            
            # Get the vol we used
            sigma = self._get_vol_for_pricing(S=underlying_price, K=K, T=T, 
                                              option_type=option_type, 
                                              current_date=current_date)
            
            greeks = calculate_all_greeks(
                S=underlying_price,
                K=K,
                T=T,
                r=risk_free_rate,
                sigma=sigma,
                option_type=option_type
            )
            
            delta = greeks['delta']
            
            # Calculate hedge quantity
            # Long option (+delta) → Short stock (-delta shares)
            # Short option (-delta) → Long stock (+delta shares)
            # Multiply by 100 (shares per contract)
            hedge_quantity = -delta * quantity * 100
            
            # Step 7: Create signal
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
    
    # Step 8: Sort by mispricing magnitude (best opportunities first)
    signals.sort(key=lambda s: abs(s.mispricing_pct), reverse=True)
    
    # Step 9: Return top N signals (limited by available slots)
    return signals[:available_slots]


def _get_vol_for_pricing(self, S, K, T, option_type, current_date):
    """Helper to get vol (same logic as calculate_theo_price but just returns vol)."""
    if isinstance(self.vol_model, GARCHForecaster):
        return self.vol_model.forecast(horizon=int(T*365))
    elif isinstance(self.vol_model, MLForecaster):
        returns = self.returns_data
        if current_date is not None:
            returns = returns[returns.index < current_date]
        return self.vol_model.forecast(returns)
    elif isinstance(self.vol_model, VolatilitySurface):
        return self.vol_model.get_vol(K, S, int(T*365), option_type)
    else:
        return float(self.vol_model) if not callable(self.vol_model) else self.vol_model(S, K, T)
```

### Key Points

1. **Position management**: Check current positions before generating new signals
2. **Filtering**: Skip low DTE, low price, or error-prone options
3. **Error handling**: Use try/except to skip problematic options
4. **Delta calculation**: Need same vol used for pricing to get accurate delta
5. **Hedge sign**: Opposite of delta exposure (negative of delta * quantity * 100)
6. **Sorting**: Best opportunities first (largest |mispricing|)
7. **Limiting**: Respect max_positions constraint

---

## Pattern 3: `create_trade_from_signal()`

### Example Implementation

```python
def create_trade_from_signal(self, signal, trade_type):
    """Convert TradeSignal to Trade object."""
    
    if trade_type == 'option':
        # Create option trade
        trade = Trade(
            symbol=signal.symbol,
            quantity=signal.quantity,  # Already signed correctly in signal
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
        raise ValueError(f"Unknown trade_type: {trade_type}. Must be 'option' or 'stock'")
    
    return trade
```

### Key Points

1. **Simple conversion**: TradeSignal → Trade
2. **Two trade types**: Option and stock (hedge)
3. **Sign preserved**: Quantities already signed correctly in signal
4. **None for stock**: Stock trades don't have strike/expiry/option_type

---

## Pattern 4: Helper Function

### Example: `count_option_positions()`

```python
def count_option_positions(portfolio) -> int:
    """
    Count number of option positions in portfolio.
    
    Args:
        portfolio: Portfolio object
        
    Returns:
        Number of option positions
    """
    count = 0
    for pos_key, pos in portfolio.positions.items():
        if pos.position_type == 'option':
            count += 1
    return count
```

### Alternative (More Pythonic)

```python
def count_option_positions(portfolio) -> int:
    """Count number of option positions in portfolio."""
    return sum(1 for pos in portfolio.positions.values() 
               if pos.position_type == 'option')
```

---

## Pattern 5: Model Setup (Phase 5)

### Example: `setup_garch_model()`

```python
def setup_garch_model(returns, horizon=7):
    """Setup GARCH forecasting model."""
    
    # Create model instance
    garch = GARCHForecaster()
    
    # Fit on historical returns
    # Note: In walk-forward backtest, you'll refit weekly with expanding window
    # Here we just do initial fit
    garch.fit(returns)
    
    return garch
```

### Example: `setup_historical_model()`

```python
def setup_historical_model(prices, window=30):
    """Setup historical volatility model."""
    
    # Calculate returns
    returns = np.log(prices / prices.shift(1)).dropna()
    
    # Calculate rolling vol (annualized)
    rolling_vol = returns.rolling(window=window).std() * np.sqrt(252)
    
    # Get latest vol
    latest_vol = rolling_vol.iloc[-1]
    
    # Return as simple float
    # Strategy will use this constant vol for all options
    return float(latest_vol)
```

---

## Pattern 6: Performance Metrics

### Example: Sharpe Ratio

```python
def calculate_sharpe_ratio(equity_curve):
    """Calculate Sharpe ratio from equity curve."""
    
    # Calculate daily returns
    returns = equity_curve.pct_change().dropna()
    
    # Mean and std of daily returns
    mean_daily_return = returns.mean()
    std_daily_return = returns.std()
    
    # Annualize (assuming 252 trading days)
    sharpe_ratio = (mean_daily_return / std_daily_return) * np.sqrt(252)
    
    return sharpe_ratio
```

### Example: Max Drawdown

```python
def calculate_max_drawdown(equity_curve):
    """Calculate maximum drawdown from equity curve."""
    
    # Calculate running maximum
    running_max = equity_curve.expanding().max()
    
    # Calculate drawdown at each point
    drawdown = (equity_curve - running_max) / running_max
    
    # Maximum drawdown (most negative value)
    max_drawdown = drawdown.min()
    
    return max_drawdown
```

---

## Pattern 7: Backtesting

### Example: `run_single_backtest()`

```python
def run_single_backtest(model_name, vol_model, returns_data=None):
    """Run backtest for a single volatility model."""
    
    print(f"\nRunning {model_name} backtest...")
    
    # Step 1: Create engine
    engine = BacktestEngine(
        start_date=CONFIG['start_date'],
        end_date=CONFIG['end_date'],
        initial_capital=CONFIG['initial_capital']
    )
    
    # Step 2: Load data
    engine.load_data(
        options_path=CONFIG['options_path'],
        prices_path=CONFIG['prices_path'],
        rates_path=CONFIG['rates_path']
    )
    
    # Step 3: Create strategy
    strategy = MispricingStrategy(
        vol_model=vol_model,
        returns_data=returns_data,
        threshold=CONFIG['mispricing_threshold'],
        max_positions=CONFIG['max_positions'],
        contracts_per_trade=CONFIG['contracts_per_trade']
    )
    
    # Step 4: Attach strategy
    engine.set_strategy(strategy)
    
    # Step 5: Run backtest
    results = engine.run()
    
    # Step 6: Print summary
    final_equity = results.iloc[-1]['equity']
    total_return = (final_equity / CONFIG['initial_capital'] - 1) * 100
    num_trades = len(engine.portfolio.trade_log)
    
    print(f"  Final equity: ${final_equity:,.2f}")
    print(f"  Total return: {total_return:.2f}%")
    print(f"  Trades: {num_trades}")
    
    return results, engine
```

---

## Common Pitfalls & Solutions

### Pitfall 1: Wrong Delta Sign

❌ **Wrong**:
```python
hedge_quantity = delta * quantity * 100  # Wrong sign!
```

✅ **Correct**:
```python
hedge_quantity = -delta * quantity * 100  # Opposite of option delta
```

**Explanation**: If you're long a call (positive delta), you need to short stock (negative quantity) to be delta-neutral.

---

### Pitfall 2: Forgetting to Annualize Vol

❌ **Wrong**:
```python
sigma = returns.std()  # Daily vol, not annualized!
```

✅ **Correct**:
```python
sigma = returns.std() * np.sqrt(252)  # Annualized vol
```

**Explanation**: Black-Scholes expects annualized volatility.

---

### Pitfall 3: Not Filtering Bad Options

❌ **Wrong**:
```python
for row in options_snapshot.iterrows():
    theo_price = calculate_theo_price(...)  # Might error on bad data
```

✅ **Correct**:
```python
for row in options_snapshot.iterrows():
    if dte < 2 or market_price < 0.10:
        continue
    try:
        theo_price = calculate_theo_price(...)
    except Exception as e:
        continue  # Skip problematic options
```

**Explanation**: Real data has edge cases. Filter and handle errors gracefully.

---

### Pitfall 4: Lookahead Bias

❌ **Wrong**:
```python
# Using ALL returns data including future
ml_model.fit(all_returns)
```

✅ **Correct**:
```python
# Only use returns UP TO current date
if current_date is not None:
    train_returns = all_returns[all_returns.index < current_date]
    ml_model.fit(train_returns)
```

**Explanation**: In backtesting, you can't use future data. Walk-forward validation is critical.

---

## Testing Pattern

### Unit Test Pattern

```python
def test_function():
    """Test a specific function."""
    
    # Setup
    input_data = create_mock_data()
    expected_output = known_correct_result
    
    # Execute
    try:
        actual_output = function_to_test(input_data)
        
        # Verify
        if abs(actual_output - expected_output) < tolerance:
            print("✅ PASS")
            return True
        else:
            print(f"❌ FAIL: Expected {expected_output}, got {actual_output}")
            return False
            
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False
```

---

## Debugging Pattern

### Add Strategic Print Statements

```python
def generate_signals(self, portfolio, options_snapshot, ...):
    print(f"\n=== SIGNAL GENERATION ===")
    print(f"Options available: {len(options_snapshot)}")
    
    option_positions = count_option_positions(portfolio)
    print(f"Current positions: {option_positions}/{self.max_positions}")
    
    signals = []
    
    for idx, row in options_snapshot.iterrows():
        # ...
        try:
            theo_price = self.calculate_theo_price(...)
            print(f"  Option {idx}: K={K}, theo=${theo_price:.2f}, "
                  f"market=${market_price:.2f}, misprice={mispricing_pct:.2%}")
        except Exception as e:
            print(f"  Option {idx}: ERROR - {e}")
            continue
    
    print(f"Signals generated: {len(signals)}")
    return signals
```

**Tip**: Remove debug prints for production, or use logging module.

---

## Best Practices

1. **Type hints**: Use them everywhere for clarity
   ```python
   def calculate_theo_price(self, S: float, K: float, T: float, 
                           r: float, option_type: str) -> float:
   ```

2. **Docstrings**: Explain what, not how (code shows how)
   ```python
   """Calculate theoretical option price using vol model."""
   ```

3. **Error handling**: Graceful degradation
   ```python
   try:
       result = risky_operation()
   except Exception as e:
       logging.warning(f"Operation failed: {e}")
       return default_value
   ```

4. **Constants**: Use CONFIG dict or class constants
   ```python
   ANNUAL_TRADING_DAYS = 252
   MIN_DTE = 2
   MAX_VOLATILITY = 2.0
   ```

5. **Validation**: Check inputs
   ```python
   if T < 0:
       raise ValueError(f"Time to expiry cannot be negative: {T}")
   ```

---

## Interview Talking Points

### Discuss Design Decisions

**Q**: "Why did you choose delta-neutral strategy?"

**A**: "I wanted to isolate the volatility bet from directional market exposure. Delta-neutral means PnL comes purely from forecast accuracy, not from whether SPY goes up or down. This is key for comparing vol models fairly—each model is evaluated on its ability to predict volatility, not direction."

---

**Q**: "How did you handle transaction costs?"

**A**: "I modeled realistic costs: $0.50 per option contract and 1 basis point for stock. This is critical because options are expensive to trade. I set a 10% mispricing threshold—this ensures each trade has enough expected profit to cover costs plus provide risk premium."

---

**Q**: "Walk me through your signal generation logic."

**A**: "First, I check if we're at max positions (risk management). Then for each available option, I forecast volatility using the model, price it with Black-Scholes to get theo price, and compare to market. If mispricing exceeds threshold, I determine direction—sell if overpriced, buy if underpriced—calculate delta for hedging, and create a signal. Finally, I sort by mispricing magnitude and take the top N opportunities."

---

**Remember**: These are PATTERNS, not complete solutions. Adapt them to your specific implementation! Good luck! 🚀

