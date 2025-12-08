# Options Trading Backtest Framework

A production-grade backtesting system for volatility-based options mispricing strategies with daily delta hedging and multiple volatility forecasting models.

## 📊 Project Overview

This project implements and compares different volatility forecasting models for identifying mispriced options on SPY. The system executes delta-neutral trades to isolate pure volatility exposure from directional market movements.

### Key Results (2023 Backtest)

| Model | Total Return | Sharpe Ratio | Max Drawdown | Win Rate |
|-------|-------------|--------------|--------------|----------|
| **GARCH(1,1)** | **12.7%** | **0.94** | **-8.9%** | **52.3%** |
| Historical Vol | 4.8% | 0.30 | -17.0% | 50.8% |
| Vol Surface | -5.0% | -0.49 | -11.2% | 45.4% |

**Key Finding:** GARCH(1,1) significantly outperforms baseline methods by better predicting short-term volatility dynamics.

---

## 🏗️ Architecture

```
options-trading/
│
├── data/
│   ├── raw/options/              # Dolt database with historical options data
│   └── processed/                # Parquet files (options, prices, rates)
│
├── src/
│   ├── pricing/                  # Black-Scholes pricing and Greeks
│   │   ├── pricer.py            # Option pricing functions
│   │   ├── greeks.py            # Delta, gamma, theta, vega, rho
│   │   └── risk_free_rate.py    # Treasury yield curve data
│   │
│   ├── volatility/               # Volatility forecasting models
│   │   ├── historical.py        # Rolling historical volatility (baseline)
│   │   ├── surface.py           # Volatility surface interpolation
│   │   └── forecasting.py       # GARCH(1,1) and XGBoost ML models
│   │
│   ├── backtest/                 # Core backtesting engine
│   │   ├── engine.py            # Portfolio management and event loop
│   │   ├── strategy.py          # Mispricing strategy logic
│   │   └── main.py              # Strategy comparison runner
│   │
│   └── preprocess.py             # Data extraction from Dolt database
│
├── notebooks/
│   └── exploration.ipynb         # Analysis and visualization
│
├── results/                      # Backtest outputs (equity curves, logs, metrics)
└── tests/                        # Unit tests
```

---

## 🔧 Core Components

### 1. Data Pipeline (`preprocess.py`)

Queries the [Options Price Reporting Authority (OPRA)](https://github.com/dolthub/options-prices) Dolt database and prepares data for backtesting.

**Data Sources:**
- Options chain data (bid, ask, IV, Greeks) from CBOE
- Underlying prices (OHLCV) from Yahoo Finance
- Risk-free rates from FRED (Treasury yields)

**Processing:**
- Filters for SPY options with sufficient liquidity
- Calculates mid prices and days to expiry
- Merges with underlying prices and interest rates
- Saves to efficient Parquet format

**Usage:**
```python
python src/preprocess.py --ticker SPY --start 2019-01-01 --end 2024-12-31
```

---

### 2. Pricing Engine (`pricing/`)

#### Black-Scholes Pricer (`pricer.py`)
Implements the classic Black-Scholes-Merton model:

```
C = S·N(d₁) - K·e^(-rT)·N(d₂)
P = K·e^(-rT)·N(-d₂) - S·N(-d₁)

where:
  d₁ = [ln(S/K) + (r + σ²/2)T] / (σ√T)
  d₂ = d₁ - σ√T
```

**Features:**
- European option pricing (calls and puts)
- Intrinsic value calculation
- Moneyness classification (ITM/ATM/OTM)
- Vectorized operations for performance

#### Greeks Calculator (`greeks.py`)
Calculates option sensitivities for risk management:

| Greek | Measures | Formula | Use Case |
|-------|----------|---------|----------|
| **Delta (Δ)** | Price sensitivity to underlying | ∂V/∂S | Delta hedging |
| **Gamma (Γ)** | Rate of delta change | ∂²V/∂S² | Hedge stability |
| **Theta (Θ)** | Time decay | ∂V/∂t | Holding cost |
| **Vega (ν)** | Volatility sensitivity | ∂V/∂σ | Vol exposure |
| **Rho (ρ)** | Interest rate sensitivity | ∂V/∂r | Rate risk |

**Additional Feature:** Implied volatility solver using Newton-Raphson and Brent methods.

---

### 3. Volatility Models (`volatility/`)

The core differentiator: Which volatility forecast produces the best theoretical prices?

#### Model 1: Historical Volatility (`historical.py`)
**Baseline:** Rolling window standard deviation of log returns.

```python
σₜ = √(252 × Var(log returns over last N days))
```

**Pros:** Simple, interpretable, no parameters  
**Cons:** Backward-looking, assumes constant volatility

#### Model 2: Volatility Surface (`surface.py`)
**Market Benchmark:** Interpolates implied volatilities from traded options.

**Approach:**
1. Extract IVs from liquid options across strikes and maturities
2. Build 2D surface: IV(K, T)
3. Interpolate for any strike/expiry combination

**Pros:** Market's actual volatility expectations  
**Cons:** Requires liquid options data, can be noisy

#### Model 3: GARCH(1,1) (`forecasting.py`)
**Statistical Forecast:** Generalized Autoregressive Conditional Heteroskedasticity.

```python
σₜ² = ω + α·rₑₜ²ₜ₋₁ + β·σₜ₋₁²

where:
  - ω: long-term variance
  - α: reaction to shocks
  - β: persistence
```

**Pros:** Captures volatility clustering and mean reversion  
**Cons:** Assumes parametric structure

**Implementation:** `arch` package with walk-forward estimation

#### Model 4: XGBoost ML (`forecasting.py`)
**Machine Learning:** Gradient-boosted trees for volatility prediction.

**Features:**
- Lagged returns (1, 5, 21 days)
- Rolling volatilities (5, 21, 63 days)
- Historical volume patterns
- Day-of-week effects

**Pros:** Non-parametric, captures complex patterns  
**Cons:** Overfitting risk, requires more data

---

### 4. Backtesting Engine (`backtest/engine.py`)

Event-driven simulation with realistic transaction costs and daily delta rebalancing.

#### Portfolio Management
**Tracks:**
- Cash balance
- Open positions (options + stock hedges)
- Greeks exposure (delta, gamma, vega)
- Equity curve over time

**Position Tracking:**
```python
@dataclass
class Position:
    symbol: str
    quantity: float          # +Long, -Short
    entry_price: float
    current_price: float
    position_type: str       # 'option' or 'stock'
    strike: Optional[float]
    expiry: Optional[datetime]
    option_type: Optional[str]
    implied_vol: Optional[float]  # For daily rebalancing
```

#### Transaction Costs
- **Options:** $0.50 per contract (realistic for retail/small funds)
- **Stock:** 1 basis point (0.01%) for hedging trades

#### Event Loop
```
FOR each option trading day:
    1. Handle expired options → settle intrinsic value
    2. Update volatility model with latest data (walk-forward)
    3. Get options snapshot for this day
    4. Generate signals (mispricing > threshold?)
    5. Execute option trades + delta hedges
    6. Mark to market
    
    FOR each day until next option trading day:
        7. Rebalance delta hedge (maintain neutrality)
        8. Mark to market with updated stock prices
```

**Key Feature:** Daily delta rebalancing ensures the portfolio stays delta-neutral despite changing market conditions and option greeks.

#### Expiry Handling
At expiration:
- **ITM options:** Settle at intrinsic value
- **OTM options:** Expire worthless
- **Cash flow:** (intrinsic value × quantity × 100) added to cash
- **PnL:** Settlement - cost basis

---

### 5. Trading Strategy (`backtest/strategy.py`)

**Core Logic:** Identify and exploit options mispriced relative to theoretical value.

#### Mispricing Calculation
```python
mispricing = (market_price - theo_price) / theo_price

IF |mispricing| > threshold (10%):
    IF market_price > theo_price:
        → SELL option (overpriced)
    ELSE:
        → BUY option (underpriced)
```

#### Position Construction
Each trade is delta-neutral:
```python
1. Trade option: qty contracts
2. Calculate option delta: Δ
3. Hedge with stock: -qty × Δ × 100 shares

Net delta = qty × Δ × 100 + (-qty × Δ × 100) = 0
```

#### Risk Management
- **Max positions:** 5 concurrent trades (prevents over-concentration)
- **Strike filter:** Only trade options ±5% from ATM (avoid illiquid wings)
- **DTE range:** 7-90 days to expiry (avoid gamma risk near expiry)
- **Minimum mispricing:** 10% threshold (cover transaction costs)

#### Signal Generation
```python
@dataclass
class TradeSignal:
    symbol: str
    action: str              # 'buy' or 'sell'
    quantity: int
    market_price: float
    theo_price: float
    mispricing_pct: float
    strike: float
    expiry: datetime
    option_type: str
    delta: float
    hedge_quantity: float    # Shares to trade
    implied_vol: float       # Store for daily rebalancing
```

---

### 6. Strategy Comparison (`backtest/main.py`)

Runs all four volatility models and compares performance.

#### Performance Metrics

**Return Metrics:**
- Total return
- Annualized return
- CAGR (Compound Annual Growth Rate)

**Risk Metrics:**
- Volatility (annualized standard deviation)
- Maximum drawdown
- Sharpe ratio: (Return - RFR) / Volatility
- Calmar ratio: Return / Max Drawdown

**Trade Metrics:**
- Win rate
- Average profit per trade
- Number of trades
- Average holding period

#### Visualization
- Equity curves comparison
- Drawdown analysis
- IV forecast accuracy
- Trade distribution by moneyness

---

## 🎯 Key Insights

### Why GARCH Outperforms

1. **Volatility Clustering:** Markets exhibit periods of high/low volatility. GARCH captures this.
2. **Mean Reversion:** Volatility reverts to long-term average. GARCH models this explicitly.
3. **Recent Weighting:** GARCH gives more weight to recent data than simple historical average.
4. **Forward-Looking:** GARCH forecasts future volatility, not just measures past.

### Delta Hedging Reality

**Theory:** Perfect delta hedge = zero directional risk  
**Practice:** 
- Greeks change continuously (gamma risk)
- Daily rebalancing != continuous rebalancing
- Transaction costs accumulate
- Slippage on large rebalances

**Result:** Small residual delta exposure, but much better than unhedged.

### Transaction Costs Matter

With $0.50/contract + 1bp on stock:
- Need ~2-3% option mispricing just to break even
- 10% threshold ensures positive expected value
- Frequent rebalancing can erode profits

### Data Quality Issues

**Challenges encountered:**
- Stale quotes in options data
- Wide bid-ask spreads (use mid prices)
- Missing IV for some strikes
- Options with zero volume

**Solutions:**
- Liquidity filters
- Robust error handling
- Walk-forward validation (no lookahead bias)

---

## 🚀 Installation & Usage

### Prerequisites
```bash
# Install Dolt (version control for databases)
sudo bash -c 'curl -L https://github.com/dolthub/dolt/releases/latest/download/install.sh | sudo bash'

# Clone options price database (WARNING: ~100GB)
cd data/raw
dolt clone dolthub/options-prices options
cd options
dolt checkout main
```

### Python Environment
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install pandas numpy scipy yfinance arch xgboost matplotlib seaborn tqdm
```

### Running the Backtest

**Step 1: Data Preprocessing**
```bash
python src/preprocess.py --ticker SPY --start 2019-01-01 --end 2024-12-31
```

**Step 2: Run Strategy Comparison**
```bash
cd src/backtest
python main.py
```

**Step 3: Analyze Results**
```bash
jupyter notebook notebooks/exploration.ipynb
```

### Output Files
- `results/equity_curves.png` - Performance visualization
- `results/metrics_comparison.csv` - Summary statistics
- `results/{model}_equity.csv` - Daily equity values
- `results/{model}_iv_log.csv` - IV predictions over time
- `{model}_trade_log.csv` - Detailed trade history

---

## 🧪 Testing

```bash
# Test individual components
python src/pricing/pricer.py
python src/pricing/greeks.py
python src/backtest/engine.py

# Run full test suite
pytest tests/
```

**Test Coverage:**
- Black-Scholes pricing accuracy
- Greeks calculation validation
- Portfolio accounting (PnL, cost basis)
- Expiry settlement
- Delta rebalancing logic
- Transaction cost calculations

---

## 📈 Example Trade Flow

**Day 1 (2023-02-22):**
1. SPY = $398.54
2. 30-day ATM put (K=$383) trading at $4.42
3. Historical volatility model forecasts σ = 17%
4. Theoretical price = $4.15 → underpriced by 6.5%
5. **Action:** Buy 1 put @ $4.42
6. Put delta = -0.19 → **Hedge:** Short 19 shares @ $398.54
7. Net delta ≈ 0 (delta neutral)

**Day 2-29 (Daily rebalancing):**
- SPY moves, option delta changes
- Rebalance stock hedge to maintain delta neutrality
- Mark positions to market

**Day 30 (Expiry):**
- SPY = $395 (ITM)
- Intrinsic value = $383 - $395 = $0 (OTM, expires worthless)
- PnL = Settlement - Cost = $0 - $442 = -$442 loss
- Close stock hedge

---

## 🔍 Code Quality & Best Practices

**Design Principles:**
- **Modularity:** Separate pricing, volatility, and backtesting concerns
- **Type Safety:** Dataclasses and type hints throughout
- **Vectorization:** NumPy/Pandas for performance
- **Walk-Forward:** No lookahead bias in vol model updates
- **Documentation:** Docstrings with formulas and examples

**Performance Optimizations:**
- Parquet format (10x faster than CSV)
- Vectorized Greeks calculations
- Efficient DataFrame operations
- Progress bars for long-running backtests

**Financial Accuracy:**
- Realistic transaction costs
- Proper option expiry settlement
- Mark-to-market accounting
- Cost basis tracking (FIFO)

---

## 🎓 Educational Value

This project demonstrates:

**Quantitative Finance:**
- Option pricing theory (Black-Scholes)
- Greek hedging strategies
- Volatility modeling and forecasting
- Risk-neutral pricing

**Software Engineering:**
- Event-driven backtesting architecture
- Data pipeline design
- OOP with abstract base classes
- Testing and validation

**Data Science:**
- Time series modeling (GARCH)
- Machine learning for forecasting
- Walk-forward validation
- Performance attribution

**Trading Systems:**
- Signal generation
- Risk management
- Transaction cost modeling
- Portfolio accounting

---

## 📚 References & Further Reading

**Volatility Modeling:**
- Engle, R. (1982). "Autoregressive Conditional Heteroscedasticity"
- Bollerslev, T. (1986). "Generalized Autoregressive Conditional Heteroskedasticity"
- Gatheral, J. (2006). "The Volatility Surface: A Practitioner's Guide"

**Options Theory:**
- Hull, J. (2018). "Options, Futures, and Other Derivatives"
- Taleb, N. (1997). "Dynamic Hedging: Managing Vanilla and Exotic Options"
- Natenberg, S. (1994). "Option Volatility and Pricing"

**Backtesting:**
- Prado, M. (2018). "Advances in Financial Machine Learning"
- Bailey, D. et al. (2014). "The Probability of Backtest Overfitting"

---

## 🐛 Known Issues & Future Work

### Current Limitations
1. **No early exit logic:** Positions held until expiry (should close profitable/worthless positions early)
2. **Simple position sizing:** Fixed contract quantity (could use Kelly criterion)
3. **No portfolio Greeks limits:** Should cap vega/gamma exposure
4. **Single underlying:** Only SPY (could extend to multi-asset)
5. **European options only:** Real SPY options are American (early exercise)

### Potential Enhancements
- [ ] Add stop-loss and take-profit exits
- [ ] Implement dynamic position sizing
- [ ] Portfolio-level Greek constraints
- [ ] Multi-leg strategies (spreads, straddles)
- [ ] Regime detection (high/low vol environments)
- [ ] Real-time trading integration
- [ ] More sophisticated ML models (LSTM, Transformers)
- [ ] Intraday data and hedging
- [ ] American option pricing (binomial trees)
- [ ] Earnings event filters

---

## 👤 Author

Built as a comprehensive learning project in quantitative finance, combining options theory, volatility modeling, and backtesting best practices.

**Skills Demonstrated:**
- Derivatives pricing and hedging
- Time series forecasting
- Event-driven system design
- Financial data engineering
- Python for quantitative finance

---

## 📝 License

This project is for educational purposes. Not financial advice. Past performance does not guarantee future results.

---

## 🙏 Acknowledgments

- [Dolthub](https://github.com/dolthub/options-prices) for the open-source options database
- [CBOE](https://www.cboe.com/) for options data standardization
- The quantitative finance community for theory and best practices

