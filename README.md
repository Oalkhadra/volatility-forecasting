# Volatility Forecasting for Delta-Neutral Options Trading

> **A quantitative research project comparing volatility forecasting methods (Historical Rolling, GARCH, XGBoost) and their impact on risk-adjusted returns in a delta-neutral options trading strategy on SPY.**

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Research Questions](#research-questions)
3. [Project Pipeline](#project-pipeline)
4. [Data Sources](#data-sources)
5. [Volatility Forecasting Models](#volatility-forecasting-models)
6. [Trading Strategy](#trading-strategy)
7. [Results](#results)
8. [Key Findings & Insights](#key-findings--insights)
9. [Statistical Significance Analysis](#statistical-significance-analysis)
10. [Project Structure](#project-structure)
11. [Future Research Directions](#future-research-directions)
12. [How to Run](#how-to-run)

---

## Executive Summary

This project implements a complete backtesting framework to evaluate how different volatility forecasting methods translate into trading P&L. The core hypothesis is that more sophisticated volatility models (GARCH, Machine Learning) should identify option mispricings more accurately than naive historical volatility, leading to better risk-adjusted returns.

**Key Result**: While XGBoost significantly outperformed GARCH in volatility forecasting accuracy (p = 0.0043), **none of the strategies produced meaningful risk-adjusted returns** (all Sharpe ratios negative or near zero). The most interesting finding: **XGBoost generated substantial option-only P&L that was largely eroded by delta hedging costs**, highlighting the critical gap between forecasting accuracy and trading profitability.

### Performance Summary (Jan 2020 - Nov 2025)

| Model | Total Return | Ann. Return | Sharpe Ratio | Max Drawdown | Win Rate |
|-------|-------------|-------------|--------------|--------------|----------|
| **XGB** | +14.59% | +2.39% | -0.03 | -16.74% | 61.1% |
| Historical | -3.33% | -0.58% | -0.48 | -15.46% | 32.1% |
| GARCH | -32.63% | -6.61% | -1.24 | -38.12% | 14.3% |

---

## Research Questions

### Primary Question
**Which volatility estimation method produces the best risk-adjusted returns when trading options with delta-neutral hedging?**

*Approach*: Run identical trading strategies with different volatility inputs (Historical, GARCH, XGBoost) and compare Sharpe ratios, total returns, and drawdowns across a 5-year backtest period.

### Secondary Questions

#### 1. Is statistical forecasting accuracy predictive of trading performance?
*Approach*: Compare out-of-sample forecasting metrics (RMSE, correlation, QLIKE) against realized P&L. The hypothesis is that better forecasting should lead to better trading—but this project reveals that's not necessarily true.

**Recommendation**: Analyze the disconnect between XGBoost's superior forecasting (RMSE 21% better than GARCH) and its marginal trading improvement. Consider metrics beyond point accuracy (e.g., directional accuracy during regime changes, tail event prediction).

#### 2. Do sophisticated volatility models justify their complexity over simpler alternatives?
*Approach*: Use Diebold-Mariano tests to determine if performance differences are statistically significant, then weigh against implementation complexity and computational costs.

**Recommendation**: Frame this as an efficiency frontier analysis. GARCH requires daily refitting (computationally expensive) yet performs worst. Historical vol requires no training. XGBoost requires pre-training but is robust.

#### 3. Why does option P&L not translate to strategy P&L?
*Approach*: Decompose total P&L into: (1) Option trading P&L, (2) Delta hedge P&L, (3) Transaction costs. This reveals where value is created vs. destroyed.

**Recommendation**: This is your most novel finding. Deep dive into hedge drag—the cost of continuously rebalancing delta-neutral positions in a trending market.

---

## Project Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           PROJECT TIMELINE                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│  1993 ──────────── 2015 ──────────── 2020 ──────────── 2025                │
│    │                 │                 │                 │                  │
│    └─── TRAINING ────┘                 │                 │                  │
│         (Vol Models)                   │                 │                  │
│                      │                 │                 │                  │
│                      └─── TESTING ─────┘                 │                  │
│                           (Accuracy)                     │                  │
│                                        │                 │                  │
│                                        └─── BACKTEST ────┘                  │
│                                             (Trading)                       │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Phase 1: Volatility Model Development (1993-2014 Training)

1. **Data Preprocessing**: Load SPY prices (OHLCV), VIX data, and calculate log returns
2. **Feature Engineering** for XGBoost:
   - Lagged VIX (yesterday's close)
   - Realized volatility at multiple horizons (1d, 5d, 22d, 30d, 60d, 120d)
   - Leverage effect features (cumulative returns)
   - Asymmetric volatility (negative semivariance from Patton & Sheppard 2015)
3. **Model Training**:
   - XGBoost: RandomizedSearchCV with TimeSeriesSplit (5-fold), optimized on Spearman correlation
   - GARCH: Rolling 30-day window, fit daily during backtest
   - Historical: Simple 30-day rolling standard deviation

### Phase 2: Model Testing (2015-2019)

- Out-of-sample forecasting accuracy evaluation
- Diebold-Mariano statistical significance tests
- Regime-conditional performance analysis

### Phase 3: Strategy Backtesting (2020-2025)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         DAILY BACKTEST LOOP                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│  For each trading day:                                                       │
│    1. Forecast realized volatility (RV) using vol model                     │
│    2. Add Variance Risk Premium (VRP) adjustment → IV estimate              │
│    3. Price all 30-DTE options using Black-Scholes with forecasted IV       │
│    4. Identify mispricings:                                                 │
│       • BUY if market price < 50% of theoretical (underpriced)              │
│       • SELL if market price > 80% above theoretical (overpriced)           │
│    5. Execute option trades with delta hedge (stock position)               │
│    6. Rebalance delta hedges daily                                          │
│    7. Handle expirations and early exits (if mispricing resolves)           │
│    8. Mark-to-market and log P&L                                            │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Data Sources

| Data | Source | Period | Format |
|------|--------|--------|--------|
| **Options** | ThetaData | 2020-2025 | EOD snapshots, monthly parquet files |
| **SPY Prices** | yfinance | 1993-2025 | Daily OHLCV |
| **VIX** | yfinance | 1993-2025 | Daily closes |
| **Risk-Free Rate** | yfinance | 1993-2025 | 13-week T-bill |

### Data Structure

```
data/
├── processed/
│   ├── spy_prices.parquet          # Daily SPY OHLCV (1993-2025)
│   ├── vix_data.parquet            # Daily VIX closes
│   ├── risk_free_rate.parquet      # 13-week T-bill rates
│   └── spy_options_YYYY_MM.parquet # Monthly option snapshots (60+ files)
└── raw/
    └── ...                          # Raw data before preprocessing
```

---

## Volatility Forecasting Models

### 1. Historical Rolling Volatility (Naive Baseline)

**Implementation**: `src/volatility/historical.py`

The simplest approach—30-day trailing standard deviation of log returns, annualized.

```
σ_t = std(r_{t-29}, r_{t-28}, ..., r_t) × √252
```

**Strengths**: No look-ahead bias, computationally trivial, no training required  
**Weaknesses**: Backward-looking, slow to react to regime changes

### 2. GARCH(1,1)

**Implementation**: `src/volatility/forecasting.py` → `GARCHForecaster`

Classic econometric model capturing volatility clustering and mean-reversion.

```
σ²_t = ω + α·r²_{t-1} + β·σ²_{t-1}

Where:
  ω = long-run variance level
  α = reaction to recent shocks (ARCH term)
  β = persistence of variance (GARCH term)
```

**Multi-step forecast**: Uses unconditional variance as attractor with persistence decay:
```
E[σ²_{t+h}] = σ²_∞ + (α+β)^h × (σ²_t - σ²_∞)
```

**Strengths**: Captures volatility clustering, industry standard, interpretable  
**Weaknesses**: Symmetric response to positive/negative shocks (leverage effect), refitting is slow

### 3. XGBoost Machine Learning

**Implementation**: `src/volatility/forecasting.py` → `MLForecaster`

Gradient boosting model trained on engineered features capturing volatility dynamics.

**Features (14 total)**:
| Category | Features |
|----------|----------|
| **VIX** | Yesterday's VIX close |
| **Realized Vol** | 1d, 5d, 22d, 30d, 60d, 120d rolling volatility |
| **Leverage Effect** | 1d, 22d, 30d, 60d, 120d cumulative returns |
| **Asymmetric Vol** | 5-day negative semivariance (Patton & Sheppard 2015) |
| **Horizon** | Forecast horizon (30 days) |

**Training Configuration**:
- Objective: Custom asymmetric squared error (penalizes underprediction 1.5x)
- CV: TimeSeriesSplit with 5 folds
- Optimization: Spearman correlation (rank-based accuracy)
- Best params: `n_estimators=300, max_depth=3, learning_rate=0.01`

**Strengths**: Captures non-linear relationships, leverage effect, regime interactions  
**Weaknesses**: Black-box, requires pre-training, potential overfitting

---

## Trading Strategy

### Variance Risk Premium (VRP) Adjustment

**Implementation**: `src/volatility/vrp.py`

Before pricing options, realized volatility forecasts are converted to implied volatility estimates by adding the Variance Risk Premium—the systematic compensation option sellers receive.

```
IV_estimate = RV_forecast + VRP_adjustment
```

**VRP Estimation**:
- Historical VRP = VIX(t) - RV(t → t+30)
- Rolling 252-day average for stable estimates
- Regime-conditional: Different VRP by VIX level

- For backtest, regime-based VRP was used, with the following values:

| Regime | VIX Level | Mean VRP |
|--------|-----------|-------------|
| Low | < 15 | 2.37% |
| Normal | 15-25 | 3.32% |
| High | ≥ 25 | 6.08% |

### Mispricing Detection

**Implementation**: `src/backtest/strategy.py` → `MispricingStrategy`

```
Mispricing % = (Market Price - Theoretical Price) / Theoretical Price
```

**Trade Signals**:
- **BUY** if market price < -50% of theoretical price (market significantly underpriced)
- **SELL** if market price > +80% of theoretical (market significantly overpriced)

*Asymmetric thresholds*: Selling has unlimited risk, so requires larger mispricing to compensate.

### Delta Hedging

Every option position is immediately delta-hedged with stock:

```
Hedge shares = -Option contracts × Delta × 100
```

**Daily Rebalancing**: Delta changes as underlying moves and time passes. Hedges are rebalanced daily to maintain delta neutrality, ensuring the strategy profits from volatility (vega) not direction.

### Position Management

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Max positions | 21 | Diversification across strikes/expiries |
| Contracts per trade | 2 | Position sizing with capital constraints |
| Max leverage | 1.0x | Conservative—no margin borrowing |
| Max short exposure | 40% | Limit tail risk from short options |
| Early exit threshold | ±25% | Close if mispricing converges to fair value |

---

## Results

### Equity Curves

All strategies underperformed SPY buy-and-hold, with GARCH experiencing a catastrophic 38% drawdown during the 2022 volatility regime.

![Equity Curves](results/plots/equity_curves.png)

### P&L Decomposition

Based on $500,000 initial capital:

| Strategy | Total Return | **Net P&L** |
|----------|-------------|-------------|
| **XGB** | +14.59% | **~+$73,000** |
| Historical | -3.33% | **~-$17,000** |
| GARCH | -32.63% | **~-$163,000** |

**Critical Insight from Analysis**: The results exploration notebook reveals that XGBoost generated substantial *option-only* P&L, but delta hedging in a trending market destroyed significant gains. This is the "hedge drag" problem—continuously shorting stock (to hedge short puts) as the market rallies loses money on the hedge leg. Run the `results_exploration.ipynb` notebook to see the full P&L waterfall breakdown by strategy.

### Trade Distribution

| Strategy | Total Trades | Puts | Calls | Buys | Sells |
|----------|-------------|------|-------|------|-------|
| XGB | 1,682 | 63% | 37% | 40% | 60% |
| GARCH | 1,919 | 10% | 90% | 75% | 25% |
| Historical | 1,634 | 24% | 76% | 65% | 35% |

**XGB was more balanced but biased toward selling options (60% sells)**—primarily puts, which works well in rising markets but carries tail risk. GARCH and Historical were predominantly buying calls, essentially paying for upside exposure that often expired worthless.

### Win Rates by Trade Type

Win rates vary significantly by strategy and trade type. Run the `results_exploration.ipynb` notebook to see the detailed 4-way breakdown (Buy/Sell × Call/Put) with current results.

**Key patterns observed**:
- XGB's selling bias (60% sells) captures variance risk premium more effectively
- GARCH and Historical's heavy call buying (75-90%) suffered in periods of declining implied volatility
- Sell strategies generally outperformed buy strategies across all models

---

## Key Findings & Insights

### 1. Forecasting Accuracy ≠ Trading Profitability

XGBoost demonstrated statistically significant superiority over GARCH (p=0.0043 on QLIKE loss) and marginally outperformed Historical (5% RMSE reduction). Yet trading performance was marginal. **The bottleneck was not volatility forecasting—it was execution.**

*Implication*: For a trading strategy, directional accuracy during regime transitions matters more than average point accuracy. Consider optimizing for trading-relevant metrics (e.g., accuracy on days with high option volume, accuracy during VIX spikes).

### 2. The Hedge Drag Problem

Delta hedging is theoretically necessary for isolating volatility exposure, but in practice:
- **In trending markets** (2020-2024 SPY +70%): Continuous hedging means constantly shorting a rising asset
- **Rebalancing costs compound**: Daily adjustments add friction
- **Gamma scalping** requires realized volatility to exceed implied—but VRP means IV typically exceeds RV

*Implication*: Consider alternative hedge schedules (weekly rebalancing, delta bands, or dynamic hedge ratios based on market regime). Pure delta-neutrality may be too expensive.

### 3. Selling Options Dominates in Bull Markets

XGB's selling bias (60% of trades) was more profitable than the buying-heavy approaches of GARCH (75% buys) and Historical (65% buys). This is effectively harvesting the variance risk premium—compensation for bearing tail risk.

*Implication*: The strategy's success depends heavily on market regime. In a prolonged bear market or volatility spike (2008, March 2020), short volatility approaches would face massive losses. Consider regime-based position sizing or tail hedges.

### 4. GARCH Fails Spectacularly

GARCH's 38% drawdown and 14% win rate reveal a systematic failure mode: its mean-reverting forecasts consistently undershoot during sustained high-volatility periods. This led to:
- Heavy call buying (90% of trades were calls) at inflated prices
- Missing selling opportunities (theoretical prices too low)
- Catastrophic losses during 2022's volatility crush

*Implication*: Pure GARCH is likely inappropriate for options trading. Consider augmenting with regime detection or using as one input to an ensemble.

### 5. Statistical Significance vs. Economic Significance

| Test | Result |
|------|--------|
| XGB vs GARCH (Diebold-Mariano) | **Significant (p=0.004)** |
| XGB vs Historical | Not significant (p=0.26) |
| GARCH vs Historical | Not significant (p=0.54) |

Despite XGB's highly significant superiority over GARCH in forecasting, the trading edge was minimal. This underscores that **statistical significance is not sufficient for trading profitability**—transaction costs, execution quality, and strategy design matter equally.

---

## Statistical Significance Analysis

### Forecasting Accuracy (Test Period: 2015-2019)

| Metric | Historical | GARCH | XGBoost |
|--------|------------|-------|---------|
| RMSE | 5.02 | 6.07 | 4.79 |
| MAE | 3.89 | 5.24 | 4.15 |
| Correlation | 0.62 | 0.10 | 0.60 |
| Spearman | 0.65 | 0.04 | 0.62 |
| QLIKE Loss | 0.146 | 0.170 | **0.116** |

### Diebold-Mariano Test Results

Using QLIKE loss (appropriate for volatility forecasting):

```
H0: No difference in predictive accuracy
H1: Model 1 has different accuracy than Model 2

XGBoost vs Naive:    DM = -1.13,  p = 0.257  (No significant difference)
GARCH vs Naive:      DM = +0.62,  p = 0.538  (No significant difference)
XGBoost vs GARCH:    DM = -2.85,  p = 0.004  (XGBoost significantly better) ✓
```

**Interpretation**: XGBoost's outperformance over GARCH is robust (>99% confidence), but neither sophisticated model significantly beats naive historical volatility at forecasting.

### Regime-Conditional Performance

| Regime | XGB RMSE | GARCH RMSE | Historical RMSE |
|--------|----------|------------|-----------------|
| Low (<12%) | 2.08 | 3.47 | 2.15 |
| Normal (12-25%) | 4.24 | 5.27 | 4.33 |
| Stress (>25%) | 9.54 | 11.37 | 10.07 |

All models struggle during high-volatility regimes, but XGBoost shows more consistent performance across conditions.

---

## Project Structure

```
options-trading/
├── README.md                          # This file
├── data/
│   ├── processed/                     # Cleaned parquet files
│   └── raw/                           # Original data downloads
├── notebooks/
│   ├── xgboost_development.ipynb      # ML model training & validation
│   ├── results_exploration.ipynb      # Post-backtest analysis
│   └── scratch_paper.ipynb            # Exploratory analysis
├── results/
│   ├── backtest_analysis/             # Generated analysis plots
│   ├── early_exit_logs/               # Early exit trade details
│   ├── equity_logs/                   # Daily equity by strategy
│   ├── expiry_logs/                   # Option expiration details
│   ├── IV_logs/                       # Daily IV predictions
│   ├── plots/                         # Equity curves, IV comparison
│   ├── trade_logs/                    # Complete trade records
│   └── final_metrics.csv              # Summary performance metrics
├── src/
│   ├── backtest/
│   │   ├── engine.py                  # Core backtest loop, portfolio management
│   │   ├── main.py                    # Configuration & orchestration
│   │   └── strategy.py                # Signal generation, mispricing detection
│   ├── pricing/
│   │   ├── pricer.py                  # Black-Scholes option pricing
│   │   └── greeks.py                  # Delta, gamma, vega, theta, rho, IV solver
│   ├── preprocess/                    # Data cleaning scripts
│   └── volatility/
│       ├── forecasting.py             # GARCH & XGBoost models
│       ├── historical.py              # Rolling volatility calculator
│       └── vrp.py                     # Variance Risk Premium estimation
├── tests/                             # Unit tests
└── validations/                       # Model validation scripts
```

### Key Module Descriptions

| Module | Purpose |
|--------|---------|
| `src/backtest/engine.py` | Core backtest loop, portfolio management, position tracking, delta rebalancing, P&L calculation |
| `src/backtest/strategy.py` | Vectorized Black-Scholes pricing, mispricing detection, signal generation, early exit logic |
| `src/backtest/main.py` | Configuration parameters, backtest orchestration, metrics calculation, visualization |
| `src/volatility/forecasting.py` | GARCH(1,1) and XGBoost volatility forecasters with pre-training support |
| `src/volatility/vrp.py` | Variance Risk Premium calculator with regime-conditional estimation |
| `src/pricing/greeks.py` | All Greeks (delta, gamma, vega, theta, rho) + Newton-Raphson IV solver |

---

## Future Research Directions

### 1. Alternative Hedge Strategies
- **Discrete rebalancing**: Weekly vs. daily hedging to reduce transaction costs
- **Delta bands**: Only rebalance when delta exposure exceeds threshold
- **Vega hedging**: Add VIX futures to hedge volatility exposure directly

### 2. Regime-Aware Position Sizing
- Reduce short vol exposure when VIX term structure inverts (backwardation)
- Increase position sizes during low-VRP regimes
- Add tail hedges (OTM puts) during high-risk periods

### 3. Enhanced Forecasting
- Ensemble methods combining GARCH, ML, and implied vol
- High-frequency data for intraday vol prediction
- Attention mechanisms (transformers) for sequence modeling

### 4. Execution Improvements
- Model bid-ask spreads explicitly
- Add slippage modeling for realistic fills
- Consider limit orders vs. market orders

### 5. Alternative Strategies
- **Dispersion trading**: Long single-stock vol, short index vol
- **Calendar spreads**: Exploit term structure mispricings
- **Volatility arbitrage**: Long realized, short implied via variance swaps

---

## How to Run

### Prerequisites

```bash
# Python 3.9+
pip install -r requirements.txt
```

Key dependencies:
- `pandas`, `numpy` for data manipulation
- `xgboost` for ML model
- `arch` for GARCH modeling
- `scipy` for statistical tests and optimization
- `matplotlib`, `plotly` for visualization

Note: Options Data requires ThetaData Subscription

### Run Full Backtest

```bash
cd src/backtest
python main.py
```

This will:
1. Load all data (options, prices, VIX, rates)
2. Initialize VRP calculator
3. Run backtests for all three volatility models
4. Generate performance metrics and visualizations
5. Save results to `results/` directory

### Explore Results

Open `notebooks/results_exploration.ipynb` to:
- Analyze P&L decomposition
- View trade distribution by option type
- Generate regime-conditional performance metrics
- Create visualizations for reporting

### Train/Evaluate XGBoost Model

Open `notebooks/xgboost_development.ipynb` to:
- Perform hyperparameter tuning
- Evaluate forecasting accuracy
- Run statistical significance tests
- Analyze feature importance

---

*Last updated: December 2025*

