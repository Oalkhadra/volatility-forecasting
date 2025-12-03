# Phase 4: Backtesting Engine & Strategy Implementation

## 🎯 Objective
Build a robust, event-driven(ish) backtesting engine to simulate trading Strategies B, C, and D. 
This is the "core" of your portfolio project—where your models meet reality.

## 🏗 Architecture

We will build a modular system to keep things clean and testable:

1.  **`Portfolio`**: Tracks cash, open positions, and equity curve.
2.  **`Strategy` (Abstract Base Class)**: Defines the interface for all strategies.
3.  **`BacktestEngine`**: The "time machine" that feeds data to strategies and executes trades.

### The Trading Loop (Weekly)
For each Saturday in the dataset:
1.  **Update**: Mark-to-market all existing positions using current prices.
2.  **Rebalance**: Adjust hedges (delta-neutral) or close expiring positions.
3.  **Signal**: Ask the Strategy for new trades.
4.  **Execute**: Open new positions updates the Portfolio.
5.  **Record**: Log PnL and Greeks for analysis.

## 🛠 Tasks

### 1. `src/backtest/engine.py`
- [ ] Implement `Portfolio` class
    - Track `cash` and `positions` dictionary
    - `mark_to_market(current_prices)`
    - `execute_trade(trade)`
- [ ] Implement `BacktestEngine` class
    - Main loop over dates
    - Data feeding mechanism

### 2. `src/backtest/strategy.py`
- [ ] Implement `BaseStrategy` class
- [ ] Implement `VolRiskPremiumStrategy` (Generic short vol logic)
    - Logic: Sell straddles/strangles when Model Vol < Market Vol
    - We will inject the different models (Hist, Surface, Forecast) into this structure.

## 💡 Key Concepts for Interviews

### 1. Path Dependency
*   **Q:** "Why can't you just vectorize the PnL calculation?"
*   **A:** Because of path dependency. Margin calls, stop-losses, and delta-hedging decisions depend on the *state* of the portfolio at time $t$, which depends on decisions made at $t-1$.

### 2. Look-Ahead Bias
*   We must be strictly careful not to use data from Saturday to make decisions for Friday's close (or vice versa depending on execution).
*   **Rule:** Signals are generated using data up to time $t$. Trades execute at prices at time $t$ (or $t+1$ for realistic lag).

### 3. Transaction Costs
*   We will assume execution at **mid-price** (optimistic) or **bid/ask** (realistic).
*   For this phase, we'll start with a fixed slip/fee per contract (e.g., $0.05/contract + slippage).

## 📂 File Structure

```text
src/
└── backtest/
    ├── __init__.py
    ├── engine.py       # Portfolio and Engine logic
    └── strategy.py     # Strategy logic
```

