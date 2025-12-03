"""
Phase 5: Strategy Comparison Script

This script runs backtests for all 4 volatility models and compares results.

Goal: Answer the question "Which volatility forecasting method produces
      the best risk-adjusted returns for options mispricing strategy?"

Models to Compare:
    1. Historical Volatility (Baseline)
    2. Volatility Surface (Market Benchmark)
    3. GARCH(1,1) (Statistical Forecast)
    4. XGBoost ML (Machine Learning Forecast)

Output:
    - Equity curves for each model
    - Performance metrics (Sharpe, max drawdown, win rate)
    - Comparison visualizations
    - Summary report
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, List, Tuple

from src.backtest.engine import BacktestEngine
from src.backtest.strategy import MispricingStrategy
from src.volatility.forecasting import GARCHForecaster, MLForecaster
from src.volatility.surface import VolatilitySurface
from src.volatility.historical import calculate_realized_volatility

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    # Data paths
    'options_path': 'data/processed/spy_options.parquet',
    'prices_path': 'data/processed/spy_prices.parquet',
    'rates_path': 'data/processed/risk_free_rate.parquet',
    
    # Backtest parameters
    'start_date': datetime(2019, 1, 1),
    'end_date': datetime(2019, 12, 31),
    'initial_capital': 100000.0,
    
    # Strategy parameters
    'mispricing_threshold': 0.10,  # 10% mispricing required
    'max_positions': 5,
    'contracts_per_trade': 1,
    
    # Model parameters
    'garch_horizon': 7,  # 7-day forecast (weekly options)
    'ml_horizon': 7,
    'historical_window': 30,  # 30-day rolling window
    
    # Output
    'results_dir': 'results/phase5',
    'save_plots': True,
    'save_data': True
}

# ============================================================================
# MODEL SETUP FUNCTIONS
# ============================================================================

def setup_historical_model(prices: pd.Series, window: int = 30) -> callable:
    """
    Setup historical volatility model.
    
    Args:
        prices: Price series
        window: Rolling window for vol calculation
        
    Returns:
        Callable that returns current historical vol
        
    TODO: IMPLEMENT THIS
    
    Hints:
        - Calculate rolling returns
        - Calculate rolling vol (annualized)
        - Return a simple float (latest vol) or a callable
        - Could also return a lambda: lambda S, K, T: latest_vol
    """
    # TODO: YOUR CODE HERE
    raise NotImplementedError("TODO: Implement historical model setup")


def setup_surface_model(options_df: pd.DataFrame) -> VolatilitySurface:
    """
    Setup volatility surface model.
    
    Args:
        options_df: Options data with implied_volatility column
        
    Returns:
        Fitted VolatilitySurface object
        
    TODO: IMPLEMENT THIS
    
    Hints:
        - Create VolatilitySurface instance
        - Fit to options data (use .fit() method)
        - Return fitted surface
        - See src/volatility/surface.py for interface
    """
    # TODO: YOUR CODE HERE
    raise NotImplementedError("TODO: Implement surface model setup")


def setup_garch_model(returns: pd.Series, horizon: int = 7) -> GARCHForecaster:
    """
    Setup GARCH forecasting model.
    
    Args:
        returns: Historical return series
        horizon: Forecast horizon in days
        
    Returns:
        Fitted GARCHForecaster object
        
    TODO: IMPLEMENT THIS
    
    Hints:
        - Create GARCHForecaster instance
        - Fit to returns (use .fit() method)
        - Return fitted model
        - In backtest, you'll refit weekly with expanding window
    """
    # TODO: YOUR CODE HERE
    raise NotImplementedError("TODO: Implement GARCH model setup")


def setup_ml_model(returns: pd.Series, horizon: int = 7) -> MLForecaster:
    """
    Setup ML forecasting model.
    
    Args:
        returns: Historical return series
        horizon: Forecast horizon in days
        
    Returns:
        Fitted MLForecaster object
        
    TODO: IMPLEMENT THIS
    
    Hints:
        - Create MLForecaster instance with horizon
        - Fit to returns (use .fit() method)
        - Return fitted model
        - In backtest, you'll refit weekly with expanding window
    """
    # TODO: YOUR CODE HERE
    raise NotImplementedError("TODO: Implement ML model setup")


# ============================================================================
# BACKTEST EXECUTION
# ============================================================================

def run_single_backtest(model_name: str, 
                       vol_model, 
                       returns_data: pd.Series = None) -> Tuple[pd.DataFrame, BacktestEngine]:
    """
    Run backtest for a single volatility model.
    
    Args:
        model_name: Name of the model (for logging)
        vol_model: Volatility model instance
        returns_data: Historical returns (for ML/GARCH refitting)
        
    Returns:
        Tuple of (results_df, engine)
        
    TODO: IMPLEMENT THIS
    
    Steps:
        1. Create BacktestEngine instance
        2. Load data (options, prices, rates)
        3. Create MispricingStrategy with vol_model
        4. Attach strategy to engine
        5. Run backtest
        6. Return results
        
    Hints:
        - Use CONFIG dict for parameters
        - Print progress messages
        - Handle errors gracefully
        - For ML/GARCH: Pass returns_data to strategy
    """
    print(f"\n{'='*60}")
    print(f"Running backtest: {model_name}")
    print(f"{'='*60}")
    
    # TODO: YOUR CODE HERE
    # 1. Create engine
    # 2. Load data
    # 3. Setup strategy
    # 4. Run backtest
    # 5. Return results
    
    raise NotImplementedError("TODO: Implement run_single_backtest()")


def run_all_backtests() -> Dict[str, pd.DataFrame]:
    """
    Run backtests for all 4 volatility models.
    
    Returns:
        Dictionary mapping model_name -> results_df
        
    TODO: IMPLEMENT THIS
    
    Steps:
        1. Load price data and calculate returns
        2. Setup each of the 4 models
        3. Run backtest for each model
        4. Store results in dictionary
        5. Return results
        
    Structure:
        results = {}
        
        # Historical
        hist_model = setup_historical_model(...)
        results['Historical'] = run_single_backtest('Historical', hist_model)
        
        # Repeat for Surface, GARCH, ML
        
        return results
    """
    # TODO: YOUR CODE HERE
    raise NotImplementedError("TODO: Implement run_all_backtests()")


# ============================================================================
# PERFORMANCE METRICS
# ============================================================================

def calculate_performance_metrics(equity_curve: pd.Series, 
                                 initial_capital: float = 100000) -> Dict[str, float]:
    """
    Calculate performance metrics for a backtest.
    
    Args:
        equity_curve: Series of portfolio equity over time
        initial_capital: Starting capital
        
    Returns:
        Dictionary with performance metrics
        
    TODO: IMPLEMENT THIS
    
    Metrics to Calculate:
        - total_return: (final - initial) / initial
        - annualized_return: Assuming 252 trading days
        - sharpe_ratio: (mean_daily_return / std_daily_return) * sqrt(252)
        - max_drawdown: Maximum peak-to-trough decline
        - win_rate: % of positive return days
        - volatility: Annualized volatility of returns
        - calmar_ratio: annualized_return / abs(max_drawdown)
        
    Hints:
        - Calculate daily returns: equity_curve.pct_change()
        - Max drawdown: (trough - peak) / peak for worst drawdown
        - Handle NaN values from pct_change()
        - Annualize using sqrt(252) for vol, 252x for returns
    """
    # TODO: YOUR CODE HERE
    raise NotImplementedError("TODO: Implement calculate_performance_metrics()")


def create_metrics_comparison_table(all_results: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Create comparison table of metrics across all models.
    
    Args:
        all_results: Dict mapping model_name -> results_df
        
    Returns:
        DataFrame with models as rows, metrics as columns
        
    TODO: IMPLEMENT THIS
    
    Steps:
        1. For each model, extract equity curve from results
        2. Calculate metrics using calculate_performance_metrics()
        3. Build DataFrame with results
        4. Sort by Sharpe ratio (or another metric)
        5. Return comparison table
        
    Expected output columns:
        - Total Return (%)
        - Annualized Return (%)
        - Sharpe Ratio
        - Max Drawdown (%)
        - Win Rate (%)
        - Volatility (%)
        - Calmar Ratio
    """
    # TODO: YOUR CODE HERE
    raise NotImplementedError("TODO: Implement create_metrics_comparison_table()")


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_equity_curves(all_results: Dict[str, pd.DataFrame], 
                      save_path: str = None):
    """
    Plot equity curves for all models on one chart.
    
    Args:
        all_results: Dict mapping model_name -> results_df
        save_path: Path to save figure (optional)
        
    TODO: IMPLEMENT THIS
    
    Requirements:
        - One line per model
        - Normalize to start at 100 (or initial capital)
        - Include legend
        - Add title, axis labels
        - Use professional color scheme
        - Add horizontal line at initial capital
        - Optionally save to file
        
    Hints:
        - Use plt.figure(figsize=(12, 6))
        - Extract equity column from each results_df
        - plt.plot(dates, equity, label=model_name)
        - plt.legend()
        - plt.grid(alpha=0.3)
    """
    # TODO: YOUR CODE HERE
    pass


def plot_drawdown_comparison(all_results: Dict[str, pd.DataFrame], 
                            save_path: str = None):
    """
    Plot drawdown curves for all models.
    
    Args:
        all_results: Dict mapping model_name -> results_df
        save_path: Path to save figure (optional)
        
    TODO: IMPLEMENT THIS (OPTIONAL)
    
    Drawdown calculation:
        - Running maximum of equity curve
        - Drawdown = (equity - running_max) / running_max
        - Plot as negative percentage
    """
    # TODO: YOUR CODE HERE
    pass


def plot_metrics_bar_chart(metrics_df: pd.DataFrame, 
                           save_path: str = None):
    """
    Create bar chart comparing key metrics.
    
    Args:
        metrics_df: DataFrame from create_metrics_comparison_table()
        save_path: Path to save figure (optional)
        
    TODO: IMPLEMENT THIS (OPTIONAL)
    
    Create subplots for:
        - Sharpe Ratio
        - Max Drawdown
        - Total Return
        - Win Rate
    """
    # TODO: YOUR CODE HERE
    pass


def plot_rolling_sharpe(all_results: Dict[str, pd.DataFrame],
                       window: int = 20,
                       save_path: str = None):
    """
    Plot rolling Sharpe ratio over time.
    
    Args:
        all_results: Dict mapping model_name -> results_df
        window: Rolling window for Sharpe calculation
        save_path: Path to save figure (optional)
        
    TODO: IMPLEMENT THIS (OPTIONAL)
    
    Shows how strategy performance evolves over time.
    Useful for identifying regime changes.
    """
    # TODO: YOUR CODE HERE
    pass


# ============================================================================
# REPORTING
# ============================================================================

def generate_summary_report(all_results: Dict[str, pd.DataFrame],
                           metrics_df: pd.DataFrame,
                           save_path: str = None):
    """
    Generate text summary report of backtest results.
    
    Args:
        all_results: Dict mapping model_name -> results_df
        metrics_df: DataFrame with performance metrics
        save_path: Path to save report (optional)
        
    TODO: IMPLEMENT THIS (OPTIONAL)
    
    Include:
        - Overview of experiment
        - Summary statistics for each model
        - Winner announcement (best Sharpe)
        - Key findings
        - Save to .txt file if save_path provided
    """
    # TODO: YOUR CODE HERE
    pass


def save_results_to_csv(all_results: Dict[str, pd.DataFrame],
                       metrics_df: pd.DataFrame,
                       output_dir: str):
    """
    Save all results to CSV files.
    
    Args:
        all_results: Dict mapping model_name -> results_df
        metrics_df: DataFrame with performance metrics
        output_dir: Directory to save files
        
    TODO: IMPLEMENT THIS
    
    Save:
        - metrics_comparison.csv: Performance metrics table
        - {model_name}_equity.csv: Equity curve for each model
        - combined_results.csv: All equity curves in one file
    """
    # TODO: YOUR CODE HERE
    pass


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main execution function.
    
    TODO: IMPLEMENT THIS
    
    Flow:
        1. Print configuration
        2. Run all backtests
        3. Calculate metrics
        4. Generate visualizations
        5. Save results
        6. Print summary
    """
    print("="*80)
    print("OPTIONS TRADING STRATEGY COMPARISON - PHASE 5")
    print("="*80)
    print("\nConfiguration:")
    for key, value in CONFIG.items():
        print(f"  {key}: {value}")
    
    # TODO: YOUR CODE HERE
    # 1. Run backtests
    # 2. Calculate metrics
    # 3. Generate plots
    # 4. Save results
    # 5. Print summary
    
    print("\n" + "="*80)
    print("PHASE 5 COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    main()


# ============================================================================
# INTERVIEW PREPARATION NOTES
# ============================================================================

"""
KEY QUESTIONS TO PREPARE FOR:

1. "Why did you choose these 4 volatility models?"
   - Historical: Naive baseline (assumes vol is constant)
   - Surface: Market benchmark (what is market pricing?)
   - GARCH: Industry standard for vol forecasting (econometric)
   - ML: Modern approach (non-parametric, captures complex patterns)

2. "Which model performed best and why?"
   - [Your analysis here after running backtests]
   - Consider: Did GARCH beat ML? Why/why not?
   - Did any model beat the surface (market)?

3. "What were the main challenges?"
   - Data quality, model overfitting, transaction costs
   - Rebalancing frequency, position sizing
   - Walk-forward validation (avoiding lookahead bias)

4. "How would you improve this strategy?"
   - Better execution (limit orders vs market orders)
   - Dynamic position sizing based on conviction
   - Ensemble of models (combine forecasts)
   - Additional filters (liquidity, bid-ask spread)
   - Risk management (stop losses, max drawdown limits)

5. "What did you learn about volatility forecasting?"
   - Vol is mean-reverting but has clusters
   - Recent vol is highly predictive of near-term vol
   - Implied vol > realized vol on average (variance risk premium)
   - Model accuracy matters less than directional correctness

6. "How realistic is this backtest?"
   - Pros: Real data, transaction costs, delta hedging
   - Cons: No slippage, perfect fills, no funding costs
   - Improvement: Add bid-ask spreads, simulate fills

7. "What role would this play in a portfolio?"
   - Market-neutral (low correlation to equity markets)
   - Volatility harvesting strategy
   - Crisis alpha potential (vol spikes)
   - Diversification benefit
"""

