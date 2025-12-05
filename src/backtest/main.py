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
from src.volatility.historical import RollingVolCalculator

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    # Data paths
    'options_path': 'data/processed/spy_options.parquet',
    'prices_path': 'data/processed/spy_prices.parquet',
    'rates_path': 'data/processed/risk_free_rate.parquet',
    
    # Backtest parameters
    'start_date': datetime(2023 ,1, 1),
    'end_date': datetime(2025, 12, 1),
    'initial_capital': 100000.0,
    
    # Strategy parameters
    'mispricing_threshold': 0.5,  # 50% mispricing required
    'max_positions': 20,
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
                       returns_data: pd.Series = None,
                       options_data: pd.DataFrame = None,
                       prices_data: pd.DataFrame = None) -> Tuple[pd.DataFrame, BacktestEngine]:
    """
    Run backtest for a single volatility model.
    
    Args:
        model_name: Name of the model (for logging)
        vol_model: Volatility model instance
        returns_data: Historical returns (for ML/GARCH refitting)
        options_data: Options data (for VolatilitySurface refitting)
        prices_data: Price data (for Historical vol calculator)
        
    Returns:
        Tuple of (results_df, engine)

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
        - For Historical: Pass prices_data to strategy
    """
    print(f"\n{'='*60}")
    print(f"Running backtest: {model_name}")
    print(f"{'='*60}")

    # 1. Create engine
    engine = BacktestEngine(
        start_date=CONFIG['start_date'],
        end_date=CONFIG['end_date'],
        initial_capital=CONFIG['initial_capital']
    )
    # 2. Load data
    engine.load_data(
        options_path=CONFIG['options_path'],
        prices_path=CONFIG['prices_path'],
        rates_path=CONFIG['rates_path']
    )

    # 3. Setup strategy
    strategy = MispricingStrategy(
        vol_model=vol_model,
        returns_data=returns_data,
        options_data=options_data,
        prices_data=prices_data,
        threshold=CONFIG['mispricing_threshold'],
        max_positions=CONFIG['max_positions'],
        contracts_per_trade=CONFIG['contracts_per_trade']
    )
    engine.set_strategy(strategy)

    # 4. Run backtest
    results = engine.run()
    print(f"  Complete! Final equity: ${results.iloc[-1]['equity']:,.2f}")
    
    return results, engine


def run_all_backtests() -> Dict[str, Tuple[pd.DataFrame, BacktestEngine]]:
    """
    Run backtests for all 4 volatility models.
    
    Returns:
        Dictionary mapping model_name -> (results_df, engine)
 
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
    # Load data
    prices = pd.read_parquet(CONFIG['prices_path'])
    options = pd.read_parquet(CONFIG['options_path'])
    
    # Calculate returns and initialize results_dict
    returns = np.log(prices['close'] / prices['close'].shift(1)).dropna()
    results = {}

    # # # Run backtest for each modeling approach
    # Historical (pass prices_data for volatility calculator updates)
    hist_model = RollingVolCalculator(prices_data=prices, 
                                        window=CONFIG['historical_window'], 
                                        annualize=True)
    results_df, engine = run_single_backtest('historical', hist_model, 
                                             returns_data=None, 
                                             options_data=None, 
                                             prices_data=prices)
    results['historical'] = (results_df, engine)

    # Volatility surface
    vol_surface = VolatilitySurface('polynomial')
    results_df, engine = run_single_backtest('vol_surface', vol_surface, 
                                             returns_data=None, 
                                             options_data=options, 
                                             prices_data=None)
    results['vol_surface'] = (results_df, engine)

    return results


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
        
    """
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


def create_metrics_comparison_table(all_results: Dict[str, Tuple[pd.DataFrame, BacktestEngine]]) -> pd.DataFrame:
    """
    Create comparison table of metrics across all models.
    
    Args:
        all_results: Dict mapping model_name -> (results_df, engine)
        
    Returns:
        DataFrame with models as rows, metrics as columns

    """
    metrics_list = []

    for model_name, (results_df, _) in all_results.items():
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


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_equity_curves(all_results: Dict[str, Tuple[pd.DataFrame, BacktestEngine]], 
                      save_path: str = None):
    """
    Plot equity curves for all models on one chart.
    
    Args:
        all_results: Dict mapping model_name -> (results_df, engine)
        save_path: Path to save figure (optional)

    """
    plt.figure(figsize=(12, 6))
    
    for model_name, (results_df, _) in all_results.items():
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


def plot_drawdown_comparison(all_results: Dict[str, Tuple[pd.DataFrame, BacktestEngine]], 
                            save_path: str = None):
    """
    Plot drawdown curves for all models.
    
    Args:
        all_results: Dict mapping model_name -> (results_df, engine)
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


def plot_rolling_sharpe(all_results: Dict[str, Tuple[pd.DataFrame, BacktestEngine]],
                       window: int = 20,
                       save_path: str = None):
    """
    Plot rolling Sharpe ratio over time.
    
    Args:
        all_results: Dict mapping model_name -> (results_df, engine)
        window: Rolling window for Sharpe calculation
        save_path: Path to save figure (optional)
        
    TODO: IMPLEMENT THIS (OPTIONAL)
    
    Shows how strategy performance evolves over time.
    Useful for identifying regime changes.
    """
    # TODO: YOUR CODE HERE
    pass


def plot_iv_comparison(all_results: Dict[str, Tuple[pd.DataFrame, BacktestEngine]],
                      save_path: str = None):
    """
    Plot implied volatility predictions from all models over time.
    
    Args:
        all_results: Dict mapping model_name -> (results_df, engine)
        save_path: Path to save figure (optional)
    
    Shows how different volatility forecasting models predict IV over time.
    Useful for understanding model divergence and identifying regime changes.
    """
    plt.figure(figsize=(14, 7))
    
    for model_name, (_, engine) in all_results.items():
        if engine.portfolio.iv_log:
            iv_df = pd.DataFrame(engine.portfolio.iv_log)
            plt.plot(iv_df['date'], iv_df['implied_vol'], 
                    label=model_name, linewidth=2, alpha=0.8)
    
    plt.title('Implied Volatility Predictions: Model Comparison', 
             fontsize=14, fontweight='bold')
    plt.xlabel('Date')
    plt.ylabel('Implied Volatility')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


# ============================================================================
# REPORTING
# ============================================================================

def generate_summary_report(all_results: Dict[str, Tuple[pd.DataFrame, BacktestEngine]],
                           metrics_df: pd.DataFrame,
                           save_path: str = None):
    """
    Generate text summary report of backtest results.
    
    Args:
        all_results: Dict mapping model_name -> (results_df, engine)
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


def save_results_to_csv(all_results: Dict[str, Tuple[pd.DataFrame, BacktestEngine]],
                       metrics_df: pd.DataFrame,
                       output_dir: str):
    """
    Save all results to CSV files.
    
    Args:
        all_results: Dict mapping model_name -> (results_df, engine)
        metrics_df: DataFrame with performance metrics
        output_dir: Directory to save files
        
    Save:
        - metrics_comparison.csv: Performance metrics table
        - {model_name}_equity.csv: Equity curve for each model
        - {model_name}_expiry_log.csv: Expiry events for each model
        - {model_name}_iv_log.csv: Daily IV predictions for each model
        - combined_results.csv: All equity curves in one file
        - combined_iv.csv: All IV predictions in one file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save metrics comparison
    metrics_path = os.path.join(output_dir, 'metrics_comparison.csv')
    metrics_df.to_csv(metrics_path)
    print(f"  Saved metrics to: {metrics_path}")
    
    # Save individual equity curves, expiry logs, and IV logs
    for model_name, (results_df, engine) in all_results.items():
        # Save equity curve
        equity_path = os.path.join(output_dir, f'{model_name}_equity.csv')
        results_df.to_csv(equity_path, index=False)
        print(f"  Saved {model_name} equity curve to: {equity_path}")
        
        # Save expiry log if available
        if engine.portfolio.expiry_log:
            expiry_df = pd.DataFrame(engine.portfolio.expiry_log)
            expiry_path = os.path.join(output_dir, f'{model_name}_expiry_log.csv')
            expiry_df.to_csv(expiry_path, index=False)
            print(f"  Saved {model_name} expiry log to: {expiry_path}")
        
        # Save IV log if available
        if engine.portfolio.iv_log:
            iv_df = pd.DataFrame(engine.portfolio.iv_log)
            iv_path = os.path.join(output_dir, f'{model_name}_iv_log.csv')
            iv_df.to_csv(iv_path, index=False)
            print(f"  Saved {model_name} IV log to: {iv_path}")
    
    # Save combined results (all equity curves in one file)
    combined_df = pd.DataFrame()
    for model_name, (results_df, _) in all_results.items():
        if combined_df.empty:
            combined_df['date'] = results_df['date']
        combined_df[f'{model_name}_equity'] = results_df['equity'].values
    
    combined_path = os.path.join(output_dir, 'combined_results.csv')
    combined_df.to_csv(combined_path, index=False)
    print(f"  Saved combined results to: {combined_path}")
    
    # Save combined IV predictions (all models' IVs side-by-side)
    combined_iv_df = pd.DataFrame()
    for model_name, (_, engine) in all_results.items():
        if engine.portfolio.iv_log:
            iv_df = pd.DataFrame(engine.portfolio.iv_log)
            if combined_iv_df.empty:
                combined_iv_df['date'] = iv_df['date']
                combined_iv_df['underlying_price'] = iv_df['underlying_price']
                combined_iv_df['day_type'] = iv_df['day_type']
            combined_iv_df[f'{model_name}_iv'] = iv_df['implied_vol'].values
    
    if not combined_iv_df.empty:
        combined_iv_path = os.path.join(output_dir, 'combined_iv.csv')
        combined_iv_df.to_csv(combined_iv_path, index=False)
        print(f"  Saved combined IV predictions to: {combined_iv_path}")


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
    plot_iv_comparison(all_results,
                      save_path=f"{CONFIG['results_dir']}/iv_comparison.png")
    
    # Save results to CSV
    print("\nStep 4: Saving results to CSV...")
    save_results_to_csv(all_results, metrics_df, CONFIG['results_dir'])
    
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

