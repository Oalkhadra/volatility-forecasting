import sys
import os

# Add parent directory (src/) to path for absolute imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, Tuple

from backtest.engine import BacktestEngine
from backtest.strategy import MispricingStrategy
from volatility.forecasting import GARCHForecaster, MLForecaster
from volatility.historical import RollingVolCalculator
from volatility.vrp import VarianceRiskPremiumCalculator

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    # Data paths
    'options_dir': 'data/processed',  # Directory containing spy_options_YYYY_MM.parquet files
    'prices_path': 'data/processed/spy_prices.parquet',
    'rates_path': 'data/processed/risk_free_rate.parquet',
    'vix_path': 'data/processed/vix_data.parquet',
    
    # Backtest parameters
    'start_date': datetime(2020 , 1, 1),
    'end_date': datetime(2025, 11, 30),
    'initial_capital': 500000.0,
    
    # Strategy parameters - asymmetric thresholds
    'buy_threshold': 0.50,         # 50% underpriced to buy (mispricing < -50%)
    'sell_threshold': 1.0,        # 80% overpriced to sell (mispricing > +100%)
    'exit_threshold_buy': 0.25,    # Exit long if mispricing rises above -50% (fair value range)
    'exit_threshold_sell': 0.50,   # Exit short if mispricing falls below +100% (fair value range)
    'max_positions': 21,
    'contracts_per_trade': 2,     # Max contracts per trade (will be reduced if needed)
    
    # Risk management parameters
    'max_leverage': 1,            # Max total capital usage / equity ratio
    'max_position_pct': 0.50,     # Max single position as % of equity
    'max_short_exposure': 0.4,   # Max margin requirement for short positions (20% of notional) as % of equity 
     
    # Model parameters
    'historical_window': 30,  # 30-day rolling window
    
    # Variance Risk Premium (VRP) parameters
    'use_vrp_calculator': True,   # Use empirical VRP instead of fixed premium
    'vrp_rv_window': 30,          # Window for realized vol in VRP calculation
    'vrp_lookback': 252,          # Lookback for rolling VRP average (1 year)
    'vrp_method': 'regime',      # VRP method: 'rolling', 'regime', or 'regime_rolling'
    'vrp_use_regime': True,       # Estimate separate VRP by regime
    
    # Legacy: Fixed risk premium (only used if use_vrp_calculator=False)
    'risk_premium_fallback': 0.1,
    
    # Output
    'results_dir': 'results',
    'save_plots': True,
    'save_data': True
}

# ============================================================================
# BACKTEST EXECUTION
# ============================================================================

def run_single_backtest(model_name: str, 
                       vol_model,
                       vrp_calculator: VarianceRiskPremiumCalculator = None,
                       risk_premium: float = 0.0) -> Tuple[pd.DataFrame, BacktestEngine]:
    """
    Run backtest for a single volatility model.
    
    Args:
        model_name: Name of the model (for logging)
        vol_model: Volatility model instance
        vrp_calculator: VRP calculator for empirical VRP adjustment (preferred)
        risk_premium: Legacy fixed variance risk premium (used if vrp_calculator is None)
 
    Returns:
        Tuple of (results_df, engine, trade_log)

    Steps:
        1. Create BacktestEngine instance
        2. Load data (options, prices, rates)
        3. Create MispricingStrategy with vol_model and VRP calculator
        4. Attach strategy to engine
        5. Run backtest
        6. Return results

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
        options_dir=CONFIG['options_dir'],
        prices_path=CONFIG['prices_path'],
        rates_path=CONFIG['rates_path']
    )

    # 3. Setup strategy with VRP calculator
    strategy = MispricingStrategy(
        vol_model=vol_model,
        options_data=engine.options_data,
        prices_data=engine.prices_data,
        buy_threshold=CONFIG['buy_threshold'],
        sell_threshold=CONFIG['sell_threshold'],
        exit_threshold_buy=CONFIG['exit_threshold_buy'],
        exit_threshold_sell=CONFIG['exit_threshold_sell'],
        max_positions=CONFIG['max_positions'],
        contracts_per_trade=CONFIG['contracts_per_trade'],
        max_leverage=CONFIG['max_leverage'],
        max_position_pct=CONFIG['max_position_pct'],
        max_short_exposure=CONFIG['max_short_exposure'],
        risk_premium=risk_premium,
        vrp_calculator=vrp_calculator,
        vrp_method=CONFIG['vrp_method']
    )
    engine.set_strategy(strategy)

    # 4. Run backtest
    results, trade_log = engine.run()
    print(f"  Complete! Final equity: ${results.iloc[-1]['equity']:,.2f}")
    
    return results, engine, trade_log


def run_all_backtests() -> Dict[str, Tuple[pd.DataFrame, BacktestEngine, pd.DataFrame]]:
    """
    Run backtests for all volatility models.
    
    Returns:
        Dictionary mapping model_name -> (results_df, engine, expiry_log)
 
    Steps:
        1. Load price and VIX data
        2. Initialize VRP calculator
        3. Setup each volatility model
        4. Run backtest for each model
        5. Store results in dictionary
        6. Return results
    """
    
    # Create trade logs directory if it doesn't exist
    trade_logs_dir = os.path.join(CONFIG['results_dir'], 'trade_logs')
    os.makedirs(trade_logs_dir, exist_ok=True)
    
    # Load price and VIX data for VRP calculation
    print("\nLoading data for VRP calculation...")
    prices_df = pd.read_parquet(CONFIG['prices_path'])
    vix_df = pd.read_parquet(CONFIG['vix_path'])
    prices_df['date'] = pd.to_datetime(prices_df['date'])
    vix_df['date'] = pd.to_datetime(vix_df['date'])
    
    # Calculate log returns if not present
    if 'log_ret' not in prices_df.columns:
        prices_df['log_ret'] = np.log(prices_df['close'] / prices_df['close'].shift(1)) * 100
    
    # Initialize VRP calculator (if enabled)
    vrp_calc = None
    if CONFIG['use_vrp_calculator']:
        print("\nInitializing Variance Risk Premium calculator...")
        vrp_calc = VarianceRiskPremiumCalculator(
            prices_df=prices_df,
            vix_df=vix_df,
            rv_window=CONFIG['vrp_rv_window'],
            vrp_lookback=CONFIG['vrp_lookback'],
            use_regime=CONFIG['vrp_use_regime']
        )
        
        # Fit on data before backtest start date (no look-ahead bias)
        vrp_calc.fit(end_date=CONFIG['start_date'])
        
        # Print VRP summary
        print("\nVRP Analysis Summary:")
        print(vrp_calc.get_analysis_summary().to_string(index=False))
    
    # Initialize results_dict
    results = {}

    ## Run backtest for each modeling approach
    
    # Historical RV model
    hist_model = RollingVolCalculator(window=CONFIG['historical_window'], annualize=True)
    results_df, engine, trade_log = run_single_backtest(
        'historical', 
        hist_model,
        vrp_calculator=vrp_calc,
        risk_premium=CONFIG['risk_premium_fallback'] if not CONFIG['use_vrp_calculator'] else 0.0
    )
    expiry_log = pd.DataFrame(engine.portfolio.expiry_log) if engine.portfolio.expiry_log else pd.DataFrame()
    results['historical'] = (results_df, engine, expiry_log)
    
    trade_log_path = os.path.join(trade_logs_dir, 'historical_trade_log.csv')
    trade_log.to_csv(trade_log_path, index=False)
    print(f"  Saved trade log to: {trade_log_path}")

    # GARCH model
    garch = GARCHForecaster()
    results_df, engine, trade_log = run_single_backtest(
        'GARCH',
        garch,
        vrp_calculator=vrp_calc,
        risk_premium=CONFIG['risk_premium_fallback'] if not CONFIG['use_vrp_calculator'] else 0.0
    )
    expiry_log = pd.DataFrame(engine.portfolio.expiry_log) if engine.portfolio.expiry_log else pd.DataFrame()
    results['GARCH'] = (results_df, engine, expiry_log)
    
    trade_log_path = os.path.join(trade_logs_dir, 'GARCH_trade_log.csv')
    trade_log.to_csv(trade_log_path, index=False)
    print(f"  Saved trade log to: {trade_log_path}")

    # XGBoost ML model - Pre-train on historical data before backtest period
    print("\nPre-training XGBoost model...")
    xgb_model = MLForecaster(pretrained=True)
    xgb_model.pretrain(prices_df, cutoff_date=CONFIG['start_date'])
    
    results_df, engine, trade_log = run_single_backtest(
        'XGB', 
        xgb_model,
        vrp_calculator=vrp_calc,
        risk_premium=CONFIG['risk_premium_fallback'] if not CONFIG['use_vrp_calculator'] else 0.0)

    expiry_log = pd.DataFrame(engine.portfolio.expiry_log) if engine.portfolio.expiry_log else pd.DataFrame()
    results['XGB'] = (results_df, engine, expiry_log)
    
    trade_log_path = os.path.join(trade_logs_dir, 'XGB_trade_log.csv')
    trade_log.to_csv(trade_log_path, index=False)
    print(f"  Saved trade log to: {trade_log_path}")

    return results


# ============================================================================
# PERFORMANCE METRICS
# ============================================================================

def calculate_performance_metrics(equity_curve: pd.Series, 
                                 expiry_log: pd.DataFrame = None,
                                 initial_capital: float = 5000) -> Dict[str, float]:
    """
    Calculate performance metrics for a backtest.
    
    Args:
        equity_curve: Series of portfolio equity over time
        expiry_log: DataFrame of option expirations with realized P&L
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
    
    # Win rate (based on individual option expirations, not days)
    if expiry_log is not None and len(expiry_log) > 0:
        # Each expiry event represents completed trades with realized P&L
        winning_trades = (expiry_log['pnl'] > 0).sum()
        total_trades = len(expiry_log)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
    else:
        # Fallback to daily returns if no expiry log available
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


def create_metrics_comparison_table(all_results: Dict[str, Tuple[pd.DataFrame, BacktestEngine, pd.DataFrame]]) -> pd.DataFrame:
    """
    Create comparison table of metrics across all models.
    
    Args:
        all_results: Dict mapping model_name -> (results_df, engine, expiry_log)
        
    Returns:
        DataFrame with models as rows, metrics as columns

    """
    metrics_list = []

    for model_name, (results_df, _, expiry_log) in all_results.items():
        equity = results_df['equity']
        metrics = calculate_performance_metrics(equity, expiry_log, CONFIG['initial_capital'])
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

def plot_equity_curves(all_results: Dict[str, Tuple[pd.DataFrame, BacktestEngine, pd.DataFrame]], 
                      save_path: str = None):
    """
    Plot equity curves for all models on one chart with SPY buy-and-hold benchmark.
    
    Args:
        all_results: Dict mapping model_name -> (results_df, engine, expiry_log)
        save_path: Path to save figure (optional)

    """
    plt.figure(figsize=(12, 6))
    
    # Get initial capital from first model's engine
    first_engine = next(iter(all_results.values()))[1]
    initial_capital = first_engine.portfolio.initial_capital
    
    # Plot all strategy equity curves
    for model_name, (results_df, _, _) in all_results.items():
        plt.plot(results_df['date'], results_df['equity'], 
                label=model_name, linewidth=2)
    
    # Calculate and plot SPY buy-and-hold benchmark
    if first_engine.prices_data is not None:
        prices_df = first_engine.prices_data.copy()
        
        # Get the actual trading dates from the first model's results
        first_results_df = next(iter(all_results.values()))[0]  # First element is results_df
        backtest_dates = first_results_df['date'].values
        
        # Filter prices to only dates in the backtest period
        prices_df = prices_df[prices_df['date'].isin(backtest_dates)]
        
        if len(prices_df) > 0:
            # Calculate buy-and-hold equity curve
            first_price = prices_df.iloc[0]['close']
            shares_bought = initial_capital / first_price
            prices_df['spy_equity'] = shares_bought * prices_df['close']
            
            # Plot SPY buy-and-hold
            plt.plot(prices_df['date'], prices_df['spy_equity'], 
                    label='SPY Buy & Hold', linewidth=2, 
                    linestyle='--', color='black', alpha=0.7)
    
    plt.axhline(y=initial_capital, color='gray', linestyle='--', 
                alpha=0.5, label='Initial Capital')
    
    plt.title('Equity Curves: All Models vs SPY Buy & Hold', fontsize=14, fontweight='bold')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Equity ($)')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_iv_comparison(all_results: Dict[str, Tuple[pd.DataFrame, BacktestEngine, pd.DataFrame]],
                      save_path: str = None):
    """
    Plot implied volatility predictions from all models over time with VIX benchmark.
    
    Args:
        all_results: Dict mapping model_name -> (results_df, engine, expiry_log)
        save_path: Path to save figure (optional)
    
    Shows how different volatility forecasting models predict IV over time
    compared to the VIX (market's volatility expectation).
    Useful for understanding model divergence and identifying regime changes.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), 
                                     gridspec_kw={'height_ratios': [2, 1]})
    
    # Load VIX data
    try:
        vix_df = pd.read_parquet(CONFIG['vix_path'])
        vix_df['date'] = pd.to_datetime(vix_df['date'])
        
        # Filter VIX to backtest date range
        vix_df = vix_df[
            (vix_df['date'] >= CONFIG['start_date']) & 
            (vix_df['date'] <= CONFIG['end_date'])
        ]
        
        # Convert VIX from percentage to decimal (VIX of 12.92 -> 0.1292)
        vix_df['vix_decimal'] = vix_df['vix_close'] / 100
        
        has_vix = True
    except Exception as e:
        print(f"Warning: Could not load VIX data: {e}")
        has_vix = False
    
    # Plot 1: Implied Volatility Comparison
    if has_vix:
        # Plot VIX first as benchmark (thicker line)
        ax1.plot(vix_df['date'], vix_df['vix_decimal'], 
                label='VIX (Market Benchmark)', linewidth=3, 
                color='black', linestyle='--', alpha=0.7, zorder=10)
    
    # Plot model predictions
    for model_name, (_, engine, _) in all_results.items():
        if engine.portfolio.iv_log:
            iv_df = pd.DataFrame(engine.portfolio.iv_log)
            iv_df['date'] = pd.to_datetime(iv_df['date'])
            ax1.plot(iv_df['date'], iv_df['implied_vol'], 
                    label=f'{model_name} (Predicted)', 
                    linewidth=2, alpha=0.8)
    
    ax1.set_title('Implied Volatility: Model Predictions vs VIX', 
                 fontsize=14, fontweight='bold')
    ax1.set_xlabel('Date', fontsize=11)
    ax1.set_ylabel('Implied Volatility (Annual)', fontsize=11)
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(alpha=0.3)
    
    # Plot 2: Difference from VIX (Model Error)
    if has_vix:
        for model_name, (_, engine, _) in all_results.items():
            if engine.portfolio.iv_log:
                iv_df = pd.DataFrame(engine.portfolio.iv_log)
                iv_df['date'] = pd.to_datetime(iv_df['date'])
                
                # Merge with VIX to calculate differences
                merged = pd.merge(iv_df, vix_df[['date', 'vix_decimal']], 
                                on='date', how='inner')
                merged['iv_diff'] = merged['implied_vol'] - merged['vix_decimal']
                
                ax2.plot(merged['date'], merged['iv_diff'] * 100,  # Convert to percentage points
                        label=f'{model_name}', linewidth=2, alpha=0.8)
        
        ax2.axhline(y=0, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
        ax2.set_title('Model Prediction Error (vs VIX)', 
                     fontsize=12, fontweight='bold')
        ax2.set_xlabel('Date', fontsize=11)
        ax2.set_ylabel('Difference from VIX (percentage points)', fontsize=11)
        ax2.legend(loc='best', fontsize=10)
        ax2.grid(alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'VIX data not available', 
                ha='center', va='center', fontsize=12, 
                transform=ax2.transAxes)
        ax2.set_title('Model Prediction Error (vs VIX) - Data Unavailable', 
                     fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

# ============================================================================
# REPORTING
# ============================================================================

def save_results_to_csv(all_results: Dict[str, Tuple[pd.DataFrame, BacktestEngine, pd.DataFrame]],
                       metrics_df: pd.DataFrame,
                       output_dir: str):
    """
    Save all results to CSV files.
    
    Args:
        all_results: Dict mapping model_name -> (results_df, engine, expiry_log)
        metrics_df: DataFrame with performance metrics
        output_dir: Directory to save files
        
    Save:
        - metrics_comparison.csv: Performance metrics table
        - {model_name}_equity.csv: Equity curve for each model
        - {model_name}_expiry_log.csv: Expiry events for each model
        - {model_name}_iv_log.csv: Daily IV predictions for each model
        - combined_results.csv: All equity curves in one file
        - combined_iv.csv: All IV predictions in one file
        
    Note: Trade logs are saved separately in trade_logs/ subdirectory 
          by run_all_backtests() function.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save metrics comparison
    metrics_path = os.path.join(output_dir, 'metrics_comparison.csv')
    metrics_df.to_csv(metrics_path)
    print(f"  Saved metrics to: {metrics_path}")
    
    # Save individual equity curves, expiry logs, and IV logs
    for model_name, (results_df, engine, _) in all_results.items():
        # Save equity curve
        equity_path = os.path.join(output_dir, f'equity_logs/{model_name}_equity.csv')
        results_df.to_csv(equity_path, index=False)
        print(f"  Saved {model_name} equity curve to: {equity_path}")
        
        # Save expiry log if available
        if engine.portfolio.expiry_log:
            expiry_df = pd.DataFrame(engine.portfolio.expiry_log)
            expiry_path = os.path.join(output_dir, f'expiry_logs/{model_name}_expiry_log.csv')
            expiry_df.to_csv(expiry_path, index=False)
            print(f"  Saved {model_name} expiry log to: {expiry_path}")
        
        # Save early exit log if available
        if engine.portfolio.early_exit_log:
            early_exit_df = pd.DataFrame(engine.portfolio.early_exit_log)
            early_exit_path = os.path.join(output_dir, f'early_exit_logs/{model_name}_early_exit_log.csv')
            os.makedirs(os.path.dirname(early_exit_path), exist_ok=True)
            early_exit_df.to_csv(early_exit_path, index=False)
            print(f"  Saved {model_name} early exit log to: {early_exit_path}")
        
        # Save IV log if available
        if engine.portfolio.iv_log:
            iv_df = pd.DataFrame(engine.portfolio.iv_log)
            iv_path = os.path.join(output_dir, f'IV_logs/{model_name}_iv_log.csv')
            iv_df.to_csv(iv_path, index=False)
            print(f"  Saved {model_name} IV log to: {iv_path}")
    
    # Save combined results (all equity curves in one file)
    combined_df = pd.DataFrame()
    for model_name, (results_df, _, _) in all_results.items():
        if combined_df.empty:
            combined_df['date'] = results_df['date']
        combined_df[f'{model_name}_equity'] = results_df['equity'].values
    
    combined_path = os.path.join(output_dir, 'combined_results.csv')
    combined_df.to_csv(combined_path, index=False)
    print(f"  Saved combined results to: {combined_path}")
    
    # Save combined IV predictions (all models' IVs side-by-side with VIX)
    combined_iv_df = pd.DataFrame()
    for model_name, (_, engine, _) in all_results.items():
        if engine.portfolio.iv_log:
            iv_df = pd.DataFrame(engine.portfolio.iv_log)
            if combined_iv_df.empty:
                combined_iv_df['date'] = iv_df['date']
                combined_iv_df['underlying_price'] = iv_df['underlying_price']
            combined_iv_df[f'{model_name}_iv'] = iv_df['implied_vol'].values
    
    if not combined_iv_df.empty:
        # Add VIX data to combined IV dataframe
        try:
            vix_df = pd.read_parquet(CONFIG['vix_path'])
            vix_df['date'] = pd.to_datetime(vix_df['date'])
            combined_iv_df['date'] = pd.to_datetime(combined_iv_df['date'])
            
            # Merge VIX data
            combined_iv_df = pd.merge(
                combined_iv_df, 
                vix_df[['date', 'vix_close']],
                on='date', 
                how='left'
            )
            
            # Convert VIX to decimal format for comparison
            combined_iv_df['vix_decimal'] = combined_iv_df['vix_close'] / 100
            
            print(f"  Added VIX data to combined IV predictions")
        except Exception as e:
            print(f"  Warning: Could not add VIX data to combined IV: {e}")
        
        combined_iv_path = os.path.join(output_dir, 'combined_iv.csv')
        combined_iv_df.to_csv(combined_iv_path, index=False)
        print(f"  Saved combined IV predictions to: {combined_iv_path}")
        append_vrp_to_combined_iv(combined_iv_path)
        print(f"Successfully appended VRP values to {combined_iv_path}")

# ============================================================================
# VRP UTILITIES
# ============================================================================

def append_vrp_to_combined_iv(combined_iv_path: str = None):
    """
    Append VRP adjustment column to combined_iv.csv.
    
    The VRP column shows what was added to RV forecasts to get IV estimates.
    To get the raw RV forecast: *_iv - vrp_adjustment
    
    Args:
        combined_iv_path: Path to combined_iv.csv (defaults to results/combined_iv.csv)
    """
    if combined_iv_path is None:
        combined_iv_path = os.path.join(CONFIG['results_dir'], 'combined_iv.csv')
    
    # Load existing combined_iv
    df = pd.read_csv(combined_iv_path)
    df['date'] = pd.to_datetime(df['date'])
    
    # Load data for VRP calculation
    prices_df = pd.read_parquet(CONFIG['prices_path'])
    vix_df = pd.read_parquet(CONFIG['vix_path'])
    
    # Create VRP calculator
    vrp_calc = VarianceRiskPremiumCalculator(
        prices_df=prices_df,
        vix_df=vix_df,
        rv_window=CONFIG['vrp_rv_window'],
        vrp_lookback=CONFIG['vrp_lookback'],
        use_regime=CONFIG['vrp_use_regime']
    )
    
    # Fit on data before first date in the IV log
    vrp_calc.fit(end_date=df['date'].min())
    
    # Calculate VRP for each date
    vrp_values = []
    for date in df['date']:
        try:
            vrp = vrp_calc.get_vrp_adjustment(pd.Timestamp(date), method=CONFIG['vrp_method'])
        except:
            vrp = np.nan
        vrp_values.append(vrp)
    
    df['vrp_adjustment'] = vrp_values
    
    # Save back
    df.to_csv(combined_iv_path, index=False)
    print(f"Added vrp_adjustment column to {combined_iv_path}")
    print(f"  Mean VRP: {np.nanmean(vrp_values):.4f} ({np.nanmean(vrp_values)*100:.2f}%)")
    print(f"  To get raw RV forecast: [model]_iv - vrp_adjustment")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main execution function.
    """
    print("="*80)
    print("OPTIONS TRADING STRATEGY COMPARISON - PHASE 5")
    print("="*80)

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

