#!/usr/bin/env python3
"""
Validation script for trading pipeline data integrity.
Cross-checks trade logs, equity curves, and expiry logs.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from collections import defaultdict

def load_data():
    """Load all three data files."""
    trade_log = pd.read_csv('results/trade_logs/GARCH_trade_log.csv')
    equity = pd.read_csv('results/GARCH_equity.csv')
    expiry_log = pd.read_csv('results/GARCH_expiry_log.csv')
    
    # Convert dates to datetime
    trade_log['timestamp'] = pd.to_datetime(trade_log['timestamp'])
    equity['date'] = pd.to_datetime(equity['date'])
    expiry_log['date'] = pd.to_datetime(expiry_log['date'])
    
    return trade_log, equity, expiry_log


def validate_trade_log_structure(trade_log):
    """Validate basic structure of trade log."""
    print("=" * 80)
    print("VALIDATION 1: Trade Log Structure")
    print("=" * 80)
    
    issues = []
    
    # Check for missing values in critical columns
    critical_cols = ['symbol', 'quantity', 'price', 'trade_type', 'timestamp']
    for col in critical_cols:
        missing = trade_log[col].isna().sum()
        if missing > 0:
            issues.append(f"  ❌ {missing} missing values in {col}")
    
    # Check that options have required fields
    option_trades = trade_log[trade_log['trade_type'] == 'option']
    for col in ['strike', 'expiry', 'option_type', 'implied_vol']:
        missing = option_trades[col].isna().sum()
        if missing > 0:
            issues.append(f"  ❌ {missing} option trades missing {col}")
    
    # Check that stock trades don't have option fields
    stock_trades = trade_log[trade_log['trade_type'] == 'stock']
    if not stock_trades[['strike', 'expiry', 'option_type']].isna().all().all():
        issues.append(f"  ❌ Stock trades have option fields filled")
    
    if issues:
        print("\n".join(issues))
        return False
    else:
        print("✓ Trade log structure is valid")
        print(f"  - Total trades: {len(trade_log)}")
        print(f"  - Option trades: {len(option_trades)}")
        print(f"  - Stock trades: {len(stock_trades)}")
        return True


def calculate_cash_flows(trade_log):
    """Calculate cash flows from trades."""
    print("\n" + "=" * 80)
    print("VALIDATION 2: Cash Flow Calculations")
    print("=" * 80)
    
    trade_log = trade_log.copy()
    trade_log['date'] = trade_log['timestamp'].dt.date
    
    # Calculate cash flow for each trade
    # Options: buying = -premium paid (quantity * price * 100)
    # Stock: buying = -cost, selling = +proceeds
    cash_flows = []
    
    for idx, trade in trade_log.iterrows():
        if trade['trade_type'] == 'option':
            # Options are always bought (positive quantity), cost is premium * 100
            cash_flow = -trade['quantity'] * trade['price'] * 100
        else:  # stock
            # Positive quantity = buy (negative cash), negative quantity = sell (positive cash)
            cash_flow = -trade['quantity'] * trade['price']
        
        cash_flows.append({
            'date': trade['date'],
            'trade_type': trade['trade_type'],
            'cash_flow': cash_flow,
            'quantity': trade['quantity']
        })
    
    cash_df = pd.DataFrame(cash_flows)
    daily_cash = cash_df.groupby('date')['cash_flow'].sum()
    
    print(f"✓ Calculated cash flows for {len(trade_log)} trades")
    print(f"  - Total cash outflow: ${cash_df[cash_df['cash_flow'] < 0]['cash_flow'].sum():,.2f}")
    print(f"  - Total cash inflow: ${cash_df[cash_df['cash_flow'] > 0]['cash_flow'].sum():,.2f}")
    print(f"  - Net cash flow: ${cash_df['cash_flow'].sum():,.2f}")
    
    return daily_cash, cash_df


def validate_position_tracking(trade_log):
    """Validate that positions are tracked correctly."""
    print("\n" + "=" * 80)
    print("VALIDATION 3: Position Tracking")
    print("=" * 80)
    
    trade_log = trade_log.copy()
    trade_log['date'] = trade_log['timestamp'].dt.date
    
    # Track positions by date
    positions_by_date = {}
    current_positions = []
    
    for date in sorted(trade_log['date'].unique()):
        day_trades = trade_log[trade_log['date'] == date].copy()
        
        # Process each trade
        for idx, trade in day_trades.iterrows():
            if trade['trade_type'] == 'option':
                # Add new option position
                current_positions.append({
                    'type': 'option',
                    'strike': trade['strike'],
                    'expiry': pd.to_datetime(trade['expiry']).date(),
                    'option_type': trade['option_type'],
                    'entry_date': date
                })
        
        # Remove expired positions
        current_positions = [p for p in current_positions 
                           if p['type'] != 'option' or p['expiry'] > date]
        
        positions_by_date[date] = len(current_positions)
    
    print(f"✓ Tracked positions across {len(positions_by_date)} trading days")
    print(f"  - Max positions: {max(positions_by_date.values())}")
    print(f"  - Average positions: {np.mean(list(positions_by_date.values())):.1f}")
    
    return positions_by_date


def validate_expiry_events(trade_log, expiry_log):
    """Validate expiry events against trade log."""
    print("\n" + "=" * 80)
    print("VALIDATION 4: Expiry Event Validation")
    print("=" * 80)
    
    issues = []
    
    # Get all option trades
    option_trades = trade_log[trade_log['trade_type'] == 'option'].copy()
    option_trades['expiry_date'] = pd.to_datetime(option_trades['expiry']).dt.date
    
    # Group by expiry date
    expiries_by_date = option_trades.groupby('expiry_date').size()
    
    # Check each expiry event
    for idx, expiry in expiry_log.iterrows():
        expiry_date = expiry['date'].date()
        num_expired = expiry['num_expired']
        
        # Count options expiring on this date that were opened before
        expected_expired = expiries_by_date.get(expiry_date, 0)
        
        if expected_expired != num_expired:
            issues.append(f"  ⚠️  {expiry_date}: Expected {expected_expired} expirations, got {num_expired}")
    
    # Check for missing expiry events
    for exp_date in expiries_by_date.index:
        if exp_date not in expiry_log['date'].dt.date.values:
            issues.append(f"  ❌ {exp_date}: Missing expiry event for {expiries_by_date[exp_date]} options")
    
    if issues:
        print(f"Found {len(issues)} expiry discrepancies:")
        print("\n".join(issues[:10]))  # Show first 10
    else:
        print(f"✓ All {len(expiry_log)} expiry events validated")
        print(f"  - Total options expired: {expiry_log['num_expired'].sum()}")
        print(f"  - Total expiry PnL: ${expiry_log['pnl'].sum():,.2f}")
    
    return len(issues) == 0


def validate_equity_calculations(trade_log, equity):
    """Validate equity calculations."""
    print("\n" + "=" * 80)
    print("VALIDATION 5: Equity Curve Validation")
    print("=" * 80)
    
    issues = []
    
    # Check starting equity
    initial_equity = equity.iloc[0]['equity']
    initial_cash = equity.iloc[0]['cash']
    
    print(f"  Starting equity: ${initial_equity:,.2f}")
    print(f"  Starting cash: ${initial_cash:,.2f}")
    
    # Check for negative cash (concerning but not necessarily wrong)
    negative_cash_days = equity[equity['cash'] < 0]
    if len(negative_cash_days) > 0:
        print(f"  ⚠️  {len(negative_cash_days)} days with negative cash")
        print(f"     Most negative: ${negative_cash_days['cash'].min():,.2f} on {negative_cash_days.iloc[negative_cash_days['cash'].argmin()]['date'].date()}")
    
    # Check equity trends
    final_equity = equity.iloc[-1]['equity']
    pnl = final_equity - initial_equity
    return_pct = (pnl / initial_equity) * 100
    
    print(f"\n  Final equity: ${final_equity:,.2f}")
    print(f"  Total P&L: ${pnl:,.2f} ({return_pct:+.2f}%)")
    
    # Check for big equity jumps (could indicate issues)
    equity_pct_change = equity['equity'].pct_change()
    large_moves = equity_pct_change[abs(equity_pct_change) > 0.2]  # >20% move
    if len(large_moves) > 0:
        print(f"  ⚠️  {len(large_moves)} days with >20% equity moves")
        for idx in large_moves.index[:3]:
            date = equity.iloc[idx]['date'].date()
            change = large_moves.iloc[large_moves.index.get_loc(idx)] * 100
            print(f"     {date}: {change:+.1f}%")
    
    return len(issues) == 0


def validate_position_counts(trade_log, equity):
    """Validate that position counts in equity match actual positions."""
    print("\n" + "=" * 80)
    print("VALIDATION 6: Position Count Validation")
    print("=" * 80)
    
    trade_log = trade_log.copy()
    trade_log['date'] = trade_log['timestamp'].dt.date
    
    # Calculate actual position counts
    positions = []
    issues = []
    
    for eq_date in equity['date'].dt.date:
        # Get all options opened before or on this date
        options_opened = trade_log[
            (trade_log['trade_type'] == 'option') & 
            (trade_log['date'] <= eq_date)
        ].copy()
        
        if len(options_opened) > 0:
            options_opened['expiry_date'] = pd.to_datetime(options_opened['expiry']).dt.date
            
            # Count options that haven't expired yet
            active_options = options_opened[options_opened['expiry_date'] > eq_date]
            actual_count = len(active_options)
        else:
            actual_count = 0
        
        # Get reported count
        reported_count = equity[equity['date'].dt.date == eq_date]['num_positions'].iloc[0]
        
        if actual_count != reported_count:
            issues.append({
                'date': eq_date,
                'actual': actual_count,
                'reported': reported_count,
                'diff': actual_count - reported_count
            })
    
    if issues:
        print(f"❌ Found {len(issues)} days with position count mismatches:")
        for issue in issues[:10]:  # Show first 10
            print(f"  {issue['date']}: Actual={issue['actual']}, Reported={issue['reported']}, Diff={issue['diff']}")
        return False
    else:
        print(f"✓ All {len(equity)} days have correct position counts")
        return True


def generate_summary_report(trade_log, equity, expiry_log):
    """Generate overall summary."""
    print("\n" + "=" * 80)
    print("SUMMARY REPORT")
    print("=" * 80)
    
    # Trading period
    start_date = trade_log['timestamp'].min().date()
    end_date = trade_log['timestamp'].max().date()
    trading_days = len(equity)
    
    print(f"\nTrading Period: {start_date} to {end_date}")
    print(f"Trading Days: {trading_days}")
    
    # Performance
    initial_equity = equity.iloc[0]['equity']
    final_equity = equity.iloc[-1]['equity']
    total_return = (final_equity / initial_equity - 1) * 100
    
    print(f"\nPerformance:")
    print(f"  Initial Equity: ${initial_equity:,.2f}")
    print(f"  Final Equity: ${final_equity:,.2f}")
    print(f"  Total Return: {total_return:+.2f}%")
    
    # Trading activity
    option_trades = trade_log[trade_log['trade_type'] == 'option']
    stock_trades = trade_log[trade_log['trade_type'] == 'stock']
    
    print(f"\nTrading Activity:")
    print(f"  Total Trades: {len(trade_log)}")
    print(f"  Option Purchases: {len(option_trades)}")
    print(f"  Stock Hedges: {len(stock_trades)}")
    print(f"  Expiry Events: {len(expiry_log)}")
    
    # Cash analysis
    final_cash = equity.iloc[-1]['cash']
    print(f"\nCash Analysis:")
    print(f"  Final Cash: ${final_cash:,.2f}")
    print(f"  Min Cash: ${equity['cash'].min():,.2f}")
    print(f"  Max Cash: ${equity['cash'].max():,.2f}")


def main():
    """Main validation routine."""
    print("\n" + "=" * 80)
    print("TRADING PIPELINE DATA VALIDATION")
    print("=" * 80)
    
    # Load data
    print("\nLoading data files...")
    trade_log, equity, expiry_log = load_data()
    print(f"✓ Loaded {len(trade_log)} trades, {len(equity)} equity records, {len(expiry_log)} expiry events")
    
    # Run validations
    validations = []
    
    validations.append(("Trade Log Structure", validate_trade_log_structure(trade_log)))
    
    daily_cash, cash_df = calculate_cash_flows(trade_log)
    
    positions_by_date = validate_position_tracking(trade_log)
    
    validations.append(("Expiry Events", validate_expiry_events(trade_log, expiry_log)))
    
    validations.append(("Equity Calculations", validate_equity_calculations(trade_log, equity)))
    
    validations.append(("Position Counts", validate_position_counts(trade_log, equity)))
    
    # Summary
    generate_summary_report(trade_log, equity, expiry_log)
    
    # Final verdict
    print("\n" + "=" * 80)
    print("VALIDATION RESULTS")
    print("=" * 80)
    
    for name, passed in validations:
        status = "✓ PASS" if passed else "❌ FAIL"
        print(f"{status}: {name}")
    
    all_passed = all(result for _, result in validations)
    
    print("\n" + "=" * 80)
    if all_passed:
        print("✓✓✓ ALL VALIDATIONS PASSED ✓✓✓")
    else:
        print("❌ SOME VALIDATIONS FAILED - REVIEW ISSUES ABOVE")
    print("=" * 80 + "\n")
    
    return all_passed


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)

