#!/usr/bin/env python3
"""
Detailed diagnostic for position counting discrepancies.
"""

import pandas as pd

# Load data
trade_log = pd.read_csv('results/trade_logs/GARCH_trade_log.csv')
equity = pd.read_csv('results/equity_logs/GARCH_equity.csv')

trade_log['timestamp'] = pd.to_datetime(trade_log['timestamp'])
trade_log['date'] = trade_log['timestamp'].dt.date
equity['date'] = pd.to_datetime(equity['date'])

# Pick a few sample dates to investigate
sample_dates = equity['date'].dt.date[:10]

print("=" * 80)
print("DETAILED POSITION COUNT ANALYSIS")
print("=" * 80)

for eq_date in sample_dates:
    print(f"\n📅 {eq_date}")
    
    # Get all options opened before or on this date
    options_opened = trade_log[
        (trade_log['trade_type'] == 'option') & 
        (trade_log['date'] <= eq_date)
    ].copy()
    
    if len(options_opened) > 0:
        options_opened['expiry_date'] = pd.to_datetime(options_opened['expiry']).dt.date
        
        # Count options that haven't expired yet
        active_options = options_opened[options_opened['expiry_date'] > eq_date]
        
        print(f"   Options opened by this date: {len(options_opened)}")
        print(f"   Active (not expired): {len(active_options)}")
        
        if len(active_options) > 0:
            print(f"   Active positions:")
            for _, opt in active_options.iterrows():
                print(f"      - {opt['option_type'].upper()} ${opt['strike']} exp {opt['expiry_date']}")
    else:
        active_options = pd.DataFrame()
        print(f"   No options opened yet")
    
    # Get reported count
    reported = equity[equity['date'].dt.date == eq_date]['num_positions'].iloc[0]
    actual = len(active_options) if len(active_options) > 0 else 0
    
    # Check for stock positions
    stock_trades = trade_log[
        (trade_log['trade_type'] == 'stock') & 
        (trade_log['date'] <= eq_date)
    ]
    net_stock = stock_trades['quantity'].sum() if len(stock_trades) > 0 else 0
    
    print(f"\n   Net stock position: {net_stock:.2f} shares")
    print(f"   Actual option positions: {actual}")
    print(f"   Reported positions: {reported}")
    print(f"   Difference: {reported - actual}")
    
    if net_stock != 0:
        print(f"   → Stock position might be counted as 1 position")

print("\n" + "=" * 80)
print("HYPOTHESIS: Are stock positions being counted?")
print("=" * 80)

# Check if difference equals 1 when stock is held
for eq_date in equity['date'].dt.date[:20]:
    options_count = len(trade_log[
        (trade_log['trade_type'] == 'option') & 
        (trade_log['date'] <= eq_date) &
        (pd.to_datetime(trade_log['expiry']).dt.date > eq_date)
    ])
    
    reported = equity[equity['date'].dt.date == eq_date]['num_positions'].iloc[0]
    stock_trades = trade_log[
        (trade_log['trade_type'] == 'stock') & 
        (trade_log['date'] <= eq_date)
    ]
    has_stock = len(stock_trades) > 0
    
    if has_stock and (reported - options_count) == 1:
        print(f"{eq_date}: Options={options_count}, Reported={reported}, Has stock=Yes → Difference=1 ✓")
    elif not has_stock and reported == options_count:
        print(f"{eq_date}: Options={options_count}, Reported={reported}, Has stock=No → Match ✓")
    else:
        print(f"{eq_date}: Options={options_count}, Reported={reported}, Has stock={has_stock} → UNEXPECTED")

