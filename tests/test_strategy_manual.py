"""
Manual test for strategy implementation.

Run this after implementing the 3 core methods in strategy.py:
1. calculate_theo_price()
2. generate_signals()
3. create_trade_from_signal()

This provides quick validation before running full Phase 5.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
from datetime import datetime

from src.backtest.engine import BacktestEngine, Portfolio, Trade
from src.backtest.strategy import MispricingStrategy, count_option_positions

def test_calculate_theo_price():
    """Test theo price calculation with simple vol model."""
    print("\n" + "="*60)
    print("TEST 1: calculate_theo_price()")
    print("="*60)
    
    # Use simple float vol model
    strategy = MispricingStrategy(vol_model=0.20, threshold=0.10)
    
    try:
        theo_price, theo_sigma = strategy.calculate_theo_price(
            S=300,
            K=300,
            T=7/365,
            r=0.02,
            option_type='call'
        )
        
        print(f"✓ Calculated theo price: ${theo_price:.2f}")
        print(f"  (Expected: $3-4 for ATM 7-day call with 20% vol)")
        
        if 2.5 < theo_price < 5.0:
            print("✓ PASS - Price in expected range")
            return True
        else:
            print("⚠ WARNING - Price outside expected range")
            return False
            
    except NotImplementedError:
        print("❌ FAIL - Method not implemented yet")
        return False
    except Exception as e:
        print(f"❌ ERROR - {e}")
        return False


def test_generate_signals():
    """Test signal generation with mock data."""
    print("\n" + "="*60)
    print("TEST 2: generate_signals()")
    print("="*60)
    
    # Create mock options snapshot
    snapshot = pd.DataFrame({
        'act_symbol': ['SPY','SPY','SPY','SPY','SPY'],
        'strike': [295.0, 300.0, 305.0, 300.0, 305.0],
        'expiration': [pd.Timestamp('2019-06-21')] * 5,
        'call_put': ['call', 'call', 'call', 'put', 'put'],
        'bid': [7.0, 4.0, 2.0, 3.5, 5.5],
        'mid_price': [8.0, 5.0, 3.0, 4.5, 6.5],  # Some overpriced, some underpriced
        'ask': [9.0, 6.0, 4.0, 5.5, 7.5],
        'underlying_price': [300.0] * 5,
        'days_to_expiry': [7] * 5
    })
    
    strategy = MispricingStrategy(vol_model=0.20, threshold=0.10, max_positions=5)
    portfolio = Portfolio(initial_capital=100000)
    
    try:
        signals = strategy.generate_signals(
            portfolio=portfolio,
            options_snapshot=snapshot,
            underlying_price=300.0,
            current_date=pd.Timestamp('2019-06-14'),
            risk_free_rate=0.02
        )
        
        print(f"✓ Generated {len(signals)} signals")
        
        if len(signals) > 0:
            for i, signal in enumerate(signals[:3]):  # Show first 3
                print(f"\n  Signal {i+1}:")
                print(f"    {signal.action.upper()} {signal.option_type.upper()} ${signal.strike:.0f}")
                print(f"    Market: ${signal.market_price:.2f} | Theo: ${signal.theo_price:.2f}")
                print(f"    Mispricing: {signal.mispricing_pct:.2%}")
                print(f"    Delta: {signal.delta:.3f} | Hedge: {signal.hedge_quantity:.0f} shares")
            
            print(f"\n✓ PASS - Signals generated successfully")
            return True
        else:
            print("⚠ WARNING - No signals generated (threshold too high?)")
            return False
            
    except NotImplementedError:
        print("❌ FAIL - Method not implemented yet")
        return False
    except Exception as e:
        print(f"❌ ERROR - {e}")
        import traceback
        traceback.print_exc()
        return False


def test_create_trade():
    """Test trade creation from signal."""
    print("\n" + "="*60)
    print("TEST 3: create_trade_from_signal()")
    print("="*60)
    
    from src.backtest.strategy import TradeSignal
    
    # Create mock signal
    signal = TradeSignal(
        symbol='SPY',
        action='buy',
        quantity=1,
        market_price=5.0,
        theo_price=4.5,
        mispricing_pct=-0.10,
        strike=300.0,
        expiry=datetime(2019, 6, 21),
        option_type='call',
        delta=0.55,
        hedge_quantity=-55.0,
        timestamp=datetime(2019, 6, 14),
        underlying_price=300.0,
        dte=7
    )
    
    strategy = MispricingStrategy(vol_model=0.20)
    
    try:
        # Test option trade
        option_trade = strategy.create_trade_from_signal(signal, 'option')
        print(f"✓ Option trade created:")
        print(f"    {option_trade.quantity} {option_trade.symbol} "
              f"${option_trade.strike:.0f} {option_trade.option_type} "
              f"@ ${option_trade.price:.2f}")
        
        # Test hedge trade
        hedge_trade = strategy.create_trade_from_signal(signal, 'stock')
        print(f"✓ Hedge trade created:")
        print(f"    {hedge_trade.quantity:.0f} shares {hedge_trade.symbol} "
              f"@ ${hedge_trade.price:.2f}")
        
        print(f"\n✓ PASS - Trades created successfully")
        return True
        
    except NotImplementedError:
        print("❌ FAIL - Method not implemented yet")
        return False
    except Exception as e:
        print(f"❌ ERROR - {e}")
        import traceback
        traceback.print_exc()
        return False


def test_mini_backtest():
    """Run a mini backtest on real data."""
    print("\n" + "="*60)
    print("TEST 4: Mini Backtest (June 2019)")
    print("="*60)
    
    try:
        # Create engine
        engine = BacktestEngine(
            start_date=datetime(2019, 6, 1),
            end_date=datetime(2019, 9, 30),
            initial_capital=100000
        )
        
        # Load data
        print("Loading data...")
        engine.load_data(
            options_path='data/processed/spy_options.parquet',
            prices_path='data/processed/spy_prices.parquet',
            rates_path='data/processed/risk_free_rate.parquet'
        )
        
        # Create strategy
        strategy = MispricingStrategy(
            vol_model=0.20,  # Simple 20% vol
            threshold=0.10,
            max_positions=5,
            contracts_per_trade=1
        )
        
        # Run backtest
        print("Running backtest...")
        engine.set_strategy(strategy)
        results = engine.run()
        
        # Check results
        print(f"\n✓ Backtest completed!")
        print(f"  Results shape: {results.shape}")
        print(f"  Initial equity: ${results.iloc[0]['equity']:,.2f}")
        print(f"  Final equity: ${results.iloc[-1]['equity']:,.2f}")
        print(f"  Return: {(results.iloc[-1]['equity'] / results.iloc[0]['equity'] - 1) * 100:.2f}%")
        print(f"  Trades executed: {len(engine.portfolio.trade_log)}")
        print(f"  Open positions: {len(engine.portfolio.positions)}")
        
        if len(engine.portfolio.trade_log) > 0:
            print(f"\n  Sample trades:")
            for i, trade in enumerate(engine.portfolio.trade_log[:3]):
                print(f"    {i+1}. {trade.trade_type}: {trade.quantity} @ ${trade.price:.2f}")
        
        print(f"\n✓ PASS - Mini backtest successful")
        return True
        
    except FileNotFoundError as e:
        print(f"⚠ SKIP - Data files not found: {e}")
        print(f"  (This is expected if you haven't downloaded data yet)")
        return None
    except NotImplementedError:
        print("❌ FAIL - Strategy methods not implemented yet")
        return False
    except Exception as e:
        print(f"❌ ERROR - {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("="*60)
    print("STRATEGY IMPLEMENTATION TEST SUITE")
    print("="*60)
    print("\nThis tests your implementation of strategy.py")
    print("Complete all tests before moving to Phase 5")
    
    results = []
    
    # Run tests
    results.append(("calculate_theo_price", test_calculate_theo_price()))
    results.append(("generate_signals", test_generate_signals()))
    results.append(("create_trade_from_signal", test_create_trade()))
    results.append(("mini_backtest", test_mini_backtest()))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, result in results if result is True)
    failed = sum(1 for _, result in results if result is False)
    skipped = sum(1 for _, result in results if result is None)
    
    for name, result in results:
        if result is True:
            status = "✓ PASS"
        elif result is False:
            status = "❌ FAIL"
        else:
            status = "⚠ SKIP"
        print(f"{status} - {name}")
    
    print(f"\nResults: {passed} passed, {failed} failed, {skipped} skipped")
    
    if failed == 0 and passed > 0:
        print("\n🎉 All tests passed! Ready for Phase 5!")
        print("\nNext step: Run full comparison")
        print("  python scripts/run_backtest_comparison.py")
    elif failed > 0:
        print("\n⚠ Some tests failed. Review implementation and try again.")
    else:
        print("\n⚠ All tests skipped. Implement strategy methods first.")
    
    print("="*60)


if __name__ == "__main__":
    main()

