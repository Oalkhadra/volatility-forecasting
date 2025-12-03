"""
Core Backtesting Engine and Portfolio Management.

This module handles:
1. Portfolio state (Cash, Positions, Greeks)
2. Weekly time stepping
3. Trade execution and accounting
4. Option expiry handling
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from dateutil.relativedelta import relativedelta
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ..pricing.pricer import black_scholes_price
from ..pricing.greeks import calculate_all_greeks

def create_option_key(symbol: str, strike: float, expiry: datetime, option_type: str) -> str:
    """Create unique identifier for an option."""
    return f"{symbol}_{strike:.0f}{option_type[0].upper()}_{expiry.strftime('%Y%m%d')}"

@dataclass
class Position:
    """Represents a single option or stock position."""
    symbol: str
    quantity: float          # Positive = Long, Negative = Short
    entry_price: float
    current_price: float
    position_type: str       # 'option' or 'stock'
    
    # Option specifics (None for stock)
    strike: Optional[float] = None
    expiry: Optional[datetime] = None
    option_type: Optional[str] = None
    
    def market_value(self) -> float:
        """Calculate current market value."""
        multiplier = 100 if self.position_type == 'option' else 1
        return self.quantity * self.current_price * multiplier
    
    def is_expired(self, current_date: datetime) -> bool:
        """Check if option has expired."""
        if self.position_type != 'option' or self.expiry is None:
            return False
        return current_date >= self.expiry
    
    def intrinsic_value(self, underlying_price: float) -> float:
        """Calculate intrinsic value at expiry."""
        if self.position_type != 'option':
            return 0.0
        
        if self.option_type == 'call':
            return max(0, underlying_price - self.strike)
        else:  # put
            return max(0, self.strike - underlying_price)

@dataclass
class Trade:
    """Represents a trade to be executed."""
    symbol: str
    quantity: float
    price: float
    trade_type: str
    timestamp: datetime
    
    strike: Optional[float] = None
    expiry: Optional[datetime] = None
    option_type: Optional[str] = None
    
    def get_key(self) -> str:
        """Get unique position key for this trade."""
        if self.trade_type == 'option':
            return create_option_key(self.symbol, self.strike, self.expiry, self.option_type)
        return self.symbol

class Portfolio:
    """Tracks cash, positions, and PnL over time."""
    
    def __init__(self, initial_capital: float = 100000.0):
        self.cash = initial_capital
        self.initial_capital = initial_capital
        self.positions: Dict[str, Position] = {}
        self.history: List[dict] = []
        self.trade_log: List[Trade] = []
        
    def execute_trade(self, trade: Trade, apply_costs: bool = True):
        """
        Process a trade execution with transaction costs.
        
        Args:
            trade: Trade to execute
            apply_costs: Whether to apply transaction costs
        """
        # Calculate base cost
        multiplier = 100 if trade.trade_type == 'option' else 1
        base_cost = trade.quantity * trade.price * multiplier
        
        # Transaction costs
        transaction_cost = 0.0
        if apply_costs:
            if trade.trade_type == 'option':
                # $0.50 per contract
                transaction_cost = abs(trade.quantity) * 0.50
            else:  # stock
                # 1 bip (0.01%)
                transaction_cost = abs(trade.quantity * trade.price) * 0.0001
        
        # Update cash (buying = negative, selling = positive)
        self.cash -= (base_cost + transaction_cost)
        
        # Log trade
        self.trade_log.append(trade)
        
        # Get position key
        pos_key = trade.get_key()
        
        # Update or create position
        if pos_key in self.positions:
            pos = self.positions[pos_key]
            
            # Weighted average cost basis (only if adding to position)
            is_same_direction = (pos.quantity > 0 and trade.quantity > 0) or \
                                (pos.quantity < 0 and trade.quantity < 0)
            
            if is_same_direction:
                total_qty = pos.quantity + trade.quantity
                if abs(total_qty) > 1e-9:
                    current_value = pos.quantity * pos.entry_price
                    new_value = trade.quantity * trade.price
                    pos.entry_price = (current_value + new_value) / total_qty
            
            pos.quantity += trade.quantity
            pos.current_price = trade.price
            
            # Remove if position closed
            if abs(pos.quantity) < 1e-9:
                del self.positions[pos_key]
        else:
            # New position
            new_pos = Position(
                symbol=trade.symbol,
                quantity=trade.quantity,
                entry_price=trade.price,
                current_price=trade.price,
                position_type=trade.trade_type,
                strike=trade.strike,
                expiry=trade.expiry,
                option_type=trade.option_type
            )
            self.positions[pos_key] = new_pos
        
    def mark_to_market(self, current_prices: Dict[str, float], date: datetime):
        """
        Update all positions with current market prices.
        
        Args:
            current_prices: Dict mapping position_key -> price
            date: Current simulation date
        """
        total_equity = self.cash
        
        for pos_key, pos in self.positions.items():
            if pos_key in current_prices:
                pos.current_price = current_prices[pos_key]
            # If not in current_prices, keep last known price
            
            total_equity += pos.market_value()
        
        # Log snapshot
        self.history.append({
            'date': date,
            'equity': total_equity,
            'cash': self.cash,
            'num_positions': len(self.positions)
        })
    
    def handle_expiries(self, current_date: datetime, underlying_price: float) -> float:
        """
        Handle option expiries and realize PnL.
        
        Args:
            current_date: Current date
            underlying_price: SPY price for settlement
            
        Returns:
            Total PnL realized from expiries
        """
        total_pnl = 0.0
        expired_keys = []
        
        for pos_key, pos in self.positions.items():
            if pos.is_expired(current_date):
                # Calculate intrinsic value
                intrinsic = pos.intrinsic_value(underlying_price)
                
                # Settlement cash flow
                # Long: receive intrinsic value
                # Short: pay intrinsic value
                settlement = pos.quantity * intrinsic * 100
                self.cash += settlement
                
                # Calculate PnL (settlement - cost basis)
                cost_basis = pos.quantity * pos.entry_price * 100
                pnl = settlement - cost_basis
                total_pnl += pnl
                
                # Mark for removal
                expired_keys.append(pos_key)
        
        # Remove expired positions
        for key in expired_keys:
            del self.positions[key]
        
        return total_pnl
    
    def get_portfolio_delta(self, S: float, r: float, sigma: float, current_date: datetime) -> float:
        """
        Calculate total portfolio delta.
        
        Args:
            S: Current underlying price
            r: Risk-free rate
            
        Returns:
            Total portfolio delta
        """
        total_delta = 0.0
        
        for pos_key, pos in self.positions.items():
            if pos.position_type == 'option':
                # Calculate time to expiry in years
                T = max(0, relativedelta(pos.expiry, current_date).years)
                
                # Get option delta
                greeks = calculate_all_greeks(
                    S=S,
                    K=pos.strike,
                    T=T,
                    r=r,
                    sigma=sigma,  # Volatility estimated from model? Need to confirm which vol to use. Can calculate IV given data for more thoroughness 
                    option_type=pos.option_type
                )
                
                # Delta contribution (per contract = 100 shares)
                total_delta += pos.quantity * greeks['delta'] * 100
            else:
                # Stock position
                total_delta += pos.quantity
        
        return total_delta

class BacktestEngine:
    """
    The main event loop for weekly options trading with daily delta rebalancing.
    """
    
    def __init__(self, 
                 start_date: datetime, 
                 end_date: datetime,
                 initial_capital: float = 100000.0):
        self.start_date = start_date
        self.end_date = end_date
        self.portfolio = Portfolio(initial_capital)
        self.strategy = None
        self.options_data = None
        self.prices_data = None
        self.rates_data = None
        
    def load_data(self, options_path: str, prices_path: str, rates_path: str):
        """
        Load market data.

        """
        # Read data, already sorted and filtered by date when stored through preprocess.py
        options_df = pd.read_parquet(options_path)        
        prices_df = pd.read_parquet(prices_path)
        rfr_df = pd.read_parquet(rates_path)

        self.options_data = options_df
        self.prices_data = prices_df
        self.rates_data = rfr_df

    def set_strategy(self, strategy):
        """Attach a strategy to the engine."""
        self.strategy = strategy
    
    def rebalance_delta(self, current_date: datetime, underlying_price: float, r: float):
        """
        Rebalance delta hedge to maintain neutrality.
        
        Args:
            current_date: Current date
            underlying_price: Current SPY price
            r: Risk-free rate
            
        TODO: Implement delta rebalancing
        
        Hints:
            1. Calculate current portfolio delta (sum of all option deltas)
            2. Calculate required stock hedge = -portfolio_delta
            3. Find current stock position (if any)
            4. Calculate rebalance quantity = required_hedge - current_stock
            5. Execute stock trade if rebalance_qty != 0
        """
        pass
        
    def run(self):
        """
        Run the backtest simulation with weekly trading and daily rebalancing.
        
        TODO: Implement main backtest loop
        
        Structure:
            OUTER LOOP (Weekly - Saturdays/Sundays):
                1. Handle expiries from previous week
                2. Get options snapshot for this week
                3. Generate signals from strategy
                4. Execute option trades
                5. Execute initial delta hedges
                
            INNER LOOP (Daily - Mon-Fri):
                1. Get current underlying price
                2. Mark to market all positions
                3. Rebalance delta hedge
                4. Log daily PnL
        
        Returns:
            DataFrame with backtest results
        """
        print("Starting backtest...")
        
        # TODO: Implement loop structure
        # Pseudocode:
        # 
        # trading_weeks = get_unique_saturdays(self.options_data)
        # 
        # for week_date in trading_weeks:
        #     # Weekly: New option positions
        #     underlying_price = get_price(week_date)
        #     expiry_pnl = self.portfolio.handle_expiries(week_date, underlying_price)
        #     
        #     options_snapshot = self.options_data[self.options_data['date'] == week_date]
        #     signals = self.strategy.generate_signals(
        #         self.portfolio, options_snapshot, underlying_price, week_date
        #     )
        #     
        #     # Execute option trades
        #     for signal in signals:
        #         option_trade = create_trade_from_signal(signal, 'option')
        #         self.portfolio.execute_trade(option_trade)
        #         
        #         # Initial delta hedge
        #         hedge_trade = create_trade_from_signal(signal, 'stock')
        #         self.portfolio.execute_trade(hedge_trade)
        #     
        #     # Daily: Rebalance loop
        #     daily_dates = get_daily_dates_in_week(week_date)
        #     for day in daily_dates:
        #         underlying_price = get_price(day)
        #         self.portfolio.mark_to_market(prices_dict, day)
        #         self.rebalance_delta(day, underlying_price, r=0.02)
        
        print("Backtest complete.")
        return pd.DataFrame(self.portfolio.history)


if __name__ == "__main__":
    # Test the Portfolio class
    print("="*60)
    print("PORTFOLIO TEST")
    print("="*60)
    
    portfolio = Portfolio(initial_capital=100000)
    print(f"Initial capital: ${portfolio.cash:,.2f}")
    
    # Test trade execution
    test_trade = Trade(
        symbol='SPY',
        quantity=10,
        price=290.0,
        trade_type='option',
        timestamp=datetime(2019, 6, 8),
        strike=290.0,
        expiry=datetime(2019, 6, 21),
        option_type='call'
    )
    
    print(f"\nExecuting test trade: {test_trade.get_key()}")
    portfolio.execute_trade(test_trade)
    
    print(f"Cash after trade: ${portfolio.cash:,.2f}")
    print(f"Positions: {len(portfolio.positions)}")
