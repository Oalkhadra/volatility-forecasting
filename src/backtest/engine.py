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
        
    def execute_trade(self, trade: Trade):
        """
        Process a trade execution.
        
        TODO: Implement trade execution
        
        Hints:
            1. Calculate cost (use trade.get_key() for position lookup)
            2. Update cash
            3. Update or create position
            4. Log trade
        """
        pass
        
    def mark_to_market(self, current_prices: Dict[str, float], date: datetime):
        """
        Update all positions with current market prices.
        
        Args:
            current_prices: Dict mapping position_key -> price
            date: Current simulation date
            
        TODO: Implement MTM
        
        Hints:
            1. For each position, update current_price if available
            2. Calculate total equity = cash + sum of all position values
            3. Log to history
        """
        pass
    
    def handle_expiries(self, current_date: datetime, underlying_price: float) -> float:
        """
        Handle option expiries and realize PnL.
        
        Args:
            current_date: Current date
            underlying_price: SPY price for settlement
            
        Returns:
            Total PnL realized from expiries
            
        TODO: Implement expiry logic
        
        Hints:
            1. Find all expired options (use pos.is_expired())
            2. For each expired option:
               - Calculate intrinsic value
               - If long: cash += quantity * intrinsic * 100
               - If short: cash -= abs(quantity) * intrinsic * 100
               - Remove position
            3. Return total PnL
        """
        pass
    
    def get_portfolio_delta(self) -> float:
        """
        Calculate total portfolio delta.
        
        TODO: OPTIONAL - For monitoring delta-neutrality
        """
        return 0.0

class BacktestEngine:
    """
    The main event loop for weekly backtesting.
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
        
    def load_data(self, options_path: str, prices_path: str):
        """
        Load market data.
        
        TODO: Implement data loading
        
        Hints:
            1. Load options parquet (spy_options.parquet)
            2. Load prices parquet (spy_prices.parquet)
            3. Filter by date range
            4. Sort by date
        """
        pass
        
    def set_strategy(self, strategy):
        """Attach a strategy to the engine."""
        self.strategy = strategy
        
    def run(self):
        """
        Run the backtest simulation.
        
        TODO: Implement main backtest loop
        
        Structure:
            1. Get unique trading dates (Saturdays/Sundays)
            2. For each date:
               a. Handle expiries from previous week
               b. Get market snapshot (options + underlying price)
               c. Generate signals from strategy
               d. Execute trades
               e. Mark to market
            3. Return results DataFrame
        """
        print("Starting backtest...")
        
        # TODO: Implement loop
        
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
