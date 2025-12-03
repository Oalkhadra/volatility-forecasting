"""
Core Backtesting Engine and Portfolio Management.

This module handles:
1. Portfolio state (Cash, Positions, Greeks)
2. Time stepping (The "Clock")
3. Trade execution and accounting
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime

@dataclass
class Position:
    """Represents a single option or stock position."""
    symbol: str              # 'SPY' or option ticker
    quantity: float          # Positive = Long, Negative = Short
    entry_price: float
    current_price: float
    position_type: str       # 'option' or 'stock'
    
    # Option specifics (None for stock)
    strike: Optional[float] = None
    expiry: Optional[datetime] = None
    option_type: Optional[str] = None # 'call' or 'put'
    
    def market_value(self) -> float:
        """
        Calculate current market value.
        For options: quantity * price * 100
        For stock: quantity * price
        """
        multiplier = 100 if self.position_type == 'option' else 1
        return self.quantity * self.current_price * multiplier

@dataclass
class Trade:
    """Represents a trade to be executed."""
    symbol: str
    quantity: float          # + for buy, - for sell
    price: float
    trade_type: str          # 'option' or 'stock'
    timestamp: datetime
    
    # Option metadata
    strike: Optional[float] = None
    expiry: Optional[datetime] = None
    option_type: Optional[str] = None

class Portfolio:
    """
    Tracks cash, positions, and PnL over time.
    """
    
    def __init__(self, initial_capital: float = 100_000.0):
        self.cash = initial_capital
        self.initial_capital = initial_capital
        self.positions: Dict[str, Position] = {}
        self.history: List[dict] = [] # Log of daily values
        self.trade_log: List[Trade] = []
        
    def execute_trade(self, trade: Trade):
        """
        Process a trade execution.
        
        TODO: Implement trade logic
        
        Hints:
            1. Calculate cost = quantity * price * multiplier (100 for options)
            2. Update self.cash (subtract cost)
            3. Update self.positions
               - If symbol exists, update quantity and average entry price
               - If new, create Position object
               - If quantity goes to 0, remove position
        """
        pass
        
    def mark_to_market(self, current_prices: Dict[str, float], date: datetime):
        """
        Update all positions with current market prices.
        
        Args:
            current_prices: Dict mapping symbol -> price
            date: Current simulation date
            
        TODO: Implement MTM logic
        
        Hints:
            1. Loop through self.positions
            2. Update .current_price for each position
            3. Calculate total portfolio value (cash + market_value of all positions)
            4. Append to self.history: {'date': date, 'equity': total_value, 'cash': self.cash}
        """
        pass
        
    def get_portfolio_greeks(self, market_data) -> Dict[str, float]:
        """
        Calculate aggregated Delta, Gamma, Vega, Theta.
        
        TODO: Optional for Phase 4, but good for analysis.
        """
        return {'delta': 0.0, 'vega': 0.0}

class BacktestEngine:
    """
    The main event loop for the backtest.
    """
    
    def __init__(self, start_date: datetime, end_date: datetime):
        self.portfolio = Portfolio()
        self.strategy = None
        self.data = None
        
    def load_data(self, data_path: str):
        """Load and prepare market data."""
        # TODO: Load parquet file, sort by date
        pass
        
    def set_strategy(self, strategy):
        """Attach a strategy to the engine."""
        self.strategy = strategy
        
    def run(self):
        """
        Run the backtest simulation.
        
        TODO: Implement the main loop
        
        Structure:
            1. For each unique date in self.data:
            2.   Get market snapshot for that date
            3.   self.portfolio.mark_to_market(snapshot)
            4.   signals = self.strategy.generate_signals(portfolio, snapshot)
            5.   For signal in signals:
            6.      trade = self.create_trade_from_signal(signal)
            7.      self.portfolio.execute_trade(trade)
        """
        print("Starting backtest...")
        # Loop logic here
        print("Backtest complete.")


