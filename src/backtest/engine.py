import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional
from datetime import datetime
import sys
import os
from tqdm import tqdm
import glob


# Add parent directory to path for absolute imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pricing.greeks import calculate_all_greeks, implied_volatility

def create_option_key(symbol: str, strike: float, expiry: datetime, option_type: str) -> str:
    """Create unique identifier for an option for logging."""
    return f"{symbol}_{strike:.0f}{option_type[0].upper()}_{expiry.strftime('%Y%m%d')}"

@dataclass
class Position:
    """Represents a single option or stock position."""
    symbol: str
    quantity: float        
    entry_price: float
    current_price: float
    position_type: str   # 'option' or 'stock'
    
    # Option specifics (None for stock)
    strike: Optional[float] = None
    expiry: Optional[datetime] = None
    option_type: Optional[str] = None
    implied_vol: Optional[float] = None # Store implied vol from entry for delta calculation
    
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
        else:
            return max(0, self.strike - underlying_price)

@dataclass
class Trade:
    """Represents a trade to be executed."""
    symbol: str
    quantity: float
    price: float
    trade_type: str
    timestamp: datetime
    
    # Option specifics (None for stock)
    strike: Optional[float] = None
    expiry: Optional[datetime] = None
    option_type: Optional[str] = None
    implied_vol: Optional[float] = None     # Store IV for daily rebalancing
    delta: Optional[float] = None           # Store calculated delta for trade log validations
    mispricing_pct: Optional[float] = None  # Mispricing % at entry
    
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
        self.expiry_log: List[dict] = []      # Log for expiry events
        self.early_exit_log: List[dict] = []  # Log for early exit events
        self.iv_log: List[dict] = []          # Log for daily IV predictions from the strategy
        
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
                transaction_cost = abs(trade.quantity) * 0.5
            else:  # stock
                # 5 bips (0.05%)
                transaction_cost = abs(trade.quantity * trade.price) * 0.0005 # Accounting for slippage, bid/ask spread, etc. (SPY is highly liquid)
        
        # Update cash (buying = negative, selling = positive)
        self.cash -= (base_cost + transaction_cost)
        
        # Log trade
        self.trade_log.append(trade)
        
        # Get position key
        pos_key = trade.get_key()
        
        # Update or create position
        if pos_key in self.positions:
            pos = self.positions[pos_key]
            
            # Weighted average cost basis
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
                option_type=trade.option_type,
                implied_vol=trade.implied_vol
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

                # Log option-level expiry event
                self.expiry_log.append({
                    'date': current_date,
                    'event': 'expiry',
                    'position_key': pos_key,
                    'entry_price': pos.entry_price,
                    'exit_price': intrinsic,
                    'quantity': pos.quantity,
                    'pnl': pnl
                })

        # Remove expired positions
        for key in expired_keys:
            del self.positions[key]
        
        # Log full daily expiry event
        if total_pnl != 0:
            self.expiry_log.append({
                'date': current_date,
                'event': 'expiry',
                'position_key': 'Daily Expiry PnL',
                'pnl': total_pnl,
                'num_expired': len(expired_keys)
            })

        return total_pnl
    
    def log_iv_prediction(self, date: datetime, implied_vol: float, underlying_price: float):
        """
        Log the implied volatility prediction used by the strategy.
        
        Args:
            date: Current date
            implied_vol: IV prediction from the volatility model
            underlying_price: Current underlying price
        """
        self.iv_log.append({
            'date': date,
            'implied_vol': implied_vol,
            'underlying_price': underlying_price
        })
    
    def log_early_exit(self, date: datetime, pos_key: str, exit_price: float, 
                       quantity: float, entry_price: float = None):
        """
        Log early exit event for analysis.
        
        Args:
            date: Exit date
            pos_key: Position key that was closed
            exit_price: Price at which position was closed
            quantity: Quantity closed (negative of original position)
            entry_price: Original entry price (for P&L calculation)
        """
        # Calculate P&L
        pnl = None
        if entry_price is not None: # Error handling (Should always pass)
            pnl = (exit_price - entry_price) * quantity * 100
        
        self.early_exit_log.append({
            'date': date,
            'event': 'early_exit',
            'position_key': pos_key,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'quantity': quantity,
            'pnl': pnl
        })
    
    def get_portfolio_delta(self, S: float, r: float, current_date: datetime) -> float:
        """
        Calculate total portfolio delta.
        
        Args:
            S: Current underlying price
            r: Risk-free rate
            
        Returns:
            Total portfolio delta
        """
        total_delta = 0.0
        
        for _, pos in self.positions.items():
            if pos.position_type == 'option':
                # Calculate time to expiry in years (use total days, not just day component)
                T = max(0, (pos.expiry - current_date).days / 365.0)
                
                # Use stored IV
                if pos.implied_vol:
                    sigma = pos.implied_vol  
                    
                else:
                    try:
                        # Calculate implied volatility at current moment
                        sigma = implied_volatility(option_price = pos.current_price,
                                                    S = S,
                                                    K=pos.strike,
                                                    T=T,
                                                    r=r,
                                                    option_type=pos.option_type)
                    except Exception as e:
                        print(f"IV Calculation caused an error: {e}")
                        print("Defaulting to 0.13 IV")
                        sigma = 0.13

                # Get option delta
                greeks = calculate_all_greeks(
                    S=S,
                    K=pos.strike,
                    T=T,
                    r=r,
                    sigma=sigma,
                    option_type=pos.option_type
                )
                
                # Delta contribution (per contract = 100 shares)
                delta_contribution = pos.quantity * greeks['delta'] * 100
                total_delta += delta_contribution
            
            else:
                # Stock position
                total_delta += pos.quantity
        
        return total_delta

class BacktestEngine:
    """
    The main event loop for daily options trading with delta rebalancing.
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
        
    def load_data(self, options_dir: str, prices_path: str, rates_path: str):
        """
        Load market data and filter to backtest date range.
        
        Args:
            options_dir: Directory containing spy_options_YYYY_MM.parquet files
            prices_path: Path to prices parquet file
            rates_path: Path to risk-free rate parquet file
        """       
        # Load all monthly options files
        options_files = sorted(glob.glob(f'{options_dir}/spy_options_*.parquet'))
        if not options_files:
            raise FileNotFoundError(f"No spy_options_*.parquet files found in {options_dir}")
        
        print(f"  Loading {len(options_files)} monthly options files...")
        options_df = pd.concat([pd.read_parquet(f) for f in options_files], ignore_index=True)
        
        prices_df = pd.read_parquet(prices_path)
        rfr_df = pd.read_parquet(rates_path)
        
        # Convert all date columns to datetime
        options_df['date'] = pd.to_datetime(options_df['date'])
        options_df['expiration'] = pd.to_datetime(options_df['expiration'])
        prices_df['date'] = pd.to_datetime(prices_df['date'])
        rfr_df['date'] = pd.to_datetime(rfr_df['date'])

        # Format option_type string
        options_df['call_put'] = options_df['call_put'].str.lower()

        # Filter data to backtest date range
        options_df = options_df[(options_df['date'] <= self.end_date) & (options_df['date'] >= self.start_date)]
        prices_df = prices_df[(prices_df['date'] <= self.end_date)]
        rfr_df = rfr_df[(rfr_df['date'] <= self.end_date) & (rfr_df['date'] >= self.start_date)]
        
        # Print records size after backtest range filter
        print(f"  Loaded {len(options_df):,} total option records")

        # Calculate returns column for prices
        prices_df['log_ret'] = np.log(prices_df['close'] / prices_df['close'].shift(1)) * 100

        self.options_data = options_df
        self.prices_data = prices_df
        self.rates_data = rfr_df

    def set_strategy(self, strategy):
        """Attach a strategy to the engine."""
        self.strategy = strategy
    
    def get_current_option_prices(self, options_snapshot: pd.DataFrame) -> Dict[str, float]:
        """
        Extract current market prices for all held options from the options snapshot.
        
        Args:
            options_snapshot: DataFrame with today's option prices
            
        Returns:
            Dictionary mapping position_key -> current mid_price
        """
        prices = {}
        
        for pos_key, pos in self.portfolio.positions.items():
            if pos.position_type == 'option':
                # Find this option in today's snapshot
                option_match = options_snapshot[
                    (options_snapshot['strike'] == pos.strike) &
                    (options_snapshot['expiration'] == pos.expiry) &
                    (options_snapshot['call_put'] == pos.option_type)
                ]
                
                if len(option_match) > 0:
                    # Use actual market mid_price
                    prices[pos_key] = option_match.iloc[0]['mid_price']
                # If not found, position keeps its last known price
        
        return prices
    
    def rebalance_delta(self, current_date: datetime, underlying_price: float, r: float, ticker: str):
        """
        Rebalance delta hedge to maintain neutrality.
        
        Args:
            current_date: Current date
            underlying_price: Current SPY price
            r: Risk-free rate
        """
        # Get total portfolio delta (options + stock combined)
        total_delta = self.portfolio.get_portfolio_delta(S=underlying_price, r=r, current_date=current_date)
        
        # To neutralize, trade the negative of total delta
        # If total_delta > 0, sell shares (negative qty). If < 0, buy shares (positive qty).
        rebalance_qty = -total_delta

        if abs(rebalance_qty) > 0.01:  # Small threshold to avoid tiny trades
            trade = Trade(ticker, rebalance_qty, underlying_price, 'stock', current_date)
            self.portfolio.execute_trade(trade, True)

    def get_trading_days(self):
        """Get all days where option data is available for trading."""
        return sorted(self.options_data['date'].unique())
        
    def get_underlying_price(self, current_date: datetime):
        """Get underlying price for the given date, with fallback to most recent price."""
        # Look for exact match first
        price = self.prices_data.loc[self.prices_data['date'] == current_date, 'close']
        if len(price) > 0:
            return float(price.iloc[0])
        else:
            # Fall back to most recent price before this date
            available_prices = self.prices_data[self.prices_data['date'] < current_date]
            if len(available_prices) > 0:
                return float(available_prices.iloc[-1]['close'])
            else:
                raise ValueError(f"No price data found on or before {current_date}")

    def get_interest_rate(self, current_date: datetime):
        """Get risk-free rate for given date, using most recent rate if exact date not available."""
        rate_match = self.rates_data.loc[self.rates_data['date'] == current_date, 'risk_free_rate']
        
        if len(rate_match) > 0:
            return rate_match.iloc[0]
        else:
            # If exact date not found, use most recent rate before this date
            prior_rates = self.rates_data[self.rates_data['date'] < current_date]
            if len(prior_rates) > 0:
                return prior_rates.iloc[-1]['risk_free_rate']
            else:
                # Fallback: use first available rate
                return self.rates_data.iloc[0]['risk_free_rate']
    
    def _get_atm_iv_forecast(self, underlying_price: float, current_date: datetime, r: float) -> float:
        """
        Get IV forecast for ATM option from the strategy's vol model.
        """
        # ATM strike (rounded to nearest strike)
        K = round(underlying_price / 5) * 5
        T = 30 / 365  # 30-day forecast horizon
        
        # Use strategy's method to get theo price and IV
        _, sigma = self.strategy.calculate_theo_price(
            S=underlying_price,
            K=K,
            T=T,
            r=r,
            option_type='call',
            current_date=current_date
        )
        
        return sigma

    def run(self):
        """
        Run the backtest simulation with daily option trading and delta rebalancing.

        Structure:
            For each trading day:
                1. Handle expiries from previous period
                2. Rebalance delta after expiries
                3. Get options snapshot for this day
                4. Update vol model with historical data
                5. Generate signals from strategy
                6. Execute option trades
                7. Execute initial delta hedges
                8. Rebalance delta to account for new positions
                9. Mark to market
        
        Returns:
            DataFrame with backtest results
        """
        print("Starting backtest...")
        
        # Get all days with option data
        trading_days = self.get_trading_days()[30:] # Add 30 day buffer for rolling modeling approaches
       
        print(f"Trading days: {len(trading_days)}")
        print(f"First: {trading_days[0]}, Last: {trading_days[-1]}")
        
        # Iterate through each trading day
        for current_date in tqdm(trading_days, desc="Backtesting"):
            # Get underlying price and interest rate for this date
            underlying_price = self.get_underlying_price(current_date)
            current_rate = self.get_interest_rate(current_date)
            
            # Handle expiries
            self.portfolio.handle_expiries(current_date, underlying_price)

            # Get options snapshot for this trading day
            options_snapshot = self.options_data[self.options_data['date'] == current_date]
            
            # Update vol model with historical data (walk-forward)
            self.strategy.update_vol_model(current_date)
            
            # Check for early exits (mispricing converged to fair value)
            early_exit_trades = self.strategy.check_early_exits(
                self.portfolio, options_snapshot, underlying_price, current_date, current_rate
            )
            
            # Execute early exit trades
            for trade in early_exit_trades:
                # Get entry price before executing trade (position still exists)
                pos_key = trade.get_key()
                entry_price = None
                if pos_key in self.portfolio.positions:
                    entry_price = self.portfolio.positions[pos_key].entry_price
                
                self.portfolio.execute_trade(trade)
                
                # Log early exit event for analysis
                if trade.trade_type == 'option':
                    self.portfolio.log_early_exit(
                        date=current_date,
                        pos_key=pos_key,
                        exit_price=trade.price,
                        quantity=trade.quantity,
                        entry_price=entry_price
                    )
            
            # Generate signals for this trading day
            signals = self.strategy.generate_signals(
                self.portfolio, options_snapshot, underlying_price, current_date
            )
            
            # Log IV prediction (use first signal's IV if available, otherwise get ATM forecast)
            if len(signals) > 0:
                # Use the IV from the first signal as representative
                predicted_iv = signals[0].implied_vol
            else:
                # No signals generated, calculate and log IV forecast
                predicted_iv = self._get_atm_iv_forecast(underlying_price, current_date, current_rate)
            
            self.portfolio.log_iv_prediction(
                date=current_date,
                implied_vol=predicted_iv,
                underlying_price=underlying_price
            )
            
            # Execute option and hedge trades
            for signal in signals:
                option_trade = self.strategy.create_trade_from_signal(signal, 'option')
                self.portfolio.execute_trade(option_trade)
                
                hedge_trade = self.strategy.create_trade_from_signal(signal, 'stock')
                self.portfolio.execute_trade(hedge_trade)
            
            # Delta rebalance after all trades to ensure neutrality
            self.rebalance_delta(current_date, underlying_price, current_rate, 'SPY')
            
            # Mark to market with actual option prices from market data
            prices_dict = {'SPY': underlying_price}

            option_prices = self.get_current_option_prices(options_snapshot)
            prices_dict.update(option_prices)
            
            self.portfolio.mark_to_market(prices_dict, current_date)
        
        print("Backtest complete.")
        return pd.DataFrame(self.portfolio.history), pd.DataFrame(self.portfolio.trade_log)