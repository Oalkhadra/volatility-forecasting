"""
Strategy Interface and Implementations.

This module defines how trading decisions are made based on
theoretical pricing vs. market pricing.
"""

from typing import List, Tuple, Optional
from dataclasses import dataclass
import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os
from scipy.stats import norm
# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pricing.pricer import black_scholes_price, intrinsic_value
from pricing.greeks import delta
from volatility.forecasting import GARCHForecaster, MLForecaster
from volatility.historical import RollingVolCalculator
from volatility.vrp import VarianceRiskPremiumCalculator
from backtest.engine import Trade

# ============================================================================
# VECTORIZED PRICING FUNCTIONS
# ============================================================================

def black_scholes_price_vectorized(
    S: float,
    K: np.ndarray,
    T: np.ndarray,
    r: float,
    sigma: np.ndarray,
    is_call: np.ndarray  # Boolean array: True for call, False for put
) -> np.ndarray:
    """
    Vectorized Black-Scholes pricing for arrays of options.
    
    Args:
        S: Underlying price (scalar - same for all options)
        K: Strike prices (array)
        T: Time to expiry in years (array)
        r: Risk-free rate (scalar)
        sigma: Volatilities (array)
        is_call: Boolean array (True=call, False=put)
        
    Returns:
        Array of option prices
    """
    
    
    # Handle edge cases with masks
    at_expiry = T < 1e-6
    zero_vol = sigma < 1e-6
    normal_case = ~at_expiry & ~zero_vol
    
    # Pre-allocate result array
    prices = np.zeros_like(K, dtype=np.float64)
    
    # At expiry: intrinsic value
    call_intrinsic = np.maximum(S - K, 0)
    put_intrinsic = np.maximum(K - S, 0)
    prices = np.where(at_expiry & is_call, call_intrinsic, prices)
    prices = np.where(at_expiry & ~is_call, put_intrinsic, prices)
    
    # Zero vol: discounted intrinsic
    discount = np.exp(-r * T)
    prices = np.where(zero_vol & ~at_expiry & is_call, np.maximum(S - K * discount, 0), prices)
    prices = np.where(zero_vol & ~at_expiry & ~is_call, np.maximum(K * discount - S, 0), prices)
    
    # Normal case: Black-Scholes formula
    if np.any(normal_case):
        sqrt_T = np.sqrt(np.where(normal_case, T, 1.0))  # Avoid sqrt(0) warning
        sigma_sqrt_T = sigma * sqrt_T
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / sigma_sqrt_T
        d2 = d1 - sigma_sqrt_T
        
        # Call price: S*N(d1) - K*e^(-rT)*N(d2)
        call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        # Put price: K*e^(-rT)*N(-d2) - S*N(-d1)
        put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        
        prices = np.where(normal_case & is_call, call_price, prices)
        prices = np.where(normal_case & ~is_call, put_price, prices)
    
    return prices


def delta_vectorized(
    S: float,
    K: np.ndarray,
    T: np.ndarray,
    r: float,
    sigma: np.ndarray,
    is_call: np.ndarray  # Boolean array: True for call, False for put
) -> np.ndarray:
    """
    Vectorized delta calculation for arrays of options.
    
    Args:
        S: Underlying price (scalar)
        K: Strike prices (array)
        T: Time to expiry in years (array)
        r: Risk-free rate (scalar)
        sigma: Volatilities (array)
        is_call: Boolean array (True=call, False=put)
        
    Returns:
        Array of deltas
    """
    
    # Handle edge cases
    at_expiry = T < 1e-6
    normal_case = ~at_expiry
    
    # Pre-allocate result
    deltas = np.zeros_like(K, dtype=np.float64)
    
    # At expiry: delta is 1 for ITM calls, -1 for ITM puts, 0 otherwise
    call_itm = S > K
    put_itm = S < K
    deltas = np.where(at_expiry & is_call & call_itm, 1.0, deltas)
    deltas = np.where(at_expiry & ~is_call & put_itm, -1.0, deltas)
    
    # Normal case: N(d1) for calls, N(d1) - 1 for puts
    if np.any(normal_case):
        # Avoid division by zero
        sigma_safe = np.where(sigma > 1e-6, sigma, 1e-6)
        sqrt_T = np.sqrt(np.where(T > 0, T, 1e-6))
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma_safe**2) * T) / (sigma_safe * sqrt_T)
        
        call_delta = norm.cdf(d1)
        put_delta = call_delta - 1.0
        
        deltas = np.where(normal_case & is_call, call_delta, deltas)
        deltas = np.where(normal_case & ~is_call, put_delta, deltas)
    
    return deltas


# ============================================================================
# CONSTANTS
# ============================================================================

# Portfolio margin rate for short options (15% of notional value)
MARGIN_RATE = 0.15

def calculate_margin_requirement(strike: float, contracts: float) -> float:
    """Calculate margin requirement for short option position."""
    return strike * abs(contracts) * 100 * MARGIN_RATE

@dataclass
class TradeSignal:
    """Signal generated by a strategy."""
    symbol: str
    action: str              # 'buy' or 'sell'
    quantity: int            # Number of contracts
    market_price: float
    theo_price: float
    mispricing_pct: float    # (market - theo) / theo
    
    # Option metadata
    strike: float
    expiry: datetime
    option_type: str         # 'call' or 'put'
    
    # Delta hedge info
    delta: float
    hedge_quantity: float    # Shares of stock to trade
    
    # Additional metadata for analysis
    timestamp: datetime
    underlying_price: float
    dte: float               # Days to expiry
    implied_vol: float       # Store IV for daily rebalancing

class MispricingStrategy():
    """
    Strategy that trades based on mispricing between theoretical and market prices.
    
    Logic:
        1. Calculate theoretical price for each option using vol model
        2. Compare to market price
        3. If |market - theo| / theo > threshold:
           - market > theo → SELL (overpriced)
           - market < theo → BUY (underpriced)
        4. For each option trade, generate delta-hedge stock trade
        
    """
    
    def __init__(self, 
                 vol_model,
                 options_data: Optional[pd.DataFrame] = None,
                 prices_data: Optional[pd.DataFrame] = None,
                 buy_threshold: float = 0.50,
                 sell_threshold: float = 0.80,
                 exit_threshold_buy: float = 0.50,
                 exit_threshold_sell: float = 1.00,
                 max_positions: int = 5,
                 contracts_per_trade: int = 100,
                 max_leverage: float = 1.0,
                 max_position_pct: float = 0.50,
                 max_short_exposure: float = 0.30,
                 risk_premium: float = 0.0,
                 vrp_calculator: Optional[VarianceRiskPremiumCalculator] = None,
                 vrp_method: str = 'rolling',
                 use_all_dte: bool = True,
                 target_dte: int = 30):
        """
        Initialize mispricing strategy.
        
        Args:
            vol_model: Volatility forecasting model (GARCH, ML, Historical, or Surface)
            returns_data: Historical returns for GARCH/ML models
            options_data: Full options dataset for vol surface refitting
            prices_data: Price data for historical volatility calculator
            buy_threshold: Minimum underpricing to buy (e.g., 0.50 = buy if market < 50% of theo)
            sell_threshold: Minimum overpricing to sell (e.g., 0.80 = sell if market > 80% above theo)
            exit_threshold_buy: Exit long if mispricing rises above this (e.g., -0.50 becomes fair)
            exit_threshold_sell: Exit short if mispricing falls below this (e.g., +1.00 becomes fair)
            max_positions: Maximum number of option positions to hold
            contracts_per_trade: Max number of contracts per trade (reduced if capital limits hit)
            max_leverage: Maximum total capital usage / equity ratio (e.g., 3.0 = 3x)
            max_position_pct: Maximum single position capital as % of equity
            max_short_exposure: Maximum short position value as % of equity
            risk_premium: Variance risk premium multiplier (DEPRECATED - use vrp_calculator instead)
                         Fallback if vrp_calculator is None.
            vrp_calculator: VarianceRiskPremiumCalculator for dynamic VRP adjustment.
                           If provided, uses empirical VRP instead of fixed risk_premium.
            vrp_method: VRP estimation method ('rolling', 'regime', or 'regime_rolling')
                       Only used if vrp_calculator is provided.
            use_all_dte: If True, trade all DTEs. If False, only trade options with target_dte.
            target_dte: Specific DTE to trade when use_all_dte=False.
        """
        self.vol_model = vol_model
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.exit_threshold_buy = exit_threshold_buy
        self.exit_threshold_sell = exit_threshold_sell
        self.options_data = options_data
        self.prices_data = prices_data
        self.max_positions = max_positions
        self.contracts_per_trade = contracts_per_trade
        self.max_leverage = max_leverage
        self.max_position_pct = max_position_pct
        self.max_short_exposure = max_short_exposure
        self.risk_premium = risk_premium
        self.vrp_calculator = vrp_calculator
        self.vrp_method = vrp_method
        self.use_all_dte = use_all_dte
        self.target_dte = target_dte
        self.model_name = vol_model.__class__.__name__
    
    def calculate_capital_usage(self, portfolio) -> Tuple[float, float, float]:
        """
        Calculate current capital usage, short exposure, and equity.
        
        Capital usage is the actual market value of all positions:
        - Long options: quantity * current_price * 100
        - Short options: quantity * current_price * 100 (absolute value)
        - Long stock: quantity * current_price
        - Short stock: quantity * current_price (absolute value)
        
        Args:
            portfolio: Portfolio object with current positions
            
        Returns:
            Tuple of (capital_used, short_exposure, current_equity)
        """
        capital_used = 0.0
        short_exposure = 0.0
        
        for _, pos in portfolio.positions.items():
            if pos.position_type == 'option':
                # Option value: quantity * price * 100 (multiplier)
                position_value = abs(pos.quantity) * pos.current_price * 100
                capital_used += position_value
                
                # Track short exposure based on margin requirement (not premium)
                if pos.quantity < 0:
                    short_exposure += calculate_margin_requirement(pos.strike, pos.quantity)
            else:
                # Stock value: quantity * price
                position_value = abs(pos.quantity) * pos.current_price
                capital_used += position_value
                # Note: Short stock (delta hedges) excluded from short_exposure limit
        
        # Calculate current equity (cash + positions value)
        current_equity = portfolio.cash
        for _, pos in portfolio.positions.items():
            current_equity += pos.market_value()
        
        return capital_used, short_exposure, current_equity
        
    def _get_vol_forecast(self, dte_days: float, current_date: datetime) -> float:
        """
        Get volatility forecast from the vol model for a given DTE.
        
        Args:
            dte_days: Days to expiry
            current_date: Current date (for historical vol lookback)
            
        Returns:
            Volatility forecast (annualized, before risk premium)
        """
        horizon = max(1, int(dte_days))
        
        if isinstance(self.vol_model, (GARCHForecaster)):
            if self.vol_model.is_fitted:
                sigma = self.vol_model.forecast(horizon=horizon)
                return max(sigma, 0.08)  # Floor at 8%
            return 0.08  # Fallback if not fitted
        
        elif isinstance(self.vol_model, (MLForecaster)):
            if self.prices_data['date'].dtype == 'object':
                self.prices_data['date'] = pd.to_datetime(self.prices_data['date'])

            # Filter to only historical price data
            historical_prices = self.prices_data[self.prices_data['date'] < current_date]

            # Fit GARCH forecaster on available data
            ret_series = historical_prices['log_ret'].dropna()

            if self.vol_model.is_fitted:
                sigma = self.vol_model.forecast(horizon=horizon, returns=ret_series)
                return max(sigma, 0.08)  # Floor at 8%
            return 0.08  # Fallback if not fitted

        elif isinstance(self.vol_model, RollingVolCalculator):
            sigma = self.vol_model.get_current_vol(self.prices_data, current_date)
            return sigma if sigma is not None else 0.11
        
        # Scalar volatility
        return float(self.vol_model)
    
    def _apply_risk_premium(self, sigma: float, current_date: datetime = None) -> float:
        """
        Apply variance risk premium adjustment to volatility.
        
        If vrp_calculator is provided, uses empirical VRP (additive).
        Otherwise falls back to fixed risk_premium multiplier (multiplicative).
        
        Args:
            sigma: Base volatility forecast (realized vol)
            current_date: Current date for VRP lookup (required if using vrp_calculator)
            
        Returns:
            Adjusted volatility estimate (approximates implied vol)
        """
        if self.vrp_calculator is not None:
            # Use empirical VRP (additive adjustment)
            date_to_use = current_date
            if date_to_use is not None:
                try:
                    vrp = self.vrp_calculator.get_vrp_adjustment(
                        pd.Timestamp(date_to_use), 
                        method=self.vrp_method
                    )
                    # VRP is additive: IV ≈ RV + VRP
                    return max(sigma + vrp, 0.05)  # Floor at 5%
                except Exception as e:
                    # Fallback to fixed premium if VRP lookup fails
                    pass
        
        # Fallback: fixed multiplicative risk premium
        if self.risk_premium > 0:
            return sigma * (1 + self.risk_premium)
        return sigma

    def calculate_theo_price(self, 
                            S: float, 
                            K: float, 
                            T: float, 
                            r: float, 
                            option_type: str,
                            current_date: datetime) -> Tuple[float, float]:
        """
        Calculate theoretical option price using the vol model.

        Args:
            S: Current underlying price
            K: Strike price
            T: Time to expiration (years)
            r: Risk-free rate
            option_type: 'call' or 'put'
            current_date: Current date for VRP adjustment
            
        Returns:
            Tuple of (theoretical_price, sigma_used)
        """
        if T < 1e-6:
            return intrinsic_value(S=S, K=K, option_type=option_type), 0.0

        sigma = self._get_vol_forecast(T * 365, current_date)
        sigma = self._apply_risk_premium(sigma, current_date)
        theo = black_scholes_price(S=S, K=K, T=T, r=r, sigma=sigma, option_type=option_type)
        
        return theo, sigma
    
    def _vectorized_get_volatilities(self, df: pd.DataFrame, current_date: datetime) -> np.ndarray:
        """
        Vectorized volatility calculation for all options in dataframe.
        
        Args:
            df: DataFrame with 'days_to_expiry' column
            current_date: Current date
            
        Returns:
            Array of volatilities for each option (with VRP/risk premium applied)
        """    
        if isinstance(self.vol_model, (GARCHForecaster, MLForecaster)):
            if self.vol_model.is_fitted:
                # Get unique DTEs and forecast once per DTE (efficient)
                unique_dtes = df['days_to_expiry'].unique()
                dte_to_vol = {dte: self._get_vol_forecast(dte, current_date) for dte in unique_dtes}
                sigmas = df['days_to_expiry'].map(dte_to_vol).values
            else:
                sigmas = np.full(len(df), 0.08)
        else:
            # Historical or scalar - same vol for all options
            single_sigma = self._get_vol_forecast(30, current_date)  # DTE doesn't matter for these
            sigmas = np.full(len(df), single_sigma)
        
        # Apply VRP adjustment (vectorized)
        if self.vrp_calculator is not None:
            try:
                vrp = self.vrp_calculator.get_vrp_adjustment(
                    pd.Timestamp(current_date), 
                    method=self.vrp_method
                )
                # VRP is additive: IV ≈ RV + VRP
                sigmas = np.maximum(sigmas + vrp, 0.05)  # Floor at 5%
            except Exception:
                # Fallback to fixed premium if VRP lookup fails
                if self.risk_premium > 0:
                    sigmas = sigmas * (1 + self.risk_premium)
        elif self.risk_premium > 0:
            # Legacy: fixed multiplicative risk premium
            sigmas = sigmas * (1 + self.risk_premium)
        
        return sigmas

    def update_vol_model(self, current_date: datetime) -> None:
        """
        Update vol model with data available up to current_date.
        
        For VolatilitySurface: Refit surface weekly
        For GARCH: Could refit with expanding window.
        For ML: Only use historically trained model.
        For Historical vol: No need to update, calculated when needed
        
        Args:
            current_date: Current date in backtest
        """
        if isinstance(self.vol_model, GARCHForecaster):
            if self.prices_data['date'].dtype == 'object':
                self.prices_data['date'] = pd.to_datetime(self.prices_data['date'])

            # Filter to only historical price data
            historical_prices = self.prices_data[self.prices_data['date'] < current_date]

            # Fit GARCH forecaster on available data
            ret_series = historical_prices['log_ret'].dropna()
            self.vol_model.fit(ret_series)
        
        elif isinstance(self.vol_model, MLForecaster):
            pass
            


    def check_early_exits(self,
                         portfolio,
                         options_snapshot: pd.DataFrame,
                         underlying_price: float,
                         current_date: datetime,
                         risk_free_rate: float = 0.02) -> List[Trade]:
        """
        Check if any positions should be closed early due to mispricing convergence.
        
        Exit when the option reaches fair value (mispricing < exit_threshold).
        This allows the strategy to realize P&L early and redeploy capital.
        
        Args:
            portfolio: Current Portfolio object with positions
            options_snapshot: DataFrame with current day's option prices
            underlying_price: Current SPY price
            current_date: Current date
            risk_free_rate: Risk-free rate
            
        Returns:
            List of Trade objects to close positions (both option and hedge)
        """
        closing_trades = []
        
        # Early exit disabled if both thresholds are 0
        if self.exit_threshold_buy <= 0 and self.exit_threshold_sell <= 0:
            return []
        
        # Loop through all option positions
        for pos_key, pos in list(portfolio.positions.items()):
            if pos.position_type != 'option':
                continue
            
            # Find current market price for this option
            option_match = options_snapshot[
                (options_snapshot['strike'] == pos.strike) &
                (options_snapshot['expiration'] == pos.expiry) &
                (options_snapshot['call_put'] == pos.option_type)
            ]
            
            # Skip if option not found in today's market data
            if len(option_match) == 0:
                continue
            
            market_price = option_match.iloc[0]['mid_price']
            
            # Calculate time to expiry
            T = max(0, (pos.expiry - current_date).days / 365.0)
            
            # Skip very near expiry options (let them expire naturally)
            if T < 0.01:  # Less than ~4 days
                continue
            
            # Calculate current theoretical price
            theo_price, sigma = self.calculate_theo_price(
                S=underlying_price,
                K=pos.strike,
                T=T,
                r=risk_free_rate,
                option_type=pos.option_type,
                current_date=current_date
            )
            
            # Calculate current mispricing
            if theo_price > 0:
                current_mispricing = (market_price - theo_price) / theo_price
            else:
                print("Warning! Negative theoretical calculated, check pipeline!")
                continue 
            
            # Check if mispricing has converged to fair value using asymmetric thresholds
            # Long positions (bought underpriced): exit if mispricing rises above -exit_threshold_buy
            # Short positions (sold overpriced): exit if mispricing falls below exit_threshold_sell
            should_exit = False
            if pos.quantity > 0:  # Long position (we bought underpriced)
                # Originally mispricing was negative (market < theo)
                # Exit if it's no longer underpriced enough: mispricing > -exit_threshold_buy
                should_exit = current_mispricing > -self.exit_threshold_buy
            else:  # Short position (we sold overpriced)
                # Originally mispricing was positive (market > theo)
                # Exit if it's no longer overpriced enough: mispricing < exit_threshold_sell
                should_exit = current_mispricing < self.exit_threshold_sell
            
            if should_exit:
                # Calculate delta for hedge unwinding
                opt_delta = delta(underlying_price, pos.strike, T, risk_free_rate, sigma, pos.option_type)
                
                # Create closing trade for option (opposite of current position)
                closing_qty = -pos.quantity  # Negative to close
                
                option_close_trade = Trade(
                    symbol=pos.symbol,
                    quantity=closing_qty,
                    price=market_price,
                    trade_type='option',
                    timestamp=current_date,
                    strike=pos.strike,
                    expiry=pos.expiry,
                    option_type=pos.option_type,
                    implied_vol=sigma,
                    delta=opt_delta
                )
                closing_trades.append(option_close_trade)
                
                # Create closing trade for delta hedge
                hedge_unwind_qty = opt_delta * 100 * pos.quantity
                
                hedge_close_trade = Trade(
                    symbol=pos.symbol,
                    quantity=hedge_unwind_qty,
                    price=underlying_price,
                    trade_type='stock',
                    timestamp=current_date,
                    strike=None,
                    expiry=None,
                    option_type=None
                )
                closing_trades.append(hedge_close_trade)
        
        return closing_trades

    def generate_signals(self, 
                        portfolio, 
                        options_snapshot: pd.DataFrame,
                        underlying_price: float,
                        current_date: datetime,
                        risk_free_rate: float = 0.02) -> List[TradeSignal]:
        """
        Generate trading signals based on mispricing.
             
        Args:
            portfolio: Current Portfolio object
            options_snapshot: DataFrame with columns:
                - strike, expiry_date, call_put, mid_price, underlying_price, days_to_expiry
            underlying_price: Current SPY price
            current_date: Current date
            risk_free_rate: Risk-free rate
            
        Returns:
            List of TradeSignal objects (sorted by 'best' opportunities)

        """
        signals = []
        
        # Step 1: Check current positions
        num_current_positions = len(portfolio.positions)

        if num_current_positions >= self.max_positions:
            return []

        else:
            new_position_count = self.max_positions - num_current_positions

        
        # Step 2: Vectorized calculation of mispricings
        if len(options_snapshot) == 0:
            return []
        
        # Create a copy to avoid modifying original
        df = options_snapshot.copy()
        
        # Filter by DTE if not using all DTEs
        if not self.use_all_dte:
            df = df[df['days_to_expiry'] == self.target_dte]
            if len(df) == 0:
                return []
        
        # Calculate time to expiry in years (vectorized)
        df['T'] = df['days_to_expiry'] / 365
        
        # Vectorized volatility calculation
        df['sigma'] = self._vectorized_get_volatilities(df, current_date)
        
        # Handle near-expiration cases (vectorized)
        near_expiry_mask = df['T'] < 1e-6
        
        # Truly vectorized theoretical price calculation using numpy arrays
        is_call = (df['call_put'] == 'call').values
        
        df['theo_price'] = black_scholes_price_vectorized(
            S=underlying_price,
            K=df['strike'].values,
            T=df['T'].values,
            r=risk_free_rate,
            sigma=df['sigma'].values,
            is_call=is_call
        )
        
        # Calculate mispricing percentage (vectorized)
        df['mispricing_pct'] = (df['mid_price'] - df['theo_price']) / df['theo_price']
        
        # Filter for mispricings above asymmetric thresholds (vectorized)
        # Buy if underpriced: mispricing_pct < -buy_threshold (negative = market < theo)
        # Sell if overpriced: mispricing_pct > sell_threshold (positive = market > theo)
        buy_candidates = df['mispricing_pct'] < -self.buy_threshold
        sell_candidates = df['mispricing_pct'] > self.sell_threshold
        df_mispriced = df[buy_candidates | sell_candidates].copy()
        

        # In strategy.py, add filter before calculating deltas
        df_mispriced = df_mispriced[
            (df_mispriced['strike'] / underlying_price).between(0.95, 1.05)
        ]


        if len(df_mispriced) == 0:
            return []
        
        # Determine action (vectorized)
        df_mispriced['action'] = np.where(df_mispriced['mispricing_pct'] > 0, 'sell', 'buy')
        
        # Truly vectorized delta calculation
        is_call_mispriced = (df_mispriced['call_put'] == 'call').values
        
        df_mispriced['delta'] = delta_vectorized(
            S=underlying_price,
            K=df_mispriced['strike'].values,
            T=df_mispriced['T'].values,
            r=risk_free_rate,
            sigma=df_mispriced['sigma'].values,
            is_call=is_call_mispriced
        )
        
        # Sort by mispricing magnitude before position sizing
        # This ensures best opportunities get capital first
        df_mispriced = df_mispriced.sort_values('mispricing_pct', key=abs, ascending=False)
        
        # Track running capital usage as we generate signals
        # This prevents over-allocation when multiple signals are generated in one batch
        capital_used, short_exposure, current_equity = self.calculate_capital_usage(portfolio)
        running_capital_used = capital_used
        running_short_exposure = short_exposure
        
        # Create TradeSignal objects with dynamic position sizing
        for _, row in df_mispriced.iterrows():
            opt_delta = row['delta']
            option_price = row['mid_price']
            action = row['action']
            strike = row['strike']
            
            # Calculate capital required for this trade
            option_capital = option_price * 100
            
            # Hedge capital (full value)
            hedge_shares = abs(opt_delta) * 100
            hedge_capital = hedge_shares * underlying_price
            
            capital_per_contract = option_capital + hedge_capital
            
            # Check if we have capacity for at least 1 contract
            max_capital = current_equity * self.max_leverage
            available_capital = max(0, max_capital - running_capital_used)
            max_position_capital = current_equity * self.max_position_pct
            
            # Check short exposure capacity
            max_short_capital = current_equity * self.max_short_exposure
            available_short_capacity = max(0, max_short_capital - running_short_exposure)
            
            if capital_per_contract > 0:
                max_contracts_total = int(available_capital / capital_per_contract)
                max_contracts_position = int(max_position_capital / option_capital) if option_capital > 0 else self.contracts_per_trade
            else:
                max_contracts_total = self.contracts_per_trade
                max_contracts_position = self.contracts_per_trade
            
            # For short positions, check short exposure limit based on margin requirement
            if action == 'sell':
                margin_per_contract = calculate_margin_requirement(strike, 1)
                max_contracts_short = int(available_short_capacity / margin_per_contract) if margin_per_contract > 0 else self.contracts_per_trade
            else:
                max_contracts_short = self.contracts_per_trade
            
            contracts = min(self.contracts_per_trade, max_contracts_total, max_contracts_position, max_contracts_short)
            
            # Skip if we can't take any contracts due to capital limits
            if contracts <= 0:
                continue
            
            # Update running capital usage for next iteration
            running_capital_used += contracts * capital_per_contract
            
            # Update running short exposure for short positions (based on margin requirement)
            if action == 'sell':
                running_short_exposure += calculate_margin_requirement(strike, contracts)
            
            # Apply direction (buy = positive, sell = negative)
            contract_qty = contracts if action == 'buy' else -contracts
            hedge_qty = -opt_delta * 100 * contract_qty
            
            opt_signal = TradeSignal(
                symbol=row['act_symbol'],
                action=action,
                quantity=contract_qty,
                market_price=option_price,
                theo_price=row['theo_price'],
                mispricing_pct=row['mispricing_pct'],
                strike=strike,
                expiry=row['expiration'],
                option_type=row['call_put'],
                delta=opt_delta,
                hedge_quantity=hedge_qty,
                timestamp=current_date,
                underlying_price=underlying_price,
                dte=row['T'],
                implied_vol=row['sigma']
            )
            signals.append(opt_signal)
            
            # Stop if we've reached max new positions
            if len(signals) >= new_position_count:
                break

        return signals
    
    def create_trade_from_signal(self, signal: TradeSignal, trade_type: str) -> Trade:
        """
        Convert a TradeSignal to a Trade object for execution.
        
        This handles the conversion from strategy signal to actual trade.
        
        Args:
            signal: TradeSignal object with trade details
            trade_type: 'option' or 'stock'
                - 'option': Create option trade from signal
                - 'stock': Create delta hedge trade
                
        Returns:
            Trade object ready for portfolio.execute_trade()
        """

        if trade_type == 'option':
            # Create option trade
            trade = Trade(
                symbol=signal.symbol,
                quantity=signal.quantity,
                price=signal.market_price,
                trade_type='option',
                timestamp=signal.timestamp,
                strike=signal.strike,
                expiry=signal.expiry,
                option_type=signal.option_type,
                implied_vol=signal.implied_vol,
                delta=signal.delta,
                mispricing_pct=signal.mispricing_pct
            )
            
        elif trade_type == 'stock':
            # Create delta hedge trade
            trade = Trade(
                symbol=signal.symbol,
                quantity=signal.hedge_quantity,
                price=signal.underlying_price,
                trade_type='stock',
                timestamp=signal.timestamp,
                strike=None,
                expiry=None,
                option_type=None
            )
        
        return trade


# ============================================================================
# HELPER FUNCTIONS FOR STRATEGY ANALYSIS
# ============================================================================

def create_option_key(symbol: str, strike: float, expiry: datetime, option_type: str) -> str:
    """Create unique identifier for an option."""
    return f"{symbol}_{strike:.0f}{option_type[0].upper()}_{expiry.strftime('%Y%m%d')}"