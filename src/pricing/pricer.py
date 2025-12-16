import numpy as np
from scipy.stats import norm
from typing import Literal



def black_scholes_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: Literal['call', 'put'] = 'call'
) -> float:
    """
    Calculate Theoretical option price using Black-Scholes model.
    
    Args:
        S: Current underlying price
        K: Strike price
        T: Time to expiration in years (e.g., 30 days = 30/365)
        r: Risk-free rate (annualized, as decimal, e.g., 0.02 for 2%)
        sigma: Implied volatility (annualized, as decimal, e.g., 0.20 for 20%)
        option_type: 'call' or 'put'
        
    Returns:
        Theoretical option price

    """
    # Input validation
    if S <= 0:
        raise ValueError(f"Underlying price must be positive, got {S}")
    if K <= 0:
        raise ValueError(f"Strike price must be positive, got {K}")
    if T < 0:
        raise ValueError(f"Time to expiration cannot be negative, got {T}")
    if sigma < 0:
        raise ValueError(f"Volatility cannot be negative, got {sigma}")
    if option_type not in ['call', 'put']:
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type}")

    # Calculate d1 and d2
    denom = sigma* np.sqrt(T)
    d_1_num = np.log(S / K) + (r + (sigma ** 2) / 2) * T
    d1 = d_1_num / denom if denom != 0 else 0
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == 'call':
        # If contract expired, return 0 or difference between underlying and strike
        if T == 0:
            return intrinsic_value(S, K, option_type)
        
        # If no vol, return discounted intrinsic value
        if sigma == 0:
            return max(S - K * np.exp(-r*T), 0)

        return (S * norm.cdf(d1)) - (K * np.exp(-r*T) * norm.cdf(d2))

    if option_type == 'put':
        # If contract expired, return 0 or difference between underlying and strike
        if T == 0:
            return intrinsic_value(S, K, option_type)

        # If no vol, return discounted intrinsic value
        if sigma == 0:
            return max((K * np.exp(-r*T)) - S, 0)

        return (K * np.exp(-r*T) * norm.cdf(-d2)) - (S * norm.cdf(-d1)) 
  

def intrinsic_value(
    S: float,
    K: float,
    option_type: Literal['call', 'put'] = 'call'
) -> float:
    """
    Calculate intrinsic value of an option.
    
    Args:
        S: Current underlying price
        K: Strike price
        option_type: 'call' or 'put'
        
    Returns:
        Intrinsic value of option

    """
    if option_type == 'call':
        return max(S - K, 0)

    if option_type == 'put':
        return max(K - S, 0)


def time_value(
    option_price: float,
    S: float,
    K: float,
    option_type: Literal['call', 'put'] = 'call'
) -> float:
    """
    Calculate time value of an option.
    
    Time value = Option price - Intrinsic value
    
    Args:
        option_price: Current option price
        S: Current underlying price
        K: Strike price
        option_type: 'call' or 'put'
        
    Returns:
        Time value of option

    """
    return option_price - intrinsic_value(S, K, option_type)


def moneyness(
    S: float,
    K: float,
    option_type: Literal['call', 'put'] = 'call'
) -> str:
    """
    Determine if option is ITM, ATM, or OTM.
    
    Args:
        S: Current underlying price
        K: Strike price
        option_type: 'call' or 'put'
        
    Returns:
        'ITM', 'ATM', or 'OTM'

    """
    # Check ATM first
    if np.abs(S - K) / S < 0.005:
        return 'ATM'
    
    # Check based on option type
    if option_type == 'call':
        return 'ITM' if S > K else 'OTM'
    else:  # put
        return 'ITM' if S < K else 'OTM'
    