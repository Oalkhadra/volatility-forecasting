import numpy as np
from scipy.stats import norm
from scipy.optimize import newton, brentq
from typing import Literal
from .pricer import black_scholes_price, moneyness, intrinsic_value


def delta(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: Literal['call', 'put']
) -> float:
    """
    Calculate option delta (∂V/∂S).
    
    Delta measures how much the option price changes when the stock price 
    moves by $1. It's also interpreted as the hedge ratio - how many shares 
    to hold to delta-hedge one option.
    
    Args:
        S: Current stock price
        K: Strike price
        T: Time to expiration (years)
        r: Risk-free rate (annual)
        sigma: Volatility (annual)
        option_type: 'call' or 'put'
    
    Returns:
        Delta value (between 0 and 1 for calls, -1 and 0 for puts)

    """

    # Calculate d1
    denom = sigma* np.sqrt(T)
    d_1_num = np.log(S / K) + (r + (sigma ** 2) / 2) * T
    d1 = d_1_num / denom if denom != 0 else 0
    
    if option_type == 'call':
        if T == 0:
            return 1 if moneyness(S, K, option_type) == 'ITM' else 0
        
        return norm.cdf(d1)


    if option_type == 'put':
        if T == 0:
            return -1 if moneyness(S, K, option_type) == 'ITM' else 0
        return norm.cdf(d1) - 1

def gamma(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: Literal['call', 'put']
) -> float:
    """
    Calculate option gamma (∂²V/∂S² = ∂Δ/∂S).
    
    Gamma measures how fast delta changes as the stock price moves.
    Gamma is highest for ATM options near expiration.

    Args:
        S: Current stock price
        K: Strike price
        T: Time to expiration (years)
        r: Risk-free rate (annual)
        sigma: Volatility (annual)
        option_type: 'call' or 'put' (doesn't matter, gamma is same)
    
    Returns:
        Gamma value (always positive)

    """
    # Handle expiration cases
    if T == 0:
        return 0 if moneyness(S, K, option_type) != 'ATM' else 1e6

    # Calculate d1
    denom = sigma* np.sqrt(T)
    d_1_num = np.log(S / K) + (r + (sigma ** 2) / 2) * T
    d1 = d_1_num / denom if denom != 0 else 0
    
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))

def vega(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float
) -> float:
    """
    Calculate option vega (∂V/∂σ).
    
    Vega measures how much the option price changes when implied volatility 
    changes by 1 percentage point (e.g., from 20% to 21%).
    
    Fun Fact: Vega is not even a real Greek letter
    
    Args:
        S: Current stock price
        K: Strike price
        T: Time to expiration (years)
        r: Risk-free rate (annual)
        sigma: Volatility (annual)
    
    Returns:
        Vega value (price change per 1% vol change)

    """
    if T == 0:
        return 0

    # Calculate d1
    denom = sigma* np.sqrt(T)
    d_1_num = np.log(S / K) + (r + (sigma ** 2) / 2) * T
    d1 = d_1_num / denom if denom != 0 else 0

    return (S * np.sqrt(T) * norm.pdf(d1)) / 100


def theta(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: Literal['call', 'put']
) -> float:
    """
    Calculate option theta (∂V/∂t).
    
    Theta measures time decay - how much value the option loses as one 
    day passes (assuming everything else stays constant).

    Args:
        S: Current stock price
        K: Strike price
        T: Time to expiration (years)
        r: Risk-free rate (annual)
        sigma: Volatility (annual)
        option_type: 'call' or 'put'
    
    Returns:
        Theta value (price change per day, typically negative)

    """
    if T == 0:
        return 0
    
    # Calculate d1 and d2
    denom = sigma* np.sqrt(T)
    d_1_num = np.log(S / K) + (r + (sigma ** 2) / 2) * T
    d1 = d_1_num / denom if denom != 0 else 0
    d2 = d1 - sigma * np.sqrt(T)

    # Calculate option value and cost
    op_val = S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
    op_cost = r * K * np.exp(-r * T)

    if option_type == 'call':
        return (-op_val - op_cost * norm.cdf(d2)) / 365
    if option_type == 'put':
        return (-op_val + op_cost * norm.cdf(-d2)) / 365


def rho(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: Literal['call', 'put']
) -> float:
    """
    Calculate option rho (∂V/∂r).
    
    Rho measures how much the option price changes when interest rates 
    change by 1 percentage point.
    
    Args:
        S: Current stock price
        K: Strike price
        T: Time to expiration (years)
        r: Risk-free rate (annual)
        sigma: Volatility (annual)
        option_type: 'call' or 'put'
    
    Returns:
        Rho value (price change per 1% rate change)

    """

    if T == 0:
        return 0

    # Calculate d1 and d2
    denom = sigma* np.sqrt(T)
    d_1_num = np.log(S / K) + (r + (sigma ** 2) / 2) * T
    d1 = d_1_num / denom if denom != 0 else 0
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == 'call':
        return (K * T * np.exp(-r * T) * norm.cdf(d2)) / 100
    
    if option_type == 'put':
        return (-K * T * np.exp(-r * T) * norm.cdf(-d2)) / 100


def implied_volatility(
    option_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: Literal['call', 'put'],
    initial_guess: float = 0.25,
    max_iterations: int = 100,
    tolerance: float = 1e-6
) -> float:
    """
    Calculate implied volatility using Newton-Raphson method with fallback.
    
    Args:
        option_price: Observed market price
        S: Current stock price
        K: Strike price
        T: Time to expiration (years)
        r: Risk-free rate (annual)
        option_type: 'call' or 'put'
        initial_guess: Starting volatility guess (default 25%)
        max_iterations: Maximum iterations
        tolerance: Convergence tolerance
    
    Returns:
        Implied volatility (annual)
     
    """
    
    # Validation: Check for No Arbitrage
    intrinsic = intrinsic_value(S, K, option_type)
    if option_price < intrinsic - 0.01:
        raise ValueError(
            f"Option price ${option_price:.2f} below intrinsic value ${intrinsic:.2f}. "
            f"This violates no-arbitrage"
            f"Option characteristics: S=${S:.2f}, K=${K:.2f}, T={T:.4f}, "
            f"Price=${option_price:.2f}, Intrinsic=${intrinsic:.2f}, Type={option_type}. "
            f"This may indicate an arbitrage violation or extreme market conditions."
        )
    
    # Edge case: at or past expiration
    if T <= 0:
        raise ValueError("Cannot calculate implied volatility at or past expiration")
    
    # Edge case: price equals intrinsic (zero time value)
    if abs(option_price - intrinsic) < 0.01:
        return 0.01  # Minimum vol (essentially zero time value)
    
    # For deep OTM options near expiry, use a higher initial guess
    if T < 0.05:  # Less than ~18 days to expiry
        if option_type == 'call' and S / K < 0.90:  # Deep OTM call
            initial_guess = min(1.0, initial_guess * 3)
        elif option_type == 'put' and S / K > 1.10:  # Deep OTM put
            initial_guess = min(1.0, initial_guess * 3)
    
    # Define the objective function: f(σ) = BS_price(σ) - market_price
    def objective(sigma):
        sigma = np.clip(sigma, 0.0001, 5.0)
        return black_scholes_price(S, K, T, r, sigma, option_type) - option_price

    # Define the derivative: f'(σ) = vega(σ) 
    def derivative(sigma):
        """Derivative of objective function (vega)."""
        sigma = np.clip(sigma, 0.0001, 5.0)

        # Get vega (per 1% vol change)
        option_vega = vega(S, K, T, r, sigma, option_type)
        
        # Scale vega from per-1% to per-100% for Newton-Raphson
        scaled_vega = option_vega * 100
        
        # Prevent division by zero
        if abs(scaled_vega) < 1e-10:
            return 1e-10
        
        return scaled_vega

    
    try:
        # Try Newton-Raphson first (fast when it works)
        solved_iv = newton(
            func=objective,
            x0=initial_guess,
            fprime=derivative,
            tol=tolerance,
            maxiter=max_iterations,
            full_output=False
        )
        
        # Bound result to reasonable range
        solved_iv = np.clip(solved_iv, 0.001, 5.0)
        
        # Final verification
        verification_price = black_scholes_price(S, K, T, r, solved_iv, option_type)
        if abs(verification_price - option_price) > 0.10:
            raise ValueError(
                f"IV solution verification failed. "
                f"Solved IV={solved_iv:.4f} gives price ${verification_price:.2f}, "
                f"but market price is ${option_price:.2f}"
            )
        
        return solved_iv
        
    except RuntimeError:
        # Newton-Raphson failed - fallback to bounded bisection method
        try:
            solved_iv = brentq(
                f=objective,
                a=0.001,  
                b=5.0,    
                xtol=tolerance,
                maxiter=max_iterations
            )
            
            # Verify solution
            verification_price = black_scholes_price(S, K, T, r, solved_iv, option_type)
            if abs(verification_price - option_price) > 0.10:
                raise ValueError(
                    f"IV solution verification failed (Brent's method). "
                    f"Solved IV={solved_iv:.4f} gives price ${verification_price:.2f}, "
                    f"but market price is ${option_price:.2f}"
                )
            
            return solved_iv
            
        except ValueError as e:
            # Even bounded method failed - likely arbitrage violation or extreme case
            raise ValueError(
                f"IV solver failed (both Newton-Raphson and Brent's method). "
                f"Option characteristics: S=${S:.2f}, K=${K:.2f}, T={T:.4f}, "
                f"Price=${option_price:.2f}, Intrinsic=${intrinsic:.2f}, Type={option_type}. "
                f"This may indicate an arbitrage violation or extreme market conditions."
            ) from e


def calculate_all_greeks(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: Literal['call', 'put']
) -> dict:
    """
    Calculate all Greeks at once for a given option.
    
    Args:
        S: Current stock price
        K: Strike price
        T: Time to expiration (years)
        r: Risk-free rate (annual)
        sigma: Volatility (annual)
        option_type: 'call' or 'put'
    
    Returns:
        Dictionary with keys: 'delta', 'gamma', 'vega', 'theta', 'rho'
    """
    return {
        'delta': delta(S, K, T, r, sigma, option_type),
        'gamma': gamma(S, K, T, r, sigma, option_type),
        'vega': vega(S, K, T, r, sigma),
        'theta': theta(S, K, T, r, sigma, option_type),
        'rho': rho(S, K, T, r, sigma, option_type)
    }
