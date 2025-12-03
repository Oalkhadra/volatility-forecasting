"""
Greeks Calculator for Options

The "Greeks" measure how sensitive an option's price is to various factors:
- Delta: Price sensitivity to stock price changes
- Gamma: Rate of delta change
- Vega: Price sensitivity to volatility changes
- Theta: Time decay (how much value lost per day)
- Rho: Price sensitivity to interest rate changes

Also includes implied volatility solver.
"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import newton
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
    
    Formula:
        Call Delta = N(d1)
        Put Delta = N(d1) - 1 = -N(-d1)
    
    Where:
        d1 = [ln(S/K) + (r + σ²/2)T] / (σ√T)
        N(x) = cumulative normal distribution
    
    Args:
        S: Current stock price
        K: Strike price
        T: Time to expiration (years)
        r: Risk-free rate (annual)
        sigma: Volatility (annual)
        option_type: 'call' or 'put'
    
    Returns:
        Delta value (between 0 and 1 for calls, -1 and 0 for puts)
    
    Examples:
        >>> delta(100, 100, 1.0, 0.05, 0.20, 'call')
        0.6368...  # ATM call delta ~0.5-0.6
        
        >>> delta(100, 100, 1.0, 0.05, 0.20, 'put')
        -0.3632... # ATM put delta ~-0.4 to -0.5
    
    Edge Cases:
        - At expiration (T=0): Delta is 1 for ITM, 0 for OTM
        - Deep ITM: Call delta → 1, Put delta → -1
        - Deep OTM: Call delta → 0, Put delta → 0
    
    Interview Question: "Why is ATM delta ~0.5?"
        ATM options have ~50% probability of finishing ITM, and delta 
        approximates the probability of the option expiring in the money.
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
    High gamma means delta is very sensitive to price moves - dangerous 
    for hedging! Gamma is highest for ATM options near expiration.
    
    Formula:
        Gamma (same for calls and puts) = N'(d1) / (S·σ·√T)
    
    Where:
        N'(x) = standard normal PDF = (1/√2π)·e^(-x²/2)
        d1 = [ln(S/K) + (r + σ²/2)T] / (σ√T)
    
    Args:
        S: Current stock price
        K: Strike price
        T: Time to expiration (years)
        r: Risk-free rate (annual)
        sigma: Volatility (annual)
        option_type: 'call' or 'put' (doesn't matter, gamma is same)
    
    Returns:
        Gamma value (always positive)
    
    Examples:
        >>> gamma(100, 100, 1.0, 0.05, 0.20, 'call')
        0.0184...  # ATM gamma
        
        >>> gamma(100, 120, 1.0, 0.05, 0.20, 'call')
        0.0089...  # OTM gamma (lower)
    
    Edge Cases:
        - At expiration (T=0): Gamma → infinity at ATM (discontinuity)
        - Deep ITM/OTM: Gamma → 0
    
    Interview Question: "Why does gamma increase near expiration?"
        As T→0, the option becomes more binary (in or out), so delta 
        changes very rapidly near the strike. This makes ATM options 
        near expiry very difficult to hedge.
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
    sigma: float,
    option_type: Literal['call', 'put']
) -> float:
    """
    Calculate option vega (∂V/∂σ).
    
    Vega measures how much the option price changes when implied volatility 
    changes by 1 percentage point (e.g., from 20% to 21%).
    
    Note: Vega is NOT actually a Greek letter (it's made up)!
    
    Formula:
        Vega (same for calls and puts) = S·√T·N'(d1)
    
    Where:
        N'(x) = standard normal PDF = (1/√2π)·e^(-x²/2)
        d1 = [ln(S/K) + (r + σ²/2)T] / (σ√T)
    
    Args:
        S: Current stock price
        K: Strike price
        T: Time to expiration (years)
        r: Risk-free rate (annual)
        sigma: Volatility (annual)
        option_type: 'call' or 'put' (doesn't matter, vega is same)
    
    Returns:
        Vega value (price change per 1% vol change)
    
    Examples:
        >>> vega(100, 100, 1.0, 0.05, 0.20, 'call')
        39.89...  # If vol goes 20% → 21%, price increases by ~$39.89
        
        >>> vega(100, 100, 0.1, 0.05, 0.20, 'call')
        12.61...  # Less time = less vega
    
    Edge Cases:
        - At expiration (T=0): Vega = 0 (no time for vol to matter)
        - ATM options: Highest vega
        - Deep ITM/OTM: Lower vega
    
    Interview Question: "Why do traders care about vega?"
        Volatility is mean-reverting. If you buy when vol is low and 
        sell when vol is high, you can profit from vega even if the 
        stock doesn't move. This is called "vol trading."
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
    
    Theta is usually NEGATIVE for long options (you lose money over time).
    
    Formula:
        Call Theta = -[S·N'(d1)·σ / (2√T)] - r·K·e^(-rT)·N(d2)
        Put Theta = -[S·N'(d1)·σ / (2√T)] + r·K·e^(-rT)·N(-d2)
    
    Where:
        d1 = [ln(S/K) + (r + σ²/2)T] / (σ√T)
        d2 = d1 - σ√T
        N'(x) = standard normal PDF
    
    Args:
        S: Current stock price
        K: Strike price
        T: Time to expiration (years)
        r: Risk-free rate (annual)
        sigma: Volatility (annual)
        option_type: 'call' or 'put'
    
    Returns:
        Theta value (price change per day, typically negative)
    
    Examples:
        >>> theta(100, 100, 1.0, 0.05, 0.20, 'call')
        -0.0198...  # Loses ~$0.02 per day
        
        >>> theta(100, 100, 0.1, 0.05, 0.20, 'call')
        -0.0888...  # More negative near expiration
    
    Edge Cases:
        - Near expiration: Theta becomes very negative for ATM
        - Deep ITM puts: Theta can be POSITIVE (due to interest on strike)
        - At expiration: Theta is undefined
    
    Interview Question: "Why is theta negative?"
        Options are "wasting assets." As time passes, there's less time 
        for the stock to move, so the option's time value decays. This 
        accelerates near expiration (gamma risk).
    """
    # Theta is usually expressed per day, so divide annual theta by 365
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
    change by 1 percentage point (e.g., from 5% to 6%).
    
    Rho is the LEAST important Greek in practice - rate changes are slow 
    and usually dominated by other factors.
    
    Formula:
        Call Rho = K·T·e^(-rT)·N(d2)
        Put Rho = -K·T·e^(-rT)·N(-d2)
    
    Where:
        d2 = d1 - σ√T
        d1 = [ln(S/K) + (r + σ²/2)T] / (σ√T)
    
    Args:
        S: Current stock price
        K: Strike price
        T: Time to expiration (years)
        r: Risk-free rate (annual)
        sigma: Volatility (annual)
        option_type: 'call' or 'put'
    
    Returns:
        Rho value (price change per 1% rate change)
    
    Examples:
        >>> rho(100, 100, 1.0, 0.05, 0.20, 'call')
        51.75...  # If rates go 5% → 6%, call gains ~$51.75
        
        >>> rho(100, 100, 1.0, 0.05, 0.20, 'put')
        -46.03... # If rates go 5% → 6%, put loses ~$46.03
    
    Edge Cases:
        - Longer maturity: Higher rho
        - At expiration: Rho = 0
        - Calls: Positive rho (benefit from higher rates)
        - Puts: Negative rho (hurt by higher rates)
    
    Interview Question: "Why do calls benefit from higher rates?"
        Higher rates reduce the present value of paying the strike price, 
        making calls more valuable. Think of it as cheaper financing.
    """
    # TODO: Implement rho calculation
    # Hint: Use np.exp(-r*T) for discounting
    # Rho is usually expressed per 1% rate change (divide by 100)
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
    max_iterations: int = 10,
    tolerance: float = 1e-6
) -> float:
    """
    Calculate implied volatility using Newton-Raphson method.
    
    Implied volatility (IV) is the "market's opinion" of future volatility.
    It's the σ value that makes Black-Scholes price match the observed 
    market price.
    
    This is an inverse problem: we know the price, solve for σ.
    
    Newton-Raphson Formula:
        σ_new = σ_old - f(σ) / f'(σ)
    
    Where:
        f(σ) = BS_price(σ) - market_price
        f'(σ) = vega(σ)
    
    Args:
        option_price: Observed market price
        S: Current stock price
        K: Strike price
        T: Time to expiration (years)
        r: Risk-free rate (annual)
        option_type: 'call' or 'put'
        initial_guess: Starting volatility guess (default 25%)
        max_iterations: Maximum Newton-Raphson iterations
        tolerance: Convergence tolerance
    
    Returns:
        Implied volatility (annual)
     
    Raises:
        ValueError: If IV cannot be found (price violates arbitrage bounds)
    
    Examples:
        >>> # If market price is $10, what vol is implied?
        >>> implied_volatility(10.0, 100, 100, 1.0, 0.05, 'call')
        0.1987...  # ~19.87% implied vol
    
    Edge Cases:
        - Price below intrinsic value: Raise ValueError (arbitrage)
        - Very deep ITM/OTM: Vega → 0, Newton-Raphson fails
        - At expiration: IV is meaningless
    
    Interview Question: "Why solve for IV instead of using historical vol?"
        Historical vol tells you what WAS. Implied vol tells you what 
        the MARKET THINKS will happen. IV is forward-looking and 
        incorporates all market participants' views.
    
    Advanced: "What's the volatility smile?"
        In reality, IV varies by strike (smile/smirk). This violates 
        Black-Scholes assumptions. Far OTM puts have higher IV due to 
        crash risk. This is called "volatility skew."
    """
    
    # Validation: Check for arbitrage violations
    intrinsic = intrinsic_value(S, K, option_type)
    if option_price < intrinsic - 0.01:
        raise ValueError(
            f"Option price ${option_price:.2f} below intrinsic value ${intrinsic:.2f}. "
            f"This violates no-arbitrage"
        )
    
    # Edge case: at or past expiration
    if T <= 0:
        raise ValueError("Cannot calculate implied volatility at or past expiration")
    
    # Edge case: price equals intrinsic (zero time value)
    if abs(option_price - intrinsic) < 0.01:
        return 0.01  # Minimum vol (essentially zero time value)
    
    # Define the objective function: f(σ) = BS_price(σ) - market_price
    def objective(sigma):
        """Function to find root of (should equal zero at solution)."""
        if sigma <= 0:
            return 1e10  # Return large value for invalid sigma

        return black_scholes_price(S, K, T, r, sigma, option_type) - option_price

    
    # Define the derivative: f'(σ) = vega(σ) 
    def derivative(sigma):
        """Derivative of objective function (vega)."""
        if sigma <= 0:
            return 1e-10 # Return large value for invalid sigma

        # Get vega (per 1% vol change)
        option_vega = vega(S, K, T, r, sigma, option_type)
        
        # Scale vega from per-1% to per-100% for Newton-Raphson
        scaled_vega = option_vega * 100
        
        # Prevent division by zero
        if abs(scaled_vega) < 1e-10:
            return 1e-10
        
        return scaled_vega

    
    try:
        # Use scipy's Newton-Raphson solver
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
        
        # Final verification: does this sigma actually produce the market price?
        verification_price = black_scholes_price(S, K, T, r, solved_iv, option_type)
        if abs(verification_price - option_price) > 0.10:
            raise ValueError(
                f"IV solution verification failed. "
                f"Solved IV={solved_iv:.4f} gives price ${verification_price:.2f}, "
                f"but market price is ${option_price:.2f}"
            )
        
        return solved_iv
        
    except RuntimeError as e:
        # Newton-Raphson failed to converge
        raise ValueError(
            f"IV solver failed to converge after {max_iterations} iterations. "
            f"Option characteristics: S=${S:.2f}, K=${K:.2f}, T={T:.4f}, "
            f"Price=${option_price:.2f}, Intrinsic=${intrinsic:.2f}"
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
    
    This is more efficient than calling each Greek separately when you 
    need all of them (avoids recalculating d1, d2, etc.).
    
    Args:
        S: Current stock price
        K: Strike price
        T: Time to expiration (years)
        r: Risk-free rate (annual)
        sigma: Volatility (annual)
        option_type: 'call' or 'put'
    
    Returns:
        Dictionary with keys: 'delta', 'gamma', 'vega', 'theta', 'rho'
    
    Example:
        >>> greeks = calculate_all_greeks(100, 100, 1.0, 0.05, 0.20, 'call')
        >>> print(f"Delta: {greeks['delta']:.4f}")
        Delta: 0.6368
        >>> print(f"Gamma: {greeks['gamma']:.4f}")
        Gamma: 0.0184
    """
    # For efficiency, call each Greek function
    # (Each function is already optimized, no need to recalculate)
    return {
        'delta': delta(S, K, T, r, sigma, option_type),
        'gamma': gamma(S, K, T, r, sigma, option_type),
        'vega': vega(S, K, T, r, sigma, option_type),
        'theta': theta(S, K, T, r, sigma, option_type),
        'rho': rho(S, K, T, r, sigma, option_type)
    }


# Validation functions to test your implementation
def validate_greeks():
    """
    Test Greeks against known values.
    
    Run this to check if your implementation is correct!
    """
    print("=" * 60)
    print("GREEKS CALCULATOR VALIDATION")
    print("=" * 60)
    
    # Test case: ATM call, 1 year to expiry
    S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.20
    
    print(f"\nTest Case: ATM Call")
    print(f"S=${S}, K=${K}, T={T}y, r={r}, σ={sigma}")
    print("-" * 60)
    
    # Known values (approximate)
    tests = [
        ('delta', delta(S, K, T, r, sigma, 'call'), 0.637, 0.01),
        ('gamma', gamma(S, K, T, r, sigma, 'call'), 0.0184, 0.001),
        ('vega', vega(S, K, T, r, sigma, 'call'), 39.89, 0.5),
        ('theta', theta(S, K, T, r, sigma, 'call'), -0.0198, 0.005),
        ('rho', rho(S, K, T, r, sigma, 'call'), 51.75, 1.0),
    ]
    
    all_passed = True
    for name, actual, expected, tol in tests:
        if actual is None:
            print(f"❌ {name.upper()}: Not implemented yet")
            all_passed = False
        elif abs(actual - expected) < tol:
            print(f"✓ {name.upper()}: {actual:.4f} (expected ~{expected:.4f})")
        else:
            print(f"❌ {name.upper()}: {actual:.4f} (expected {expected:.4f}, diff={abs(actual-expected):.4f})")
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ ALL GREEKS TESTS PASSED!")
    else:
        print("❌ SOME TESTS FAILED - Debug and try again")
    print("=" * 60)


if __name__ == "__main__":
    # Run validation when script is executed
    validate_greeks()
    
    print("\n\n" + "=" * 60)
    print("IMPLIED VOLATILITY TEST")
    print("=" * 60)
    
    # Test IV solver
    S, K, T, r = 100, 100, 1.0, 0.05
    true_sigma = 0.20
    
    # First, get the market price for known vol
    market_price = black_scholes_price(S, K, T, r, true_sigma, 'call')
    print(f"\nMarket price: ${market_price:.4f} (with σ={true_sigma})")
    
    # Now solve for IV (should recover 0.20)
    try:
        solved_iv = implied_volatility(market_price, S, K, T, r, 'call')
        if solved_iv is None:
            print("❌ IV Solver: Not implemented yet")
        elif abs(solved_iv - true_sigma) < 0.001:
            print(f"✓ IV Solver: {solved_iv:.4f} (expected {true_sigma})")
            print("✓ IMPLIED VOLATILITY SOLVER WORKING!")
        else:
            print(f"❌ IV Solver: {solved_iv:.4f} (expected {true_sigma})")
    except Exception as e:
        print(f"❌ IV Solver Error: {e}")
    
    print("=" * 60)

