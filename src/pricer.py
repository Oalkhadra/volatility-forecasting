"""
Black-Scholes option pricing model.

This module implements the Black-Scholes-Merton formula for European options.
Used to calculate theoretical option prices given volatility estimates.

Key formulas:
    d1 = [ln(S/K) + (r + σ²/2)T] / (σ√T)
    d2 = d1 - σ√T
    
    Call = S·N(d1) - K·e^(-rT)·N(d2)
    Put = K·e^(-rT)·N(-d2) - S·N(-d1)

Where:
    S = Current stock price
    K = Strike price
    T = Time to expiration (years)
    r = Risk-free rate (annual)
    σ = Volatility (annual)
    N(x) = Cumulative standard normal distribution
"""

import numpy as np
from scipy.stats import norm
from typing import Union, Literal


def black_scholes_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: Literal['call', 'put'] = 'call'
) -> float:
    """
    Calculate European option price using Black-Scholes model.
    
    Args:
        S: Current underlying price
        K: Strike price
        T: Time to expiration in years (e.g., 30 days = 30/365)
        r: Risk-free rate (annualized, as decimal, e.g., 0.02 for 2%)
        sigma: Implied volatility (annualized, as decimal, e.g., 0.20 for 20%)
        option_type: 'call' or 'put'
        
    Returns:
        Theoretical option price
        
    Example:
        >>> price = black_scholes_price(S=100, K=100, T=1.0, r=0.05, sigma=0.20)
        >>> print(f"Call price: ${price:.2f}")
        Call price: $10.45
        
    TODO: Implement Black-Scholes formula
    
    Hints:
        1. Handle edge case: T = 0 (expiration) → return intrinsic value
        2. Handle edge case: sigma = 0 → return discounted intrinsic value
        3. Calculate d1 = [ln(S/K) + (r + sigma²/2)*T] / (sigma*sqrt(T))
        4. Calculate d2 = d1 - sigma*sqrt(T)
        5. Use scipy.stats.norm.cdf() for N(d1) and N(d2)
        6. Calculate call = S*N(d1) - K*exp(-r*T)*N(d2)
        7. For put, use put-call parity or put formula directly
        
    Notes:
        - All inputs should be positive (except r can be negative in rare cases)
        - T is in YEARS (divide days by 365)
        - sigma and r are ANNUALIZED rates
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
    
    # TODO: Handle edge case - at expiration (T = 0)
    # Hint: Return max(S - K, 0) for call or max(K - S, 0) for put
    
    # TODO: Handle edge case - zero volatility (sigma = 0)
    # Hint: Return discounted intrinsic value
    
    # TODO: Calculate d1
    # Formula: d1 = [ln(S/K) + (r + sigma²/2)*T] / (sigma*sqrt(T))
    # Use: np.log() for natural log, np.sqrt() for square root
    
    # TODO: Calculate d2
    # Formula: d2 = d1 - sigma*sqrt(T)
    
    # TODO: Calculate cumulative normal distributions
    # Use: norm.cdf(d1) and norm.cdf(d2)
    
    # TODO: Calculate option price based on type
    # Call: S*N(d1) - K*exp(-r*T)*N(d2)
    # Put: K*exp(-r*T)*N(-d2) - S*N(-d1)
    # Use: np.exp() for exponential
    
    pass  # Remove this after implementation


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
        Intrinsic value (always >= 0)
        
    TODO: Implement intrinsic value calculation
    
    Hints:
        - Call intrinsic value: max(S - K, 0)
        - Put intrinsic value: max(K - S, 0)
    """
    # TODO: Implement
    pass


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
        Time value (can be negative if option trades below intrinsic)
        
    TODO: Implement time value calculation
    
    Hint: Use intrinsic_value() function
    """
    # TODO: Implement
    pass


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
        
    TODO: Implement moneyness classification
    
    Hints:
        - Use 0.5% threshold for ATM (|S-K|/S < 0.005)
        - Call ITM: S > K
        - Put ITM: K > S
    """
    # TODO: Implement
    pass


# Validation function to test your implementation
def validate_pricer():
    """
    Test Black-Scholes implementation against known values.
    
    These are standard test cases from academic literature.
    Your implementation should match within 0.01.
    """
    print("="*60)
    print("BLACK-SCHOLES PRICER VALIDATION")
    print("="*60)
    
    test_cases = [
        {
            'name': 'ATM Call (Hull Example 13.6)',
            'params': {'S': 42, 'K': 40, 'T': 0.5, 'r': 0.10, 'sigma': 0.20},
            'type': 'call',
            'expected': 4.76
        },
        {
            'name': 'ATM Put (Hull Example 13.6)',
            'params': {'S': 42, 'K': 40, 'T': 0.5, 'r': 0.10, 'sigma': 0.20},
            'type': 'put',
            'expected': 0.81
        },
        {
            'name': 'OTM Call',
            'params': {'S': 100, 'K': 110, 'T': 1.0, 'r': 0.05, 'sigma': 0.25},
            'type': 'call',
            'expected': 8.02
        },
        {
            'name': 'ITM Put',
            'params': {'S': 100, 'K': 110, 'T': 1.0, 'r': 0.05, 'sigma': 0.25},
            'type': 'put',
            'expected': 12.35
        },
        {
            'name': 'Deep ITM Call',
            'params': {'S': 150, 'K': 100, 'T': 0.25, 'r': 0.05, 'sigma': 0.30},
            'type': 'call',
            'expected': 51.39
        }
    ]
    
    all_passed = True
    
    for test in test_cases:
        try:
            result = black_scholes_price(**test['params'], option_type=test['type'])
            diff = abs(result - test['expected'])
            passed = diff < 0.10  # Allow 10 cent difference
            
            status = "✓ PASS" if passed else "✗ FAIL"
            print(f"\n{status} - {test['name']}")
            print(f"  Expected: ${test['expected']:.2f}")
            print(f"  Got:      ${result:.2f}")
            print(f"  Diff:     ${diff:.2f}")
            
            if not passed:
                all_passed = False
        except Exception as e:
            print(f"\n✗ ERROR - {test['name']}")
            print(f"  {str(e)}")
            all_passed = False
    
    print("\n" + "="*60)
    if all_passed:
        print("✓ ALL TESTS PASSED - Pricer is working correctly!")
    else:
        print("✗ SOME TESTS FAILED - Review your implementation")
    print("="*60)
    
    return all_passed


if __name__ == "__main__":
    # Run validation when script is executed
    validate_pricer()

