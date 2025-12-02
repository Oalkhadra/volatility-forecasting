"""
Unit tests for Greeks calculator.

Run with: pytest tests/test_greeks.py -v
"""

import pytest
import numpy as np
from src.greeks import (
    delta,
    gamma,
    vega,
    theta,
    rho,
    implied_volatility,
    calculate_all_greeks
)
from src.pricer import black_scholes_price


class TestDelta:
    """Test delta calculation."""
    
    def test_atm_call_delta(self):
        """ATM call delta should be around 0.5-0.6."""
        d = delta(S=100, K=100, T=1.0, r=0.05, sigma=0.20, option_type='call')
        assert 0.5 < d < 0.7, f"ATM call delta should be ~0.5-0.6, got {d}"
    
    def test_atm_put_delta(self):
        """ATM put delta should be around -0.4 to -0.5."""
        d = delta(S=100, K=100, T=1.0, r=0.05, sigma=0.20, option_type='put')
        assert -0.5 < d < -0.3, f"ATM put delta should be ~-0.4 to -0.5, got {d}"
    
    def test_deep_itm_call_delta(self):
        """Deep ITM call delta should approach 1."""
        d = delta(S=150, K=100, T=1.0, r=0.05, sigma=0.20, option_type='call')
        assert d > 0.9, f"Deep ITM call delta should be close to 1, got {d}"
    
    def test_deep_otm_call_delta(self):
        """Deep OTM call delta should approach 0."""
        d = delta(S=100, K=150, T=1.0, r=0.05, sigma=0.20, option_type='call')
        assert d < 0.1, f"Deep OTM call delta should be close to 0, got {d}"
    
    def test_put_call_delta_relationship(self):
        """Call delta - Put delta = 1."""
        S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.20
        call_delta = delta(S, K, T, r, sigma, 'call')
        put_delta = delta(S, K, T, r, sigma, 'put')
        assert abs(call_delta - put_delta - 1.0) < 0.01, "Call delta - Put delta should equal 1"


class TestGamma:
    """Test gamma calculation."""
    
    def test_gamma_positive(self):
        """Gamma should always be positive."""
        g = gamma(S=100, K=100, T=1.0, r=0.05, sigma=0.20, option_type='call')
        assert g > 0, f"Gamma should be positive, got {g}"
    
    def test_gamma_same_for_call_and_put(self):
        """Gamma should be same for calls and puts."""
        params = {'S': 100, 'K': 100, 'T': 1.0, 'r': 0.05, 'sigma': 0.20}
        call_gamma = gamma(**params, option_type='call')
        put_gamma = gamma(**params, option_type='put')
        assert abs(call_gamma - put_gamma) < 1e-10, "Call and put gamma should be identical"
    
    def test_atm_gamma_highest(self):
        """ATM options should have highest gamma."""
        params = {'S': 100, 'T': 1.0, 'r': 0.05, 'sigma': 0.20, 'option_type': 'call'}
        atm_gamma = gamma(**params, K=100)
        itm_gamma = gamma(**params, K=80)
        otm_gamma = gamma(**params, K=120)
        
        assert atm_gamma > itm_gamma, "ATM gamma should exceed ITM gamma"
        assert atm_gamma > otm_gamma, "ATM gamma should exceed OTM gamma"
    
    def test_gamma_increases_near_expiration(self):
        """Gamma should increase as expiration approaches."""
        params = {'S': 100, 'K': 100, 'r': 0.05, 'sigma': 0.20, 'option_type': 'call'}
        gamma_long = gamma(**params, T=1.0)
        gamma_short = gamma(**params, T=0.1)
        
        assert gamma_short > gamma_long, "Short-dated options should have higher gamma"


class TestVega:
    """Test vega calculation."""
    
    def test_vega_positive(self):
        """Vega should always be positive (long options benefit from vol increase)."""
        v = vega(S=100, K=100, T=1.0, r=0.05, sigma=0.20, option_type='call')
        assert v > 0, f"Vega should be positive, got {v}"
    
    def test_vega_same_for_call_and_put(self):
        """Vega should be same for calls and puts."""
        params = {'S': 100, 'K': 100, 'T': 1.0, 'r': 0.05, 'sigma': 0.20}
        call_vega = vega(**params, option_type='call')
        put_vega = vega(**params, option_type='put')
        assert abs(call_vega - put_vega) < 1e-10, "Call and put vega should be identical"
    
    def test_vega_increases_with_time(self):
        """Longer-dated options should have higher vega."""
        params = {'S': 100, 'K': 100, 'r': 0.05, 'sigma': 0.20, 'option_type': 'call'}
        vega_short = vega(**params, T=0.25)
        vega_long = vega(**params, T=1.0)
        
        assert vega_long > vega_short, "Longer maturity should have higher vega"
    
    def test_atm_vega_highest(self):
        """ATM options should have highest vega."""
        params = {'S': 100, 'T': 1.0, 'r': 0.05, 'sigma': 0.20, 'option_type': 'call'}
        atm_vega = vega(**params, K=100)
        itm_vega = vega(**params, K=80)
        otm_vega = vega(**params, K=120)
        
        assert atm_vega > itm_vega, "ATM vega should exceed ITM vega"
        assert atm_vega > otm_vega, "ATM vega should exceed OTM vega"


class TestTheta:
    """Test theta calculation."""
    
    def test_theta_negative_for_long_call(self):
        """Long calls typically have negative theta."""
        t = theta(S=100, K=100, T=1.0, r=0.05, sigma=0.20, option_type='call')
        assert t < 0, f"Long call theta should be negative, got {t}"
    
    def test_theta_negative_for_long_put(self):
        """Long puts typically have negative theta."""
        t = theta(S=100, K=100, T=1.0, r=0.05, sigma=0.20, option_type='put')
        # Note: Deep ITM puts can have positive theta due to interest
        # But ATM puts should be negative
        assert t < 0, f"ATM long put theta should be negative, got {t}"
    
    def test_theta_increases_near_expiration(self):
        """Theta magnitude should increase as expiration approaches."""
        params = {'S': 100, 'K': 100, 'r': 0.05, 'sigma': 0.20, 'option_type': 'call'}
        theta_long = theta(**params, T=1.0)
        theta_short = theta(**params, T=0.1)
        
        # Both negative, but short-dated should be MORE negative
        assert abs(theta_short) > abs(theta_long), "Short-dated theta should be more negative"


class TestRho:
    """Test rho calculation."""
    
    def test_call_rho_positive(self):
        """Call rho should be positive."""
        r_val = rho(S=100, K=100, T=1.0, r=0.05, sigma=0.20, option_type='call')
        assert r_val > 0, f"Call rho should be positive, got {r_val}"
    
    def test_put_rho_negative(self):
        """Put rho should be negative."""
        r_val = rho(S=100, K=100, T=1.0, r=0.05, sigma=0.20, option_type='put')
        assert r_val < 0, f"Put rho should be negative, got {r_val}"
    
    def test_rho_increases_with_maturity(self):
        """Longer maturity options should have higher rho magnitude."""
        params = {'S': 100, 'K': 100, 'r': 0.05, 'sigma': 0.20, 'option_type': 'call'}
        rho_short = rho(**params, T=0.25)
        rho_long = rho(**params, T=1.0)
        
        assert abs(rho_long) > abs(rho_short), "Longer maturity should have higher rho"


class TestImpliedVolatility:
    """Test implied volatility solver."""
    
    def test_iv_recovers_known_vol(self):
        """Given a BS price, IV solver should recover the original volatility."""
        S, K, T, r, true_sigma = 100, 100, 1.0, 0.05, 0.25
        
        # Get market price with known vol
        market_price = black_scholes_price(S, K, T, r, true_sigma, 'call')
        
        # Solve for IV
        solved_iv = implied_volatility(market_price, S, K, T, r, 'call')
        
        assert abs(solved_iv - true_sigma) < 0.001, \
            f"IV solver should recover vol: expected {true_sigma}, got {solved_iv}"
    
    def test_iv_works_for_puts(self):
        """IV solver should work for puts too."""
        S, K, T, r, true_sigma = 100, 100, 1.0, 0.05, 0.30
        
        market_price = black_scholes_price(S, K, T, r, true_sigma, 'put')
        solved_iv = implied_volatility(market_price, S, K, T, r, 'put')
        
        assert abs(solved_iv - true_sigma) < 0.001, \
            f"IV solver for puts: expected {true_sigma}, got {solved_iv}"
    
    def test_iv_works_for_itm_options(self):
        """IV solver should work for ITM options."""
        S, K, T, r, true_sigma = 120, 100, 0.5, 0.05, 0.20
        
        market_price = black_scholes_price(S, K, T, r, true_sigma, 'call')
        solved_iv = implied_volatility(market_price, S, K, T, r, 'call')
        
        assert abs(solved_iv - true_sigma) < 0.001, \
            f"IV solver for ITM: expected {true_sigma}, got {solved_iv}"
    
    def test_iv_works_for_otm_options(self):
        """IV solver should work for OTM options."""
        S, K, T, r, true_sigma = 100, 120, 0.5, 0.05, 0.35
        
        market_price = black_scholes_price(S, K, T, r, true_sigma, 'call')
        solved_iv = implied_volatility(market_price, S, K, T, r, 'call')
        
        assert abs(solved_iv - true_sigma) < 0.001, \
            f"IV solver for OTM: expected {true_sigma}, got {solved_iv}"
    
    def test_iv_rejects_price_below_intrinsic(self):
        """IV solver should raise error if price < intrinsic value."""
        S, K, T, r = 120, 100, 1.0, 0.05
        intrinsic = 20  # S - K
        invalid_price = 15  # Less than intrinsic
        
        with pytest.raises(ValueError):
            implied_volatility(invalid_price, S, K, T, r, 'call')
    
    def test_iv_convergence_with_different_initial_guesses(self):
        """IV solver should converge regardless of initial guess."""
        S, K, T, r, true_sigma = 100, 100, 1.0, 0.05, 0.25
        market_price = black_scholes_price(S, K, T, r, true_sigma, 'call')
        
        # Try different initial guesses
        for initial in [0.10, 0.25, 0.50, 1.0]:
            solved_iv = implied_volatility(
                market_price, S, K, T, r, 'call', 
                initial_guess=initial
            )
            assert abs(solved_iv - true_sigma) < 0.001, \
                f"IV solver with initial={initial}: expected {true_sigma}, got {solved_iv}"


class TestCalculateAllGreeks:
    """Test combined Greeks calculation."""
    
    def test_all_greeks_returns_dict(self):
        """calculate_all_greeks should return a dictionary with all Greeks."""
        greeks = calculate_all_greeks(100, 100, 1.0, 0.05, 0.20, 'call')
        
        assert isinstance(greeks, dict), "Should return a dictionary"
        assert 'delta' in greeks, "Should contain delta"
        assert 'gamma' in greeks, "Should contain gamma"
        assert 'vega' in greeks, "Should contain vega"
        assert 'theta' in greeks, "Should contain theta"
        assert 'rho' in greeks, "Should contain rho"
    
    def test_all_greeks_matches_individual_calls(self):
        """Values from calculate_all_greeks should match individual Greek functions."""
        S, K, T, r, sigma, opt_type = 100, 100, 1.0, 0.05, 0.20, 'call'
        
        greeks = calculate_all_greeks(S, K, T, r, sigma, opt_type)
        
        assert abs(greeks['delta'] - delta(S, K, T, r, sigma, opt_type)) < 1e-10
        assert abs(greeks['gamma'] - gamma(S, K, T, r, sigma, opt_type)) < 1e-10
        assert abs(greeks['vega'] - vega(S, K, T, r, sigma, opt_type)) < 1e-10
        assert abs(greeks['theta'] - theta(S, K, T, r, sigma, opt_type)) < 1e-10
        assert abs(greeks['rho'] - rho(S, K, T, r, sigma, opt_type)) < 1e-10


class TestGreeksNumericalProperties:
    """Test numerical relationships between Greeks."""
    
    def test_delta_approximates_finite_difference(self):
        """Delta should approximate (dPrice/dS) using finite differences."""
        S, K, T, r, sigma, opt_type = 100, 100, 1.0, 0.05, 0.20, 'call'
        
        # Analytical delta
        analytical_delta = delta(S, K, T, r, sigma, opt_type)
        
        # Numerical delta: (Price(S+h) - Price(S-h)) / (2h)
        h = 0.01
        price_up = black_scholes_price(S + h, K, T, r, sigma, opt_type)
        price_down = black_scholes_price(S - h, K, T, r, sigma, opt_type)
        numerical_delta = (price_up - price_down) / (2 * h)
        
        assert abs(analytical_delta - numerical_delta) < 0.001, \
            f"Delta should match finite difference: {analytical_delta} vs {numerical_delta}"
    
    def test_gamma_approximates_second_derivative(self):
        """Gamma should approximate (d²Price/dS²) using finite differences."""
        S, K, T, r, sigma, opt_type = 100, 100, 1.0, 0.05, 0.20, 'call'
        
        # Analytical gamma
        analytical_gamma = gamma(S, K, T, r, sigma, opt_type)
        
        # Numerical gamma: (Price(S+h) - 2*Price(S) + Price(S-h)) / h²
        h = 0.01
        price_up = black_scholes_price(S + h, K, T, r, sigma, opt_type)
        price_mid = black_scholes_price(S, K, T, r, sigma, opt_type)
        price_down = black_scholes_price(S - h, K, T, r, sigma, opt_type)
        numerical_gamma = (price_up - 2 * price_mid + price_down) / (h ** 2)
        
        assert abs(analytical_gamma - numerical_gamma) < 0.001, \
            f"Gamma should match finite difference: {analytical_gamma} vs {numerical_gamma}"
    
    def test_vega_approximates_vol_sensitivity(self):
        """Vega should approximate (dPrice/dσ) using finite differences."""
        S, K, T, r, sigma, opt_type = 100, 100, 1.0, 0.05, 0.20, 'call'
        
        # Analytical vega
        analytical_vega = vega(S, K, T, r, sigma, opt_type) * 100
        
        # Numerical vega: (Price(σ+h) - Price(σ-h)) / (2h)
        # Note: our vega is per 1% vol, so we need to scale
        h = 0.01  # 1% vol change
        price_up = black_scholes_price(S, K, T, r, sigma + h, opt_type)
        price_down = black_scholes_price(S, K, T, r, sigma - h, opt_type)
        numerical_vega = (price_up - price_down) / (2 * h)
        
        assert abs(analytical_vega - numerical_vega) < 0.1, \
            f"Vega should match finite difference: {analytical_vega} vs {numerical_vega}"


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, '-v'])

