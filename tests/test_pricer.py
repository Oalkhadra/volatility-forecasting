"""
Unit tests for Black-Scholes pricer.

Run with: pytest tests/test_pricer.py -v
"""

import pytest
import numpy as np
from src.pricer import (
    black_scholes_price,
    intrinsic_value,
    time_value,
    moneyness
)


class TestBlackScholesPrice:
    """Test Black-Scholes pricing function."""
    
    def test_atm_call(self):
        """Test ATM call option pricing."""
        price = black_scholes_price(S=100, K=100, T=1.0, r=0.05, sigma=0.20, option_type='call')
        # Known value from Black-Scholes formula
        assert abs(price - 10.45) < 0.10, f"Expected ~10.45, got {price}"
    
    def test_atm_put(self):
        """Test ATM put option pricing."""
        price = black_scholes_price(S=100, K=100, T=1.0, r=0.05, sigma=0.20, option_type='put')
        # Known value from Black-Scholes formula
        assert abs(price - 5.57) < 0.10, f"Expected ~5.57, got {price}"
    
    def test_put_call_parity(self):
        """Test that put-call parity holds: C - P = S - K*e^(-rT)"""
        S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.20
        
        call_price = black_scholes_price(S, K, T, r, sigma, 'call')
        put_price = black_scholes_price(S, K, T, r, sigma, 'put')
        
        lhs = call_price - put_price
        rhs = S - K * np.exp(-r * T)
        
        assert abs(lhs - rhs) < 0.01, f"Put-call parity violated: {lhs} != {rhs}"
    
    def test_deep_itm_call(self):
        """Test deep ITM call behaves correctly."""
        price = black_scholes_price(S=150, K=100, T=0.5, r=0.05, sigma=0.20, option_type='call')
        intrinsic = 150 - 100
        # Deep ITM should be mostly intrinsic value
        assert price > intrinsic, "ITM call price should exceed intrinsic value"
        assert price < intrinsic + 10, "Deep ITM call should be close to intrinsic"
    
    def test_deep_otm_call(self):
        """Test deep OTM call has low price."""
        price = black_scholes_price(S=100, K=150, T=0.5, r=0.05, sigma=0.20, option_type='call')
        # Deep OTM should be cheap
        assert price < 5, f"Deep OTM call should be cheap, got {price}"
    
    def test_expiration_call(self):
        """Test option at expiration equals intrinsic value."""
        S, K = 110, 100
        price = black_scholes_price(S, K, T=0, r=0.05, sigma=0.20, option_type='call')
        expected = max(S - K, 0)
        assert abs(price - expected) < 0.01, f"At expiration, price should equal intrinsic"
    
    def test_expiration_put(self):
        """Test put at expiration equals intrinsic value."""
        S, K = 90, 100
        price = black_scholes_price(S, K, T=0, r=0.05, sigma=0.20, option_type='put')
        expected = max(K - S, 0)
        assert abs(price - expected) < 0.01, f"At expiration, price should equal intrinsic"
    
    def test_increasing_volatility_increases_price(self):
        """Test that higher volatility increases option price."""
        params = {'S': 100, 'K': 100, 'T': 1.0, 'r': 0.05, 'option_type': 'call'}
        
        price_low_vol = black_scholes_price(**params, sigma=0.10)
        price_high_vol = black_scholes_price(**params, sigma=0.30)
        
        assert price_high_vol > price_low_vol, "Higher vol should increase option price"
    
    def test_increasing_time_increases_price(self):
        """Test that more time increases option price."""
        params = {'S': 100, 'K': 100, 'r': 0.05, 'sigma': 0.20, 'option_type': 'call'}
        
        price_short = black_scholes_price(**params, T=0.25)
        price_long = black_scholes_price(**params, T=1.0)
        
        assert price_long > price_short, "More time should increase option price"
    
    def test_invalid_inputs(self):
        """Test that invalid inputs raise errors."""
        with pytest.raises(ValueError):
            black_scholes_price(S=-100, K=100, T=1.0, r=0.05, sigma=0.20)  # Negative S
        
        with pytest.raises(ValueError):
            black_scholes_price(S=100, K=-100, T=1.0, r=0.05, sigma=0.20)  # Negative K
        
        with pytest.raises(ValueError):
            black_scholes_price(S=100, K=100, T=-1.0, r=0.05, sigma=0.20)  # Negative T
        
        with pytest.raises(ValueError):
            black_scholes_price(S=100, K=100, T=1.0, r=0.05, sigma=-0.20)  # Negative sigma


class TestIntrinsicValue:
    """Test intrinsic value calculation."""
    
    def test_itm_call(self):
        """Test ITM call intrinsic value."""
        iv = intrinsic_value(S=110, K=100, option_type='call')
        assert iv == 10, f"ITM call intrinsic should be 10, got {iv}"
    
    def test_otm_call(self):
        """Test OTM call intrinsic value."""
        iv = intrinsic_value(S=90, K=100, option_type='call')
        assert iv == 0, f"OTM call intrinsic should be 0, got {iv}"
    
    def test_itm_put(self):
        """Test ITM put intrinsic value."""
        iv = intrinsic_value(S=90, K=100, option_type='put')
        assert iv == 10, f"ITM put intrinsic should be 10, got {iv}"
    
    def test_otm_put(self):
        """Test OTM put intrinsic value."""
        iv = intrinsic_value(S=110, K=100, option_type='put')
        assert iv == 0, f"OTM put intrinsic should be 0, got {iv}"


class TestTimeValue:
    """Test time value calculation."""
    
    def test_time_value_atm(self):
        """Test that ATM options have positive time value."""
        tv = time_value(option_price=10.45, S=100, K=100, option_type='call')
        assert tv > 0, "ATM option should have positive time value"
        assert tv == pytest.approx(10.45, abs=0.01), "ATM call should be all time value"


class TestMoneyness:
    """Test moneyness classification."""
    
    def test_atm(self):
        """Test ATM classification."""
        m = moneyness(S=100, K=100, option_type='call')
        assert m == 'ATM', f"Should be ATM, got {m}"
    
    def test_itm_call(self):
        """Test ITM call classification."""
        m = moneyness(S=110, K=100, option_type='call')
        assert m == 'ITM', f"Should be ITM, got {m}"
    
    def test_otm_call(self):
        """Test OTM call classification."""
        m = moneyness(S=90, K=100, option_type='call')
        assert m == 'OTM', f"Should be OTM, got {m}"
    
    def test_itm_put(self):
        """Test ITM put classification."""
        m = moneyness(S=90, K=100, option_type='put')
        assert m == 'ITM', f"Should be ITM, got {m}"
    
    def test_otm_put(self):
        """Test OTM put classification."""
        m = moneyness(S=110, K=100, option_type='put')
        assert m == 'OTM', f"Should be OTM, got {m}"


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, '-v'])

