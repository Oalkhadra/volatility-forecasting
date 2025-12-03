"""
Volatility estimation module.

This package contains three different approaches to estimating volatility:
- historical: Realized historical volatility (Strategy B)
- surface: Volatility surface fitting (Strategy C)
- forecasting: Statistical forecasting models (Strategy D)
"""

from .historical import calculate_realized_volatility, RollingVolCalculator
from .surface import fit_volatility_surface, VolatilitySurface
from .forecasting import GARCHForecaster, MLForecaster

__all__ = [
    'calculate_realized_volatility',
    'RollingVolCalculator',
    'fit_volatility_surface',
    'VolatilitySurface',
    'GARCHForecaster',
    'MLForecaster'
]

