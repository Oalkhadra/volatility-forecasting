import numpy as np
import pandas as pd
from typing import Optional, Dict
from dataclasses import dataclass


@dataclass
class VRPEstimate:
    """Container for VRP estimate with metadata."""
    vrp: float                    # Point estimate
    regime: str                   # Current regime classification
    sample_size: int              # Number of observations used


class VarianceRiskPremiumCalculator:
    """
    Calculate and manage variance risk premium estimates.
    
    Uses regime-based VRP: computes historical VRP statistics for each
    VIX regime (low/medium/high) and returns the regime mean at inference time.
    """
    
    # Regime thresholds (VIX levels, in percentage points)
    # These are based on historical VIX distribution
    REGIME_THRESHOLDS = {
        'low': 15.0,      # VIX < 15: calm markets
        'medium': 25.0,   # 15 <= VIX < 25: normal markets
                          # VIX >= 25: high volatility / crisis
    }
    
    def __init__(self, 
                 prices_df: pd.DataFrame,
                 vix_df: pd.DataFrame,
                 rv_window: int = 30):
        """
        Initialize VRP calculator.
        
        Args:
            prices_df: DataFrame with 'date' and 'close' columns
            vix_df: DataFrame with 'date' and 'vix_close' columns
            rv_window: Window for calculating realized volatility (days)
        """
        self.prices_df = prices_df.copy()
        self.vix_df = vix_df.copy()
        self.rv_window = rv_window
        
        # Ensure date columns are datetime
        self.prices_df['date'] = pd.to_datetime(self.prices_df['date'])
        self.vix_df['date'] = pd.to_datetime(self.vix_df['date'])
        
        # Store full VIX series for regime lookups during backtest
        self._full_vix_lookup = self.vix_df.set_index('date')['vix_close'].to_dict()
        
        # Will be populated by fit()
        self.vrp_series: Optional[pd.DataFrame] = None
        self.regime_vrp: Optional[Dict[str, Dict[str, float]]] = None
        self.is_fitted = False
        
    def _calculate_realized_volatility(self, df: pd.DataFrame) -> pd.Series:
        """Calculate rolling realized volatility from prices."""
        log_returns = np.log(df['close'] / df['close'].shift(1))
        rv = log_returns.rolling(window=self.rv_window).std() * np.sqrt(252)

        return rv
    
    def _classify_regime(self, vix_level: float) -> str:
        """
        Classify market regime based on VIX level.
        
        Simple threshold-based classification:
        - Low: VIX < 15 (calm, complacent markets)
        - Medium: 15 <= VIX < 25 (normal uncertainty)
        - High: VIX >= 25 (elevated fear, potential crisis)
        """
        if vix_level < self.REGIME_THRESHOLDS['low']:
            return 'low'
        elif vix_level < self.REGIME_THRESHOLDS['medium']:
            return 'medium'
        else:
            return 'high'
    
    def fit(self, end_date: Optional[pd.Timestamp] = None) -> 'VarianceRiskPremiumCalculator':
        """
        Estimate VRP from historical data.
        
        Args:
            end_date: Only use data before this date (for walk-forward)
                     If None, uses all available data
                     
        Returns:
            self (for method chaining)
        """
        # Merge prices and VIX
        df = self.prices_df.merge(self.vix_df, on='date', how='inner')
        
        # Filter to historical data
        if end_date is not None:
            end_date = pd.to_datetime(end_date)
            df = df[df['date'] < end_date].copy()
        
        # Need at least 2x rv_window for forward-looking calculation
        min_required = self.rv_window * 2
        if len(df) < min_required:
            raise ValueError(
                f"Insufficient data for VRP estimation. "
                f"Need at least {min_required} days, "
                f"have {len(df)}"
            )
        
        # Calculate realized volatility
        df['rv'] = self._calculate_realized_volatility(df)
        
        # Convert VIX to decimal
        df['iv'] = df['vix_close'] / 100
        
        # Calculate forward-looking realized vol
        df['forward_rv'] = df['rv'].shift(-self.rv_window)
        
        # VRP = IV at time t minus RV that actually occurred over next window
        df['vrp'] = df['iv'] - df['forward_rv']
        
        # Classify regimes
        df['regime'] = df['vix_close'].apply(self._classify_regime)
        
        # Store the series
        self.vrp_series = df[['date', 'vix_close', 'iv', 'rv', 'forward_rv', 
                              'vrp', 'regime']].copy()
        
        # Calculate regime-specific VRP
        self._fit_regime_vrp(df)
        
        self.is_fitted = True
        print(f"VRP Calculator fitted on {len(df)} observations")
        print(f"  Date range: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")
        print(f"  Mean VRP: {df['vrp'].mean():.4f} ({df['vrp'].mean()*100:.2f}%)")
        print(f"  Std VRP: {df['vrp'].std():.4f}")
        
        return self
    
    def _fit_regime_vrp(self, df: pd.DataFrame) -> None:
        """Calculate VRP statistics by regime."""
        # Only use rows where we have forward_rv
        valid_df = df.dropna(subset=['vrp'])
        
        regime_stats = {}
        for regime in ['low', 'medium', 'high']:
            regime_data = valid_df[valid_df['regime'] == regime]['vrp']
            
            if len(regime_data) > 0:
                regime_stats[regime] = {
                    'mean': regime_data.mean(),
                    'std': regime_data.std(),
                    'median': regime_data.median(),
                    'count': len(regime_data),
                    'q25': regime_data.quantile(0.25),
                    'q75': regime_data.quantile(0.75)
                }
            else:
                # Fallback if no data for regime
                regime_stats[regime] = {
                    'mean': df['vrp'].mean(),
                    'std': df['vrp'].std(),
                    'median': df['vrp'].median(),
                    'count': 0,
                    'q25': df['vrp'].quantile(0.25),
                    'q75': df['vrp'].quantile(0.75)
                }
        
        self.regime_vrp = regime_stats
        
        print("\n  Regime-Conditional VRP:")
        for regime, stats in regime_stats.items():
            print(f"    {regime.upper():8s}: mean={stats['mean']:.4f} "
                  f"({stats['mean']*100:.2f}%), n={stats['count']}")
    
    def get_vrp(self, 
                current_date: pd.Timestamp,
                current_vix: float = None
                ) -> VRPEstimate:
        """
        Get VRP estimate for a given date.
        
        Args:
            current_date: Date to get VRP for
            current_vix: Current VIX level for regime classification.
                
        Returns:
            VRP Estimate with VRP value and metadata
        """
        if not self.is_fitted:
            raise ValueError("Must call fit() before get_vrp()")
        
        current_date = pd.to_datetime(current_date)
        
        # Get current VIX for regime classification
        if current_vix is not None:
            vix_for_regime = current_vix
        elif current_date in self._full_vix_lookup:
            vix_for_regime = self._full_vix_lookup[current_date]
        else:
            # Try to find closest date in full VIX lookup
            vix_dates = list(self._full_vix_lookup.keys())
            closest_date = min(vix_dates, key=lambda d: abs((d - current_date).days))
            if abs((closest_date - current_date).days) <= 5:
                vix_for_regime = self._full_vix_lookup[closest_date]
            else:
                # Fallback to last available in vrp_series
                historical = self.vrp_series[self.vrp_series['date'] <= current_date]
                if len(historical) == 0:
                    raise ValueError(f"No VRP data available for {current_date}")
                vix_for_regime = historical.iloc[-1]['vix_close']
        
        current_regime = self._classify_regime(vix_for_regime)

        # Use regime-specific mean from fit()
        if self.regime_vrp is None:
            raise ValueError("Regime VRP not fitted. Call fit() before get_vrp().")
        vrp_value = self.regime_vrp[current_regime]['mean']
        sample_size = self.regime_vrp[current_regime]['count']
        
        return VRPEstimate(
            vrp=vrp_value if not pd.isna(vrp_value) else 0.04,  # 4% fallback
            regime=current_regime,
            sample_size=sample_size
        )
    
    def get_vrp_adjustment(self, 
                          current_date: pd.Timestamp,
                          current_vix: float = None
                          ) -> float:
        """
        Convenience method to get just the VRP value.
        
        Args:
            current_date: Date to get VRP for
            current_vix: Current VIX level for regime classification.
                        If None, looks up from stored VIX data.
            
        Returns:
            VRP adjustment (add to RV forecast to get IV estimate)
        """
        estimate = self.get_vrp(current_date, current_vix=current_vix)
        return estimate.vrp
    
    def get_analysis_summary(self) -> pd.DataFrame:
        """
        Get summary statistics for analysis/reporting.
        
        Returns:
            DataFrame with VRP statistics by regime and overall
        """
        if not self.is_fitted:
            raise ValueError("Must call fit() first")
        
        results = []
        
        # Overall stats
        valid_vrp = self.vrp_series['vrp'].dropna()
        results.append({
            'regime': 'Overall',
            'mean_vrp': valid_vrp.mean(),
            'std_vrp': valid_vrp.std(),
            'median_vrp': valid_vrp.median(),
            'min_vrp': valid_vrp.min(),
            'max_vrp': valid_vrp.max(),
            'count': len(valid_vrp),
            'pct_positive': (valid_vrp > 0).mean() * 100
        })
        
        # By regime
        if self.regime_vrp is not None:
            for regime, stats in self.regime_vrp.items():
                regime_data = self.vrp_series[
                    self.vrp_series['regime'] == regime
                ]['vrp'].dropna()
                
                results.append({
                    'regime': regime.capitalize(),
                    'mean_vrp': stats['mean'],
                    'std_vrp': stats['std'],
                    'median_vrp': stats['median'],
                    'min_vrp': regime_data.min() if len(regime_data) > 0 else np.nan,
                    'max_vrp': regime_data.max() if len(regime_data) > 0 else np.nan,
                    'count': stats['count'],
                    'pct_positive': (regime_data > 0).mean() * 100 if len(regime_data) > 0 else np.nan
                })
        
        return pd.DataFrame(results)


def calculate_historical_vrp(
    prices_df: pd.DataFrame,
    vix_df: pd.DataFrame,
    rv_window: int = 30
) -> pd.DataFrame:
    """
    Standalone function to calculate historical VRP series.
    
    Useful for analysis without the full calculator class.
    
    Args:
        prices_df: DataFrame with 'date' and 'close' columns
        vix_df: DataFrame with 'date' and 'vix_close' columns
        rv_window: Window for realized volatility calculation
        
    Returns:
        DataFrame with columns: date, vix, rv, vrp, regime
    """
    calc = VarianceRiskPremiumCalculator(
        prices_df, vix_df, rv_window
    )
    calc.fit()
    return calc.vrp_series
