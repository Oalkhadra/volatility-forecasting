import pandas as pd
import yfinance as yf


def get_risk_free_rate(
    start_date: str,
    end_date: str,
    tenor: str = '^IRX'  # 13-week T-bill
) -> pd.DataFrame:
    """
    Download Treasury rate data from Yahoo Finance.
    
    Args:
        start_date: Start date 'YYYY-MM-DD'
        end_date: End date 'YYYY-MM-DD'
        tenor: Yahoo ticker for Treasury rate
               '^IRX' = 13-week (default, best for short-dated options)
               '^FVX' = 5-year
               '^TNX' = 10-year
    
    Returns:
        DataFrame with columns ['date', 'risk_free_rate']
        Rate is in decimal form (e.g., 0.02 for 2%)
        
    """
    try:
        # Download data using yfinance
        print(f"  Downloading {tenor} treasury rate data...")
        rf_data = yf.download(tenor, start=start_date, end=end_date, progress=False, auto_adjust=False)
        
        if rf_data.empty:
            print("  Warning: No treasury data available, using constant 2.0%")
            return _create_constant_rate(start_date, end_date, 0.02)
        
        # Reset index and extract date
        rf_data = rf_data.reset_index()
        
        # Flatten MultiIndex columns if they exist
        if isinstance(rf_data.columns, pd.MultiIndex):
            rf_data.columns = [col[0] if col[0] else col[1] for col in rf_data.columns.values]
      
        # Convert to decimal
        rf_data['risk_free_rate'] = rf_data['Close'] / 100
        rf_data['date'] = pd.to_datetime(rf_data['Date']).dt.strftime('%Y-%m-%d')
        
        # Keep only relevant columns
        rf_data = rf_data[['date', 'risk_free_rate']]
        
        # Forward fill missing dates (weekends/holidays)
        all_dates = pd.date_range(start_date, end_date, freq='D')
        full_df = pd.DataFrame({'date': all_dates.strftime('%Y-%m-%d')})
        merged = full_df.merge(rf_data, on='date', how='left')
        merged['risk_free_rate'] = merged['risk_free_rate'].ffill()

        # Backward fill any leading NaNs
        merged['risk_free_rate'] = merged['risk_free_rate'].bfill()
        print(f"  ✓ Downloaded rate data: mean={merged['risk_free_rate'].mean():.4f}")
        
        return merged
        
    except Exception as e:
        print(f"  Error downloading treasury data: {e}")
        print("  Using constant 2.0% as fallback")
        return _create_constant_rate(start_date, end_date, 0.02)


def _create_constant_rate(
    start_date: str,
    end_date: str,
    rate: float
) -> pd.DataFrame:
    """Create dataframe with constant risk-free rate."""
    dates = pd.date_range(start_date, end_date, freq='D')
    return pd.DataFrame({
        'date': dates.strftime('%Y-%m-%d'),
        'risk_free_rate': rate
    })
