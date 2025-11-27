"""
Data preprocessing script for SPY and QQQ options data.

This script queries the Dolt database and saves processed data to parquet files.
"""

import subprocess
import pandas as pd
from pathlib import Path
from typing import Optional, List
from io import StringIO
import yfinance as yf
from risk_free_rate import get_risk_free_rate

# Constants
DOLT_REPO_PATH = "data/raw/options"
OUTPUT_DIR = "data/processed"
START = '2019-06-03'
END = '2020-01-01'

def run_dolt_query(query: str, repo_path: str = DOLT_REPO_PATH) -> pd.DataFrame:
    """
    Execute a SQL query against the Dolt database.
    
    Args:
        query: SQL query string
        repo_path: Path to Dolt repository
        
    Returns:
        DataFrame with query results
        
    TODO: Implement subprocess call to dolt sql
    Hint: Use subprocess.run with ["dolt", "sql", "-q", query, "-r", "csv"]
    """
    try:
        # Execute dolt sql command
        result = subprocess.run(
            ["dolt", "sql", "-q", query, "-r", "csv"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=True
        )
        
        # Parse CSV output into DataFrame
        df = pd.read_csv(StringIO(result.stdout))
        
        return df
        
    except subprocess.CalledProcessError as e:
        print(f"Query failed: {e.stderr}")
        raise
    except FileNotFoundError:
        print(f"Dolt repository not found at {repo_path}")
        raise


def get_option_chain_data(
    ticker: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> pd.DataFrame:
    """
    Query option chain data for a specific ticker.
    
    Args:
        ticker: Ticker symbol (e.g., 'SPY', 'QQQ')
        start_date: Start date in 'YYYY-MM-DD' format (optional)
        end_date: End date in 'YYYY-MM-DD' format (optional)
        
    Returns:
        DataFrame with option chain data
        
    TODO: Build SQL query and call run_dolt_query
    Hint: SELECT * FROM option_chain WHERE act_symbol = '...'
    """
    df = run_dolt_query(f"SELECT * FROM option_chain WHERE act_symbol='{ticker}' AND date <= '{end_date}' AND date >= '{start_date}'")

    return df


def get_volatility_history(
    ticker: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> pd.DataFrame:
    """
    Query volatility history for a specific ticker.
    
    Args:
        ticker: Ticker symbol
        start_date: Start date in 'YYYY-MM-DD' format (optional)
        end_date: End date in 'YYYY-MM-DD' format (optional)
        
    Returns:
        DataFrame with volatility history
        
    TODO: Build SQL query for volatility_history table
    """
    df = run_dolt_query(f"SELECT * FROM volatility_history WHERE act_symbol='{ticker}' AND date <= '{end_date}' AND date >= '{start_date}'")

    return df


def clean_option_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and validate option chain data with liquidity filtering.
    """
    # Convert dates
    df['date'] = pd.to_datetime(df['date'])
    df['expiration'] = pd.to_datetime(df['expiration'])
    
    # Add derived columns
    df['mid_price'] = (df['bid'] + df['ask']) / 2
    df['spread'] = df['ask'] - df['bid']
    df['spread_pct'] = df['spread'] / df['mid_price']
    df['days_to_expiry'] = (df['expiration'] - df['date']).dt.days
    
    # Liquidity filters
    liquid_df = df[
        # Valid prices
        (df['bid'] > 0) &
        (df['ask'] > 0) &
        (df['bid'] <= df['ask']) &
        
        # Reasonable spread (< 10% is typical for liquid SPY/QQQ options)
        (df['spread_pct'] < 0.10) &
        
        # Standard expiration window
        (df['days_to_expiry'] >= 7) &
        (df['days_to_expiry'] <= 60) &
        
        # Reasonable option price (avoid near-zero options)
        (df['mid_price'] > 0.10)
    ].copy()
    
    liquid_df['date'] = liquid_df['date'].dt.strftime('%Y-%m-%d')

    return liquid_df


def clean_volatility_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and validate volatility history data.
    
    Args:
        df: Raw volatility data
        
    Returns:
        Cleaned DataFrame
    """
    
    # Filter for valid volatility values
    clean_df = df[
        # Historical volatility should be positive
        (df['hv_current'].notna()) &
        (df['hv_current'] > 0) &
        
        # Implied volatility should be positive
        (df['iv_current'].notna()) &
        (df['iv_current'] > 0) &
        
        # Remove extreme outliers (vol > 500% is unrealistic for SPY/QQQ)
        (df['hv_current'] < 5.0) &
        (df['iv_current'] < 5.0)
    ].copy()
    
    return clean_df


def get_underlying_price_data(
    ticker: str,
    start_date: str,
    end_date: str
) -> pd.DataFrame:
    """
    Download underlying price data from Yahoo Finance.
    
    Args:
        ticker: Ticker symbol
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
        
    Returns:
        DataFrame with OHLCV data
    """
    # Download data with UNADJUSTED prices (options use actual prices, not adjusted)
    df = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=False)
    
    # Reset index to make date a column
    df = df.reset_index()
    
    # Flatten multi-level column index (yfinance returns tuples like ('Close', 'SPY'))
    if isinstance(df.columns, pd.MultiIndex):
        # Extract the first level (the price columns like 'Close', 'High', etc.)
        df.columns = [col[0] if col[0] else col[1] for col in df.columns.values]
    
    # Normalize column names to lowercase
    df.columns = df.columns.str.lower()
    
    # Ensure we use 'close' column (not 'adj close') for unadjusted prices
    if 'adj close' in df.columns and 'close' in df.columns:
        # Remove adj close to avoid confusion
        df = df.drop(columns=['adj close'])
    
    # Add ticker column
    df['ticker'] = ticker.upper()

    df['date'] = df['date'].dt.strftime('%Y-%m-%d')

    return df


def merge_underlying_prices(
    options_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    ticker: str
) -> pd.DataFrame:
    """
    Merge options data with underlying prices using backward fill.
    
    Matches Saturday option snapshots with the most recent trading day's close
    (typically Friday). Uses pd.merge_asof to handle date mismatches.
    
    Args:
        options_df: Options dataframe with 'date' column (Saturday snapshots)
        prices_df: Prices dataframe with 'date' and 'close' columns (weekdays)
        ticker: Ticker symbol for filtering
        
    Returns:
        Merged dataframe with underlying_price column added
    """
    # Make copies to avoid modifying originals
    options_df = options_df.copy()
    prices_df = prices_df.copy()
    
    # Convert to datetime for merge_asof
    options_df['date'] = pd.to_datetime(options_df['date'])
    prices_df['date'] = pd.to_datetime(prices_df['date'])
    
    # Filter prices for this ticker (if multi-ticker dataframe)
    if 'ticker' in prices_df.columns:
        prices_df = prices_df[prices_df['ticker'] == ticker.upper()]
    
    # Sort both dataframes by date (required for merge_asof)
    options_df = options_df.sort_values('date')
    prices_df = prices_df.sort_values('date')
    
    # Merge using most recent price before or on the option date
    merged_df = pd.merge_asof(
        options_df,
        prices_df[['date', 'close', 'volume']],
        on='date',
        direction='backward'
    )
    
    # Rename columns for clarity
    merged_df = merged_df.rename(columns={
        'close': 'underlying_price',
        'volume': 'underlying_volume'
    })
    
    # Verify no missing underlying prices
    missing = merged_df['underlying_price'].isna().sum()
    if missing > 0:
        print(f"Warning: {missing} rows missing underlying price for {ticker}")
        # Drop rows with missing prices
        merged_df = merged_df.dropna(subset=['underlying_price'])
    
    # Convert date back to string format to match original format
    merged_df['date'] = merged_df['date'].dt.strftime('%Y-%m-%d')
    
    return merged_df


def merge_risk_free_rate(
    options_df: pd.DataFrame,
    rf_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge risk-free rate data with options data.
    
    Matches each option date with the corresponding risk-free rate.
    
    Args:
        options_df: Options dataframe with 'date' column
        rf_df: Risk-free rate dataframe with 'date' and 'risk_free_rate' columns
        
    Returns:
        Merged dataframe with risk_free_rate column added
    """
    # Make copies to avoid modifying originals
    options_df = options_df.copy()
    rf_df = rf_df.copy()
    
    # Convert to datetime for merge_asof
    options_df['date'] = pd.to_datetime(options_df['date'])
    rf_df['date'] = pd.to_datetime(rf_df['date'])
    
    # Sort both dataframes by date
    options_df = options_df.sort_values('date')
    rf_df = rf_df.sort_values('date')
    
    # Merge using most recent rate before or on the option date
    merged_df = pd.merge_asof(
        options_df,
        rf_df[['date', 'risk_free_rate']],
        on='date',
        direction='backward'
    )
    
    # Verify no missing rates
    missing = merged_df['risk_free_rate'].isna().sum()
    if missing > 0:
        print(f"Warning: {missing} rows missing risk-free rate, using 2% default")
        merged_df['risk_free_rate'] = merged_df['risk_free_rate'].fillna(0.02)
    
    # Convert date back to string format
    merged_df['date'] = merged_df['date'].dt.strftime('%Y-%m-%d')
    
    return merged_df


def save_to_parquet(
    df: pd.DataFrame,
    filename: str,
    output_dir: str = OUTPUT_DIR
) -> None:
    """
    Save DataFrame to parquet file.
    
    Args:
        df: DataFrame to save
        filename: Output filename (without extension)
        output_dir: Output directory
    """
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Build full filepath
    filepath = output_path / f"{filename}.parquet"

    # Save to parquet with compression
    df.to_parquet(filepath, compression='snappy', index=False)
    
    # Print file size
    print(f"{filename}: {len(df)} rows")


def process_ticker(
    ticker: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> None:
    """
    Process all data for a single ticker and save to files.
    """
    print(f"\nProcessing {ticker}...")
    
    # Step 1: Load option chain data
    print(f"  Loading option chain data...")
    options_df = get_option_chain_data(
        ticker=ticker,
        start_date=start_date,
        end_date=end_date
    )
    
    if options_df.empty:
        print(f"No option data found for {ticker}")
        return
    
    # Step 2: Load volatility history
    print(f"  Loading volatility data...")
    vol_df = get_volatility_history(
        ticker=ticker,
        start_date=start_date,
        end_date=end_date
    )
    
    # Step 3: Download underlying price data
    print(f"  Downloading underlying price data...")
    price_df = get_underlying_price_data(
        ticker=ticker,
        start_date=start_date,
        end_date=end_date
    )
    
    if price_df.empty:
        print(f"No price data found for {ticker}")
        return
    
    # Step 4: Download risk-free rate data
    print(f"  Downloading risk-free rate data...")
    rf_df = get_risk_free_rate(start_date, end_date)
    
    # Step 5: Clean option chain data
    print(f"  Cleaning option data...")
    clean_options_df = clean_option_data(options_df)
    
    # Step 6: Merge underlying prices with options data
    print(f"  Merging underlying prices...")
    clean_options_df = merge_underlying_prices(
        clean_options_df,
        price_df,
        ticker
    )
    
    # Step 7: Merge risk-free rate with options data
    print(f"  Merging risk-free rate...")
    clean_options_df = merge_risk_free_rate(
        clean_options_df,
        rf_df
    )
    
    # Step 8: Clean volatility data
    if not vol_df.empty:
        clean_vol_df = clean_volatility_data(vol_df)
    else:
        clean_vol_df = pd.DataFrame()
    
    # Step 9: Save to parquet files
    print(f"  Saving processed data...")
    options_filename = f"{ticker.lower()}_options"
    save_to_parquet(clean_options_df, options_filename)
    
    if not clean_vol_df.empty:
        vol_filename = f"{ticker.lower()}_volatility"
        save_to_parquet(clean_vol_df, vol_filename)
    
    price_filename = f"{ticker.lower()}_prices"
    save_to_parquet(price_df, price_filename)
    
    print(f"✓ {ticker} processing complete")

def main():
    """
    Main function to process all tickers.
    """
    print("="*50)
    print("Options Data Preprocessing Pipeline")
    print(f"Date range: {START} to {END}")
    print("="*50)
    
    # Process SPY (includes risk-free rate merge)
    process_ticker('SPY', START, END)
    
    print("\n" + "="*50)
    print(f"All data saved to: {OUTPUT_DIR}")
    print("="*50)


if __name__ == "__main__":
    main()