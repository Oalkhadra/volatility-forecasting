"""
ThetaData Options Preprocessing

Downloads historical EOD options data from ThetaData with:
- All expirations and strikes in one API call
- Underlying prices from yfinance
- Risk-free rates from Treasury

Uses /v3/option/history/eod endpoint for historical data (Value tier).
API Reference: https://docs.thetadata.us/operations/option_history_eod.html

Note: IV is not included (requires Standard tier). Use separate notebook to calculate IV.
"""

import httpx
import pandas as pd
import numpy as np
import yfinance as yf
from io import StringIO
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Add parent for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from pricing.risk_free_rate import get_risk_free_rate

# =============================================================================
# CONFIGURATION
# =============================================================================

# Date range for options data
START = '2023-12-05'
END = '2024-12-04'

# ThetaData terminal connection
THETA_HOST = "localhost"
THETA_PORT = 25503
BASE_URL = f"http://{THETA_HOST}:{THETA_PORT}"

# Output directory
OUTPUT_DIR = "data/processed"

# Ticker to process
TICKER = "SPY"


# =============================================================================
# THETADATA FUNCTIONS
# =============================================================================

def fetch_options_eod(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch historical EOD options data from ThetaData.
    
    Args:
        symbol: Underlying symbol (e.g., "SPY")
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
    
    Returns:
        DataFrame with all options data
        
    API Reference: https://docs.thetadata.us/operations/option_history_eod.html
    """
    print(f"  Fetching {symbol} options EOD: {start_date} to {end_date}...")
    
    # Generate list of trading days
    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.strptime(end_date, '%Y-%m-%d')
    
    all_data = []
    current_dt = start_dt
    
    with httpx.Client(timeout=300.0) as client:
        while current_dt <= end_dt:
            date_str = current_dt.strftime('%Y%m%d')
            date_display = current_dt.strftime('%Y-%m-%d')
            
            # Skip weekends
            if current_dt.weekday() < 5:  # Monday = 0, Friday = 4
                print(f"    {date_display}...", end=" ", flush=True)
                
                response = client.get(
                    f"{BASE_URL}/v3/option/history/eod",
                    params={
                        "symbol": symbol,
                        "expiration": "*",  # All expirations
                        "strike": "*",      # All strikes
                        "right": "both",    # Calls and puts
                        "start_date": date_str,
                        "end_date": date_str,  # Same day (required for expiration=*)
                        "format": "csv"
                    }
                )
                
                if response.status_code == 200:
                    df = pd.read_csv(StringIO(response.text))
                    if not df.empty:
                        all_data.append(df)
                        print(f"✓ {len(df):,} records")
                    else:
                        print("no data")
                else:
                    print(f"error {response.status_code}")
            
            current_dt += timedelta(days=1)
    
    if not all_data:
        print("  No data found!")
        return pd.DataFrame()
    
    combined = pd.concat(all_data, ignore_index=True)
    print(f"  ✓ Total: {len(combined):,} raw option records")
    return combined


def process_raw_options(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    Transform raw ThetaData EOD response into clean format.
    
    ThetaData EOD columns include:
        symbol, expiration, strike, right, timestamp,
        open, high, low, close, volume, count, bid, ask
    """
    if df.empty:
        return df
    
    print("  Processing raw data...")
    
    # Extract date from 'timestamp' column (format: YYYY-MM-DDTHH:mm:ss.SSS)
    date_col = 'timestamp' if 'timestamp' in df.columns else 'created'
    df['date'] = pd.to_datetime(df[date_col]).dt.strftime('%Y-%m-%d')
    
    # Standardize option type
    df['call_put'] = df['right'].str.lower()
    
    # Clean expiration format
    df['expiration'] = pd.to_datetime(df['expiration']).dt.strftime('%Y-%m-%d')
    
    # Calculate days to expiry
    df['days_to_expiry'] = (
        pd.to_datetime(df['expiration']) - pd.to_datetime(df['date'])
    ).dt.days
    
    # Add symbol column
    df['act_symbol'] = symbol
    
    print(f"  ✓ Processed {len(df):,} records")
    print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"  Expirations: {df['expiration'].nunique()}")
    
    return df


# =============================================================================
# YFINANCE FUNCTIONS
# =============================================================================

def get_underlying_prices(ticker: str, start_date: str, end_date: str, full_history: bool = False) -> pd.DataFrame:
    """Download underlying price data from Yahoo Finance.
    
    Args:
        ticker: Stock ticker symbol
        start_date: Start date for price data
        end_date: End date for price data  
        full_history: If True, download all available history (for ML pre-training)
    """
    print(f"  Downloading {ticker} prices...")
    
    # Add buffer days for yfinance
    if full_history:
        # Get all available history for ML model pre-training
        start_dt = datetime(2010, 1, 1)  # Go back to 2010 for full history
        print(f"    Loading FULL history from {start_dt.strftime('%Y-%m-%d')}")
    else:
        start_dt = datetime.strptime(start_date, '%Y-%m-%d') - timedelta(days=5)
    end_dt = datetime.strptime(end_date, '%Y-%m-%d') + timedelta(days=5)
    
    df = yf.download(ticker, start=start_dt, end=end_dt, progress=False, auto_adjust=False)
    
    if df.empty:
        print(f"    Warning: No price data found")
        return pd.DataFrame()
    
    df = df.reset_index()
    
    # Flatten multi-level columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] if col[0] else col[1] for col in df.columns.values]
    
    df.columns = df.columns.str.lower()
    
    if 'adj close' in df.columns:
        df = df.drop(columns=['adj close'])
    
    df['ticker'] = ticker.upper()
    df['date'] = df['date'].dt.strftime('%Y-%m-%d')
    
    print(f"    ✓ Got {len(df)} days of price data")
    return df


def get_vix_data(start_date: str, end_date: str) -> pd.DataFrame:
    """Download VIX data from Yahoo Finance."""
    print(f"  Downloading VIX data...")
    
    start_dt = datetime.strptime(start_date, '%Y-%m-%d') - timedelta(days=5)
    end_dt = datetime.strptime(end_date, '%Y-%m-%d') + timedelta(days=5)
    
    df = yf.download("^VIX", start=start_dt, end=end_dt, progress=False, auto_adjust=False)
    
    if df.empty:
        return pd.DataFrame()
    
    df = df.reset_index()
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] if col[0] else col[1] for col in df.columns.values]
    
    df.columns = df.columns.str.lower()
    
    if 'close' in df.columns:
        df = df[['date', 'close']].rename(columns={'close': 'vix_close'})
        
    df['date'] = df['date'].dt.strftime('%Y-%m-%d')
    
    print(f"    ✓ Got {len(df)} days of VIX data")
    return df


# =============================================================================
# DATA ENRICHMENT
# =============================================================================

def merge_underlying_prices(options_df: pd.DataFrame, prices_df: pd.DataFrame) -> pd.DataFrame:
    """Merge underlying close price by date from yfinance."""
    print("  Merging underlying prices from yfinance...")
    
    if prices_df.empty:
        options_df['underlying_price'] = np.nan
        options_df['underlying_volume'] = 0
        return options_df
    
    price_map = prices_df.set_index('date')['close'].to_dict()
    volume_map = prices_df.set_index('date')['volume'].to_dict()
    
    options_df['underlying_price'] = options_df['date'].map(price_map)
    options_df['underlying_volume'] = options_df['date'].map(volume_map).fillna(0).astype(int)
    
    # Drop rows without underlying price
    before = len(options_df)
    options_df = options_df.dropna(subset=['underlying_price'])
    dropped = before - len(options_df)
    if dropped > 0:
        print(f"    Dropped {dropped:,} rows missing underlying price")
    
    print(f"    ✓ Merged {len(options_df):,} records")
    return options_df


def merge_risk_free_rate(options_df: pd.DataFrame, rf_df: pd.DataFrame) -> pd.DataFrame:
    """Merge risk-free rate by date."""
    print("  Merging risk-free rates...")
    
    if rf_df.empty:
        options_df['risk_free_rate'] = 0.05
        print("    Using default 5% rate")
        return options_df
    
    rf_map = rf_df.set_index('date')['risk_free_rate'].to_dict()
    options_df['risk_free_rate'] = options_df['date'].map(rf_map).fillna(0.05)
    
    return options_df


# =============================================================================
# DATA FILTERING
# =============================================================================

def apply_filters(df: pd.DataFrame, min_dte: int = 1, max_dte: int = 120) -> pd.DataFrame:
    """
    Apply liquidity and quality filters.
    
    Filters:
        - DTE between min_dte and max_dte
        - Valid bid/ask (bid > 0, ask > 0, bid <= ask)
        - Spread < 10%
        - Mid price > $0.10
        - No arbitrage violations (price > intrinsic)
    """
    print("  Applying filters...")
    initial_count = len(df)
    
    # DTE filter
    df = df[(df['days_to_expiry'] >= min_dte) & (df['days_to_expiry'] <= max_dte)].copy()
    
    # Check if bid/ask are available
    has_quotes = 'bid' in df.columns and 'ask' in df.columns
    
    if has_quotes and df['bid'].notna().any():
        # Calculate mid price and spread
        df['mid_price'] = (df['bid'] + df['ask']) / 2
        df['spread'] = df['ask'] - df['bid']
        df['spread_pct'] = df['spread'] / df['mid_price'].replace(0, np.nan)
        
        # Quote quality filters
        df = df[
            (df['bid'] > 0) &
            (df['ask'] > 0) &
            (df['bid'] <= df['ask']) &
            (df['spread_pct'] < 0.10) &
            (df['mid_price'] > 0.10)
        ].copy()
    else:
        # No quotes available, use close price
        print("    Note: Using close price (no bid/ask available)")
        df['mid_price'] = df['close']
        df['spread'] = np.nan
        df['spread_pct'] = np.nan
        df = df[df['mid_price'] > 0.10].copy()
    
    # Arbitrage filter: price must exceed intrinsic value
    intrinsic = np.where(
        df['call_put'] == 'call',
        np.maximum(df['underlying_price'] - df['strike'], 0),
        np.maximum(df['strike'] - df['underlying_price'], 0)
    )
    
    violations = (df['mid_price'] <= intrinsic).sum()
    if violations > 0:
        print(f"    Arbitrage violations: {violations:,}")
    
    df = df[df['mid_price'] > intrinsic].copy()
    
    removed = initial_count - len(df)
    print(f"    Removed {removed:,} ({100*removed/initial_count:.1f}%)")
    print(f"    Remaining: {len(df):,} options")
    
    return df


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def download_and_process(
    symbol: str = "SPY",
    start_date: str = START,
    end_date: str = END,
    output_dir: str = OUTPUT_DIR
) -> pd.DataFrame:
    """
    Full pipeline: download, enrich, filter, and save options data.
    
    Args:
        symbol: Underlying ticker
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        output_dir: Where to save parquet files
    
    Returns:
        Processed DataFrame
    """
    print(f"\n{'='*60}")
    print(f"ThetaData Options Preprocessing")
    print(f"{'='*60}")
    print(f"Symbol: {symbol}")
    print(f"Date range: {start_date} to {end_date}")
    
    # 1. Fetch historical EOD options data
    print(f"\n1. Fetching options data...")
    raw_df = fetch_options_eod(symbol, start_date, end_date)
    
    if raw_df.empty:
        print("No options data found!")
        return pd.DataFrame()
    
    # 2. Process raw data
    print(f"\n2. Processing raw data...")
    df = process_raw_options(raw_df, symbol)
    
    # 3. Get underlying prices (full history for ML pre-training)
    print(f"\n3. Fetching underlying prices...")
    prices_df = get_underlying_prices(symbol, start_date, end_date, full_history=True)
    
    # 4. Get risk-free rate
    print(f"\n4. Fetching risk-free rate...")
    rf_df = get_risk_free_rate(start_date, end_date)
    
    # 5. Merge underlying prices
    print(f"\n5. Enriching data...")
    df = merge_underlying_prices(df, prices_df)
    df = merge_risk_free_rate(df, rf_df)
    
    # 6. Apply filters
    print(f"\n6. Filtering data...")
    df = apply_filters(df, min_dte=1, max_dte=30)
    
    # 7. Select final columns
    print(f"\n7. Finalizing schema...")
    columns = [
        'date', 'act_symbol', 'expiration', 'strike', 'call_put',
        'bid', 'ask', 'mid_price', 'spread', 'spread_pct',
        'days_to_expiry', 'underlying_price', 'underlying_volume', 'risk_free_rate',
        'open', 'high', 'low', 'close', 'volume'
    ]
    df = df[[c for c in columns if c in df.columns]]
    df = df.sort_values(['date', 'expiration', 'strike', 'call_put']).reset_index(drop=True)
    
    print(f"  Final columns: {list(df.columns)}")
    
    # 8. Save
    print(f"\n8. Saving files...")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save options
    options_file = output_path / f"{symbol.lower()}_options.parquet"
    df.to_parquet(options_file, compression='snappy', index=False)
    print(f"  ✓ Options: {options_file} ({len(df):,} rows)")
    
    # Save prices
    if not prices_df.empty:
        prices_file = output_path / f"{symbol.lower()}_prices.parquet"
        prices_df.to_parquet(prices_file, compression='snappy', index=False)
        print(f"  ✓ Prices: {prices_file}")
    
    # Save risk-free rate
    if not rf_df.empty:
        rf_file = output_path / "risk_free_rate.parquet"
        rf_df.to_parquet(rf_file, compression='snappy', index=False)
        print(f"  ✓ Risk-free rate: {rf_file}")
    
    # Summary
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"  Total options: {len(df):,}")
    print(f"  Trading days: {df['date'].nunique()}")
    print(f"  Expirations: {df['expiration'].nunique()}")
    print(f"  Strikes: {df['strike'].nunique()}")
    print(f"  Calls: {(df['call_put'] == 'call').sum():,}")
    print(f"  Puts: {(df['call_put'] == 'put').sum():,}")
    
    return df


if __name__ == "__main__":
    # Process options
    df = download_and_process(TICKER, START, END, OUTPUT_DIR)
    
    # Also download VIX
    print(f"\nDownloading VIX data...")
    vix_df = get_vix_data(START, END)
    if not vix_df.empty:
        vix_file = Path(OUTPUT_DIR) / "vix_data.parquet"
        vix_df.to_parquet(vix_file, compression='snappy', index=False)
        print(f"✓ VIX saved: {vix_file}")
    
    print("\n✓ Done!")
