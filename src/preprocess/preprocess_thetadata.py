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
from dateutil.relativedelta import relativedelta
from calendar import monthrange

# Add parent for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from pricing.risk_free_rate import get_risk_free_rate

# =============================================================================
# CONFIGURATION
# =============================================================================

# Date range for options data
START = '2020-01-01'
END = '2025-11-30'  # Adjust to your desired end date

# ThetaData terminal connection
THETA_HOST = "localhost"
THETA_PORT = 25503
BASE_URL = f"http://{THETA_HOST}:{THETA_PORT}"

# Output directory
OUTPUT_RAW = "data/raw"
OUTPUT_DIR = "data/processed"

# Ticker to process
TICKER = "SPY"


# =============================================================================
# DATE UTILITIES
# =============================================================================

def generate_month_ranges(start_date: str, end_date: str):
    """
    Generate list of (year, month, start_date, end_date) tuples for each month.
    
    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
    
    Yields:
        Tuple of (year, month, month_start, month_end) for each month
    """
    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.strptime(end_date, '%Y-%m-%d')
    
    current_dt = start_dt.replace(day=1)  # Start at beginning of month
    
    while current_dt <= end_dt:
        year = current_dt.year
        month = current_dt.month
        
        # First day of month (or start_date if in first month)
        if current_dt.year == start_dt.year and current_dt.month == start_dt.month:
            month_start = start_dt
        else:
            month_start = current_dt
        
        # Last day of month (or end_date if in last month)
        last_day = monthrange(year, month)[1]
        month_end = current_dt.replace(day=last_day)
        if month_end > end_dt:
            month_end = end_dt
        
        yield (year, month, month_start.strftime('%Y-%m-%d'), month_end.strftime('%Y-%m-%d'))
        
        # Move to next month
        current_dt += relativedelta(months=1)


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
    
    # Extract date from 'timestamp' column (format: YYYY-MM-DDTHH:mm:ss.SSS or YYYY-MM-DDTHH:mm:ss)
    date_col = 'timestamp' if 'timestamp' in df.columns else 'created'
    df['date'] = pd.to_datetime(df[date_col], format='ISO8601').dt.strftime('%Y-%m-%d')
    
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
        start_dt = datetime(1993, 1, 29)  # Go back to 1993 for full history
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
    df = yf.download("^VIX", period="max", auto_adjust=False)
    
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

def apply_filters(df: pd.DataFrame, min_dte: int = 1, max_dte: int = 30) -> pd.DataFrame:
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

def process_month(
    symbol: str,
    year: int,
    month: int,
    month_start: str,
    month_end: str,
    prices_df: pd.DataFrame,
    rf_df: pd.DataFrame,
    output_dir: str = OUTPUT_DIR,
    skip_existing: bool = True
) -> tuple[pd.DataFrame, int, int]:
    """
    Process one month of options data.
    
    Args:
        symbol: Underlying ticker
        year: Year
        month: Month (1-12)
        month_start: Start date for this month (YYYY-MM-DD)
        month_end: End date for this month (YYYY-MM-DD)
        prices_df: Pre-loaded underlying prices
        rf_df: Pre-loaded risk-free rates
        output_dir: Where to save parquet files
        skip_existing: If True, skip months that already have processed files
    
    Returns:
        Tuple of (processed_df, raw_count, processed_count)
    """
    month_str = f"{year}_{month:02d}"
    
    # Check if already exists
    output_path = Path(output_dir)
    processed_file = output_path / f"{symbol.lower()}_options_{month_str}.parquet"
    
    if skip_existing and processed_file.exists():
        print(f"  ⏭️  Skipping {month_str} (already exists)")
        # Load and return counts
        existing_df = pd.read_parquet(processed_file)
        return existing_df, len(existing_df), len(existing_df)
    
    print(f"\n{'='*60}")
    print(f"Processing {year}-{month:02d}")
    print(f"{'='*60}")
    
    # 1. Fetch historical EOD options data
    print(f"1. Fetching options data for {month_start} to {month_end}...")
    raw_df = fetch_options_eod(symbol, month_start, month_end)
    
    if raw_df.empty:
        print("  No options data found for this month!")
        return pd.DataFrame(), 0, 0
    
    # Store raw data
    raw_path = Path(OUTPUT_RAW)
    raw_path.mkdir(parents=True, exist_ok=True)
    raw_file = raw_path / f"{symbol.lower()}_options_raw_{month_str}.parquet"
    raw_df.to_parquet(raw_file, compression='snappy', index=False)
    print(f"  ✓ Raw saved: {raw_file} ({len(raw_df):,} rows)")
    
    # 2. Process raw data
    print(f"\n2. Processing raw data...")
    df = process_raw_options(raw_df, symbol)
    
    # 3. Merge underlying prices and risk-free rate
    print(f"\n3. Enriching data...")
    df = merge_underlying_prices(df, prices_df)
    df = merge_risk_free_rate(df, rf_df)
    
    # 4. Apply filters
    print(f"\n4. Filtering data...")
    df = apply_filters(df, min_dte=1, max_dte=120)
    
    # 5. Select final columns
    print(f"\n5. Finalizing schema...")
    columns = [
        'date', 'act_symbol', 'expiration', 'strike', 'call_put',
        'bid', 'ask', 'mid_price', 'spread', 'spread_pct',
        'days_to_expiry', 'underlying_price', 'underlying_volume', 'risk_free_rate',
        'open', 'high', 'low', 'close', 'volume'
    ]
    df = df[[c for c in columns if c in df.columns]]
    df = df.sort_values(['date', 'expiration', 'strike', 'call_put']).reset_index(drop=True)
    
    # 6. Save processed data
    print(f"\n6. Saving processed data...")
    output_path.mkdir(parents=True, exist_ok=True)
    df.to_parquet(processed_file, compression='snappy', index=False)
    print(f"  ✓ Processed saved: {processed_file} ({len(df):,} rows)")
    
    # Summary for this month
    print(f"\n  Month Summary:")
    print(f"    Raw records: {len(raw_df):,}")
    print(f"    After filtering: {len(df):,}")
    print(f"    Trading days: {df['date'].nunique()}")
    print(f"    Expirations: {df['expiration'].nunique()}")
    print(f"    Calls: {(df['call_put'] == 'call').sum():,}")
    print(f"    Puts: {(df['call_put'] == 'put').sum():,}")
    
    return df, len(raw_df), len(df)


def download_and_process(
    symbol: str = "SPY",
    start_date: str = START,
    end_date: str = END,
    output_dir: str = OUTPUT_DIR,
    skip_existing: bool = True
) -> dict:
    """
    Full pipeline: download, enrich, filter, and save options data MONTHLY.
    
    Args:
        symbol: Underlying ticker
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        output_dir: Where to save parquet files
        skip_existing: If True, skip months that already have processed files
    
    Returns:
        Dictionary with summary statistics
    """
    print(f"\n{'='*70}")
    print(f"ThetaData Options Preprocessing - MONTHLY PIPELINE")
    print(f"{'='*70}")
    print(f"Symbol: {symbol}")
    print(f"Date range: {start_date} to {end_date}")
    print(f"Skip existing: {skip_existing}")
    
    # 1. Get underlying prices ONCE (full history for ML)
    print(f"\n{'='*70}")
    print("Loading underlying prices and risk-free rates (once)...")
    print(f"{'='*70}")
    prices_df = get_underlying_prices(symbol, start_date, end_date, full_history=True)
    rf_df = get_risk_free_rate(start_date, end_date)
    
    # Save these once
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if not prices_df.empty:
        prices_file = output_path / f"{symbol.lower()}_prices.parquet"
        prices_df.to_parquet(prices_file, compression='snappy', index=False)
        print(f"✓ Prices saved: {prices_file}")
    
    if not rf_df.empty:
        rf_file = output_path / "risk_free_rate.parquet"
        rf_df.to_parquet(rf_file, compression='snappy', index=False)
        print(f"✓ Risk-free rate saved: {rf_file}")
    
    # 2. Process each month
    print(f"\n{'='*70}")
    print("Processing options data by month...")
    print(f"{'='*70}")
    
    months = list(generate_month_ranges(start_date, end_date))
    print(f"Total months to process: {len(months)}\n")
    
    total_raw = 0
    total_processed = 0
    successful_months = 0
    
    for i, (year, month, month_start, month_end) in enumerate(months, 1):
        print(f"\n[Month {i}/{len(months)}]")
        try:
            df, raw_count, proc_count = process_month(
                symbol, year, month, month_start, month_end,
                prices_df, rf_df, output_dir, skip_existing
            )
            total_raw += raw_count
            total_processed += proc_count
            if proc_count > 0:
                successful_months += 1
        except Exception as e:
            print(f"  ❌ Error processing {year}-{month:02d}: {e}")
            continue
    
    # 3. Final Summary
    print(f"\n{'='*70}")
    print("FINAL SUMMARY")
    print(f"{'='*70}")
    print(f"  Months processed: {successful_months}/{len(months)}")
    print(f"  Total raw records: {total_raw:,}")
    print(f"  Total filtered records: {total_processed:,}")
    print(f"  Filter rate: {100*(1-total_processed/total_raw) if total_raw > 0 else 0:.1f}%")
    print(f"\n  Files saved to:")
    print(f"    Raw: {Path(OUTPUT_RAW).absolute()}")
    print(f"    Processed: {Path(output_dir).absolute()}")
    
    return {
        'months_processed': successful_months,
        'total_months': len(months),
        'total_raw': total_raw,
        'total_processed': total_processed
    }


# =============================================================================
# DATA LOADING UTILITIES
# =============================================================================

def load_monthly_data(
    symbol: str,
    start_date: str,
    end_date: str,
    output_dir: str = OUTPUT_DIR,
    data_type: str = "processed"
) -> pd.DataFrame:
    """
    Load multiple months of data into a single DataFrame.
    
    Args:
        symbol: Underlying ticker
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        output_dir: Directory containing parquet files
        data_type: Either "processed" or "raw"
    
    Returns:
        Combined DataFrame from all monthly files
    """
    print(f"Loading {data_type} data for {symbol} from {start_date} to {end_date}...")
    
    months = list(generate_month_ranges(start_date, end_date))
    all_dfs = []
    
    path = Path(OUTPUT_RAW if data_type == "raw" else output_dir)
    prefix = f"{symbol.lower()}_options_{data_type}_" if data_type == "raw" else f"{symbol.lower()}_options_"
    
    for year, month, _, _ in months:
        month_str = f"{year}_{month:02d}"
        file_path = path / f"{prefix}{month_str}.parquet"
        
        if file_path.exists():
            df = pd.read_parquet(file_path)
            all_dfs.append(df)
            print(f"  ✓ Loaded {month_str}: {len(df):,} rows")
        else:
            print(f"  ⚠️  Missing {month_str}")
    
    if not all_dfs:
        print("  No data files found!")
        return pd.DataFrame()
    
    combined = pd.concat(all_dfs, ignore_index=True)
    print(f"  Total: {len(combined):,} rows")
    
    return combined


if __name__ == "__main__":
    # Process options monthly
    # summary = download_and_process(
    #     symbol=TICKER,
    #     start_date=START,
    #     end_date=END,
    #     output_dir=OUTPUT_DIR,
    #     skip_existing=True  # Set to False to re-download existing months
    # )
    
    # Also download VIX
    print(f"\n{'='*70}")
    print("Downloading VIX data...")
    print(f"{'='*70}")
    vix_df = get_vix_data(START, END)
    if not vix_df.empty:
        vix_file = Path(OUTPUT_DIR) / "vix_data.parquet"
        vix_df.to_parquet(vix_file, compression='snappy', index=False)
        print(f"✓ VIX saved: {vix_file}")
    
    print(f"\n{'='*70}")
    print("✓ ALL DONE!")
    print(f"{'='*70}")
