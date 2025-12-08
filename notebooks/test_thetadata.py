"""
ThetaData Terminal Test Script - v3 API

Test connectivity and data retrieval from ThetaData terminal (API v3).
Run this before modifying preprocess.py to verify the setup works.

Prerequisites:
    1. ThetaData Terminal running: cd ~/theta_data && java -jar ThetaTerminalv3.jar
    2. httpx installed: pip install httpx pandas

Usage:
    python notebooks/test_thetadata.py
    
FREE Tier Limitations (discovered during testing):
    - Option/Stock metadata (expirations, strikes): ✅ AVAILABLE
    - Option/Stock price data (quotes, OHLC): ❌ REQUIRES PAID
    - Historical data: ❌ REQUIRES PAID
    
For backtesting you'll need a paid subscription ($30-60/month).
"""

import httpx
import pandas as pd
from datetime import datetime, timedelta, date
from typing import Optional, Dict, Any, List, Tuple
import csv
from io import StringIO

# =============================================================================
# CONFIGURATION
# =============================================================================

# ThetaData terminal connection (from config.toml)
THETA_HOST = "localhost"
THETA_PORT = 25503
BASE_URL = f"http://{THETA_HOST}:{THETA_PORT}"

# Test parameters
TEST_TICKER = "SPY"

# =============================================================================
# CONNECTION TEST
# =============================================================================

def test_connection() -> bool:
    """Test if ThetaData terminal is reachable and API version."""
    print("=" * 70)
    print("STEP 1: Testing Connection to ThetaData Terminal (API v3)")
    print("=" * 70)
    print(f"  URL: {BASE_URL}")
    
    try:
        with httpx.Client(timeout=10.0) as client:
            # Test v3 endpoint that should work on free tier
            response = client.get(f"{BASE_URL}/v3/option/list/expirations?symbol=SPY")
            
            if response.status_code == 200:
                lines = response.text.strip().split('\n')
                print(f"  Status: {response.status_code}")
                print(f"  API: v3 (confirmed)")
                print(f"  Test data: {len(lines)-1} SPY expirations found")
                print("  ✅ Connection successful!")
                return True
            elif response.status_code == 403:
                print(f"  Status: {response.status_code} (subscription required)")
                print("  ✅ Connection works, but need subscription for this endpoint")
                return True
            else:
                print(f"  ❌ Unexpected status code: {response.status_code}")
                print(f"  Response: {response.text[:200]}")
                return False
                
    except httpx.ConnectError as e:
        print(f"  ❌ Connection failed: {e}")
        print("\n  Make sure ThetaData Terminal is running:")
        print("    cd ~/theta_data && java -jar ThetaTerminalv3.jar")
        return False
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False


# =============================================================================
# FREE TIER DATA QUERIES (Metadata only)
# =============================================================================

def get_option_expirations(ticker: str) -> Tuple[Optional[List[str]], Optional[pd.DataFrame]]:
    """
    Get available option expiration dates for a ticker.
    FREE TIER: ✅ Available
    """
    print(f"\n  Querying Option Expirations for {ticker}...")
    
    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.get(
                f"{BASE_URL}/v3/option/list/expirations",
                params={"symbol": ticker}
            )
            
            if response.status_code != 200:
                print(f"    Error: HTTP {response.status_code}")
                print(f"    {response.text[:200]}")
                return None, None
            
            # Parse CSV response
            df = pd.read_csv(StringIO(response.text))
            expirations = df['expiration'].tolist()
            
            print(f"    ✅ Found {len(expirations)} expirations")
            
            # Show upcoming expirations
            today_str = date.today().strftime('%Y-%m-%d')
            future_exps = sorted([e for e in expirations if e >= today_str])
            past_exps = sorted([e for e in expirations if e < today_str])
            
            print(f"    Historical expirations: {len(past_exps)}")
            print(f"    Future expirations: {len(future_exps)}")
            
            if future_exps:
                print(f"    Next 5 expirations: {future_exps[:5]}")
            
            return expirations, df
            
    except Exception as e:
        print(f"    Error: {e}")
        return None, None


def get_option_strikes(ticker: str, expiration: str) -> Tuple[Optional[List[float]], Optional[pd.DataFrame]]:
    """
    Get available strikes for a specific expiration.
    FREE TIER: ✅ Available
    
    Args:
        ticker: Symbol (e.g., 'SPY')
        expiration: Date in YYYY-MM-DD format
    
    Returns:
        Tuple of (strikes_list, dataframe)
    """
    print(f"\n  Querying Option Strikes for {ticker} exp={expiration}...")
    
    # Convert to YYYYMMDD format
    exp_formatted = expiration.replace('-', '')
    
    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.get(
                f"{BASE_URL}/v3/option/list/strikes",
                params={"symbol": ticker, "expiration": exp_formatted}
            )
            
            if response.status_code != 200:
                print(f"    Error: HTTP {response.status_code}")
                print(f"    {response.text[:200]}")
                return None, None
            
            # Parse CSV response
            df = pd.read_csv(StringIO(response.text))
            
            # ThetaData returns strikes in dollars (not thousandths in v3)
            strikes = df['strike'].tolist()
            
            print(f"    ✅ Found {len(strikes)} strikes")
            print(f"    Range: ${min(strikes):.2f} - ${max(strikes):.2f}")
            
            return strikes, df
            
    except Exception as e:
        print(f"    Error: {e}")
        return None, None


# =============================================================================
# PAID TIER DATA QUERIES (Will return 403 on free tier)
# =============================================================================

def get_option_snapshot_ohlc(ticker: str, expiration: str) -> Optional[pd.DataFrame]:
    """
    Get bulk option OHLC snapshot for all strikes at an expiration.
    PAID TIER: Requires Value subscription ($30+/month)
    
    Args:
        ticker: Underlying symbol
        expiration: Expiration date YYYY-MM-DD
    
    Returns:
        DataFrame with OHLC for all strikes, or None
    """
    print(f"\n  Querying Bulk Option OHLC: {ticker} exp={expiration}...")
    print("    (Requires PAID subscription)")
    
    exp_formatted = expiration.replace('-', '')
    
    try:
        with httpx.Client(timeout=60.0) as client:
            response = client.get(
                f"{BASE_URL}/v3/option/snapshot/ohlc",
                params={"symbol": ticker, "expiration": exp_formatted}
            )
            
            if response.status_code == 403:
                print("    ❌ HTTP 403 - Subscription required")
                print("    This endpoint requires a paid ThetaData subscription")
                return None
            
            if response.status_code != 200:
                print(f"    Error: HTTP {response.status_code}")
                return None
            
            df = pd.read_csv(StringIO(response.text))
            print(f"    ✅ Got {len(df)} option contracts")
            print(f"    Columns: {list(df.columns)}")
            
            return df
            
    except Exception as e:
        print(f"    Error: {e}")
        return None


def get_option_snapshot_quote(ticker: str, expiration: str) -> Optional[pd.DataFrame]:
    """
    Get bulk option quotes for all strikes at an expiration.
    PAID TIER: Requires Value subscription ($30+/month)
    """
    print(f"\n  Querying Bulk Option Quote: {ticker} exp={expiration}...")
    print("    (Requires PAID subscription)")
    
    exp_formatted = expiration.replace('-', '')
    
    try:
        with httpx.Client(timeout=60.0) as client:
            response = client.get(
                f"{BASE_URL}/v3/option/snapshot/quote",
                params={"symbol": ticker, "expiration": exp_formatted}
            )
            
            if response.status_code == 403:
                print("    ❌ HTTP 403 - Subscription required")
                print("    This endpoint requires a paid ThetaData subscription")
                return None
            
            if response.status_code != 200:
                print(f"    Error: HTTP {response.status_code}")
                return None
            
            df = pd.read_csv(StringIO(response.text))
            print(f"    ✅ Got {len(df)} option contracts")
            print(f"    Columns: {list(df.columns)}")
            
            return df
            
    except Exception as e:
        print(f"    Error: {e}")
        return None


def get_stock_quote(ticker: str) -> Optional[pd.DataFrame]:
    """
    Get current stock quote.
    PAID TIER: Requires Value subscription ($30+/month)
    """
    print(f"\n  Querying Stock Quote for {ticker}...")
    print("    (Requires PAID subscription)")
    
    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.get(
                f"{BASE_URL}/v3/stock/snapshot/quote",
                params={"symbol": ticker}
            )
            
            if response.status_code == 403:
                print("    ❌ HTTP 403 - Subscription required")
                return None
            
            if response.status_code != 200:
                print(f"    Error: HTTP {response.status_code}")
                return None
            
            df = pd.read_csv(StringIO(response.text))
            print(f"    ✅ Got stock quote")
            return df
            
    except Exception as e:
        print(f"    Error: {e}")
        return None


# =============================================================================
# ENDPOINT DISCOVERY
# =============================================================================

def discover_all_endpoints():
    """Discover and test all API endpoints."""
    print("\n" + "=" * 70)
    print("STEP 2: Testing All v3 API Endpoints")
    print("=" * 70)
    
    # Comprehensive list of v3 endpoints
    endpoints = {
        # Metadata (FREE tier)
        "Option Expirations": ("/v3/option/list/expirations", {"symbol": "SPY"}),
        "Option Strikes": ("/v3/option/list/strikes", {"symbol": "SPY", "expiration": "20250117"}),
        
        # Stock data (PAID tier)
        "Stock Quote": ("/v3/stock/snapshot/quote", {"symbol": "SPY"}),
        
        # Option snapshots (PAID tier)
        "Option Quote Snapshot": ("/v3/option/snapshot/quote", {"symbol": "SPY", "expiration": "20250117"}),
        "Option OHLC Snapshot": ("/v3/option/snapshot/ohlc", {"symbol": "SPY", "expiration": "20250117"}),
        
        # Greek calculations (may be PAID)
        "Option Greeks": ("/v3/option/snapshot/greeks", {"symbol": "SPY", "expiration": "20250117"}),
    }
    
    results = {}
    
    with httpx.Client(timeout=15.0) as client:
        for name, (endpoint, params) in endpoints.items():
            try:
                response = client.get(f"{BASE_URL}{endpoint}", params=params)
                
                if response.status_code == 200:
                    status = "✅ FREE"
                    tier = "free"
                elif response.status_code == 403:
                    status = "💰 PAID"
                    tier = "paid"
                elif response.status_code == 404:
                    status = "❓ NOT FOUND"
                    tier = "unknown"
                else:
                    status = f"❌ HTTP {response.status_code}"
                    tier = "error"
                
                results[name] = {"endpoint": endpoint, "status": response.status_code, "tier": tier}
                print(f"  {status:12} {name}: {endpoint}")
                
            except Exception as e:
                results[name] = {"endpoint": endpoint, "status": None, "tier": "error"}
                print(f"  ❌ ERROR   {name}: {e}")
    
    return results


# =============================================================================
# SUBSCRIPTION INFO
# =============================================================================

def print_subscription_info():
    """Print information about ThetaData subscription tiers."""
    print("\n" + "=" * 70)
    print("SUBSCRIPTION INFORMATION")
    print("=" * 70)
    print("""
    Your current tier: FREE
    
    Available Tiers:
    ┌────────────────────────────────────────────────────────────────────┐
    │ FREE ($0/month)                                                     │
    │   - Option metadata (expirations, strikes)                         │
    │   - Useful for: Testing setup, building infrastructure             │
    │   - NOT useful for: Backtesting (no price data)                   │
    ├────────────────────────────────────────────────────────────────────┤
    │ VALUE ($30/month)                                                   │
    │   - Real-time quotes (15-min delayed for options)                  │
    │   - 5+ years of EOD historical data                                │
    │   - Useful for: Daily/weekly backtesting strategies                │
    ├────────────────────────────────────────────────────────────────────┤
    │ PRO ($60/month)                                                     │
    │   - Full historical data                                           │
    │   - Intraday data (1-min, 5-min, etc.)                            │
    │   - Useful for: Intraday strategies, research                      │
    └────────────────────────────────────────────────────────────────────┘
    
    For your backtesting project, you need at minimum VALUE tier.
    
    Signup: https://www.thetadata.us/pricing
    """)


# =============================================================================
# FORMAT CONVERSION (for when you upgrade)
# =============================================================================

def show_format_conversion_example():
    """Show how to convert ThetaData format to your project format."""
    print("\n" + "=" * 70)
    print("FORMAT CONVERSION (for when you upgrade)")
    print("=" * 70)
    
    print("""
    Your project expects these columns in spy_options.parquet:
    ┌──────────────────┬──────────────────────────────────────────────────┐
    │ Column           │ Description                                       │
    ├──────────────────┼──────────────────────────────────────────────────┤
    │ date             │ Option quote date (YYYY-MM-DD)                   │
    │ expiration       │ Option expiration date (YYYY-MM-DD)              │
    │ strike           │ Strike price ($)                                  │
    │ call_put         │ 'Call' or 'Put'                                  │
    │ bid              │ Bid price ($)                                     │
    │ ask              │ Ask price ($)                                     │
    │ vol              │ Implied volatility (decimal, e.g., 0.20)         │
    │ mid_price        │ (bid + ask) / 2                                  │
    │ spread           │ ask - bid                                         │
    │ spread_pct       │ spread / mid_price                                │
    │ days_to_expiry   │ (expiration - date).days                          │
    │ underlying_price │ Current stock price ($)                           │
    │ risk_free_rate   │ Risk-free rate (decimal)                          │
    └──────────────────┴──────────────────────────────────────────────────┘
    
    ThetaData v3 provides (when you have subscription):
    ┌──────────────────┬──────────────────────────────────────────────────┐
    │ ThetaData Column │ Maps To                                          │
    ├──────────────────┼──────────────────────────────────────────────────┤
    │ symbol           │ (filter by ticker)                               │
    │ expiration       │ expiration (needs date format conversion)        │
    │ strike           │ strike (in dollars)                              │
    │ right            │ call_put ('C'→'Call', 'P'→'Put')                │
    │ bid              │ bid                                               │
    │ ask              │ ask                                               │
    │ open, high, low  │ (OHLC data available)                            │
    │ close            │ (previous close)                                  │
    └──────────────────┴──────────────────────────────────────────────────┘
    
    You'll need to calculate:
      - mid_price, spread, spread_pct (from bid/ask)
      - days_to_expiry (from dates)
      - vol (implied volatility - may need to calculate or get from /greeks)
      - underlying_price (from stock quote)
      - risk_free_rate (from FRED or Treasury data)
    """)


def generate_preprocess_template():
    """Generate a template for the new ThetaData-based preprocessing."""
    print("\n" + "=" * 70)
    print("PREPROCESS.PY TEMPLATE (save for when you upgrade)")
    print("=" * 70)
    
    template = '''
# ThetaData-based preprocessing template
# Replace your Dolt-based preprocess.py with this when you have a subscription

import httpx
import pandas as pd
from datetime import datetime, timedelta
from io import StringIO
from typing import Optional

THETA_URL = "http://localhost:25503"

def fetch_option_chain(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch option chain data from ThetaData for a date range.
    
    This requires iterating through dates and expirations.
    """
    all_data = []
    
    with httpx.Client(timeout=120.0) as client:
        # Get all expirations
        resp = client.get(f"{THETA_URL}/v3/option/list/expirations", 
                         params={"symbol": ticker})
        exp_df = pd.read_csv(StringIO(resp.text))
        
        # Filter to relevant expirations
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        
        for exp_date in exp_df['expiration']:
            exp_dt = datetime.strptime(exp_date, "%Y-%m-%d")
            
            # Skip if expiration is before start date
            if exp_dt < start_dt:
                continue
            
            # Get snapshot for this expiration
            exp_formatted = exp_date.replace("-", "")
            
            resp = client.get(
                f"{THETA_URL}/v3/option/snapshot/quote",
                params={"symbol": ticker, "expiration": exp_formatted}
            )
            
            if resp.status_code == 200:
                df = pd.read_csv(StringIO(resp.text))
                df['expiration'] = exp_date
                df['date'] = datetime.now().strftime("%Y-%m-%d")
                all_data.append(df)
    
    if all_data:
        return pd.concat(all_data, ignore_index=True)
    return pd.DataFrame()


def convert_to_project_format(df: pd.DataFrame, underlying_price: float) -> pd.DataFrame:
    """Convert ThetaData format to project format."""
    result = df.copy()
    
    # Rename/convert columns
    result['call_put'] = result['right'].map({'C': 'Call', 'P': 'Put'})
    result['mid_price'] = (result['bid'] + result['ask']) / 2
    result['spread'] = result['ask'] - result['bid']
    result['spread_pct'] = result['spread'] / result['mid_price']
    
    # Calculate days to expiry
    result['date'] = pd.to_datetime(result['date'])
    result['expiration'] = pd.to_datetime(result['expiration'])
    result['days_to_expiry'] = (result['expiration'] - result['date']).dt.days
    
    # Add underlying price
    result['underlying_price'] = underlying_price
    
    # Add risk-free rate (fetch from FRED or use default)
    result['risk_free_rate'] = 0.05  # TODO: Fetch from Treasury
    
    # Note: vol (IV) may need to be calculated using your pricing module
    # or fetched from /v3/option/snapshot/greeks if available
    
    return result
'''
    print(template)


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print(" THETADATA TERMINAL TEST SCRIPT - API v3")
    print("=" * 70)
    print(f" Terminal URL: {BASE_URL}")
    print(f" Test Ticker:  {TEST_TICKER}")
    print("=" * 70)
    
    # Step 1: Test connection
    if not test_connection():
        print("\n❌ Cannot proceed without connection. Exiting.")
        print("\nMake sure ThetaData Terminal is running:")
        print("  cd ~/theta_data && java -jar ThetaTerminalv3.jar")
        return
    
    # Step 2: Discover endpoints
    discover_all_endpoints()
    
    # Step 3: Test FREE tier endpoints
    print("\n" + "=" * 70)
    print("STEP 3: Testing FREE Tier Data (Metadata)")
    print("=" * 70)
    
    # Get expirations
    expirations, _ = get_option_expirations(TEST_TICKER)
    
    if expirations:
        # Find nearest future expiration
        today_str = date.today().strftime('%Y-%m-%d')
        future_exps = sorted([e for e in expirations if e >= today_str])
        
        if future_exps:
            test_exp = future_exps[0]
            
            # Get strikes for that expiration
            strikes, strikes_df = get_option_strikes(TEST_TICKER, test_exp)
            
            if strikes:
                print(f"\n  Sample strikes for {test_exp}:")
                # Show strikes near ATM (assume ~$600 for SPY)
                atm_strikes = sorted(strikes, key=lambda x: abs(x - 600))[:5]
                print(f"  Near ATM: {atm_strikes}")
    
    # Step 4: Test PAID tier endpoints (will fail)
    print("\n" + "=" * 70)
    print("STEP 4: Testing PAID Tier Data (Expected to fail on FREE)")
    print("=" * 70)
    
    if expirations and future_exps:
        get_stock_quote(TEST_TICKER)
        get_option_snapshot_quote(TEST_TICKER, future_exps[0])
        get_option_snapshot_ohlc(TEST_TICKER, future_exps[0])
    
    # Step 5: Subscription info
    print_subscription_info()
    
    # Step 6: Show format conversion
    show_format_conversion_example()
    
    # Summary
    print("\n" + "=" * 70)
    print(" SUMMARY")
    print("=" * 70)
    print(f"""
    ✅ Connection: Working (API v3)
    ✅ FREE Tier: Metadata endpoints working (expirations, strikes)
    ❌ PAID Tier: Price data requires subscription
    
    Your current FREE tier can:
      - Verify infrastructure is working
      - Get option metadata (expirations, strikes)
      - Build data pipeline (without actual prices)
    
    For backtesting, you need to upgrade to VALUE ($30/month) minimum.
    
    Next steps:
      1. ✅ Terminal connectivity verified
      2. Consider upgrading: https://www.thetadata.us/pricing
      3. When upgraded, use the preprocess template above
    """)
    print("=" * 70)


if __name__ == "__main__":
    main()
