"""
Volatility Preprocessing

Calculates implied volatility (IV) for all options in the processed dataset
and adds a 'vol' column. Uses fully vectorized NumPy operations for speed.

Input: data/processed/spy_options.parquet (from preprocess_thetadata.py)
Output: data/processed/spy_options.parquet (enriched with 'vol' column)

Performance: ~132k options in ~1-2 seconds (vs minutes with loop-based approach)
"""

import pandas as pd
import numpy as np
from scipy.stats import norm
from pathlib import Path
import time
import sys
import argparse

# Add parent for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "processed"
INPUT_FILE = DATA_DIR / "spy_options.parquet"
OUTPUT_FILE = DATA_DIR / "spy_options.parquet"  # Overwrites with enriched data

# IV solver parameters
MAX_ITERATIONS = 50
TOLERANCE = 1e-6
MIN_VOL = 0.001
MAX_VOL = 5.0


# =============================================================================
# VECTORIZED BLACK-SCHOLES
# =============================================================================

def bs_price_vec(S: np.ndarray, K: np.ndarray, T: np.ndarray, r: np.ndarray, 
                 sigma: np.ndarray, is_call: np.ndarray) -> np.ndarray:
    """
    Vectorized Black-Scholes price calculation.
    
    All inputs are numpy arrays of the same shape.
    """
    # Avoid division by zero
    sqrt_T = np.sqrt(np.maximum(T, 1e-10))
    sigma_safe = np.maximum(sigma, 1e-10)
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma_safe**2) * T) / (sigma_safe * sqrt_T)
    d2 = d1 - sigma_safe * sqrt_T
    
    # Vectorized call/put calculation
    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    
    return np.where(is_call, call_price, put_price)


def bs_vega_vec(S: np.ndarray, K: np.ndarray, T: np.ndarray, r: np.ndarray, 
                sigma: np.ndarray) -> np.ndarray:
    """
    Vectorized vega calculation (same for calls and puts).
    
    Returns vega scaled for Newton-Raphson (per 1.0 vol change, not per 1%).
    """
    sqrt_T = np.sqrt(np.maximum(T, 1e-10))
    sigma_safe = np.maximum(sigma, 1e-10)
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma_safe**2) * T) / (sigma_safe * sqrt_T)
    
    return S * sqrt_T * norm.pdf(d1)


def intrinsic_value_vec(S: np.ndarray, K: np.ndarray, is_call: np.ndarray) -> np.ndarray:
    """Vectorized intrinsic value calculation."""
    call_intrinsic = np.maximum(S - K, 0)
    put_intrinsic = np.maximum(K - S, 0)
    return np.where(is_call, call_intrinsic, put_intrinsic)


# =============================================================================
# VECTORIZED IV SOLVER
# =============================================================================

def initial_vol_guess(S: np.ndarray, K: np.ndarray, T: np.ndarray, 
                      price: np.ndarray, is_call: np.ndarray) -> np.ndarray:
    """
    Brenner-Subrahmanyam approximation for initial IV guess.
    
    Formula: σ ≈ √(2π/T) * (C/S) for ATM options
    Adjusted for moneyness.
    """
    # Simple approximation that works reasonably well
    # For ATM: sigma ≈ price / (0.4 * S * sqrt(T))
    sqrt_T = np.sqrt(np.maximum(T, 1e-10))
    
    # Brenner-Subrahmanyam for ATM
    atm_approx = np.sqrt(2 * np.pi / np.maximum(T, 0.01)) * (price / S)
    
    # Clamp to reasonable range
    return np.clip(atm_approx, 0.05, 2.0)


def implied_volatility_vec(
    price: np.ndarray,
    S: np.ndarray,
    K: np.ndarray,
    T: np.ndarray,
    r: np.ndarray,
    is_call: np.ndarray,
    max_iterations: int = MAX_ITERATIONS,
    tolerance: float = TOLERANCE
) -> np.ndarray:
    """
    Vectorized implied volatility calculation using Newton-Raphson.
    
    Calculates IV for all options simultaneously using array operations.
    This is 100-1000x faster than looping through options individually.
    
    Args:
        price: Market prices (array)
        S: Underlying prices (array)
        K: Strike prices (array)
        T: Time to expiry in years (array)
        r: Risk-free rates (array)
        is_call: Boolean array (True for calls)
        max_iterations: Max Newton-Raphson iterations
        tolerance: Convergence tolerance
    
    Returns:
        Array of implied volatilities (np.nan where calculation failed)
    """
    n = len(price)
    
    # Initialize with smart guess
    sigma = initial_vol_guess(S, K, T, price, is_call)
    
    # Track which options still need solving
    active = np.ones(n, dtype=bool)
    
    # Validate inputs - mark invalid as inactive
    intrinsic = intrinsic_value_vec(S, K, is_call)
    invalid = (T <= 0) | (price <= 0) | (S <= 0) | (K <= 0) | (price < intrinsic - 0.01)
    active[invalid] = False
    sigma[invalid] = np.nan
    
    # Newton-Raphson iterations (vectorized)
    for i in range(max_iterations):
        if not active.any():
            break
            
        # Calculate BS price and vega for active options
        bs_price = bs_price_vec(S[active], K[active], T[active], r[active], 
                                sigma[active], is_call[active])
        vega = bs_vega_vec(S[active], K[active], T[active], r[active], sigma[active])
        
        # Price difference
        diff = bs_price - price[active]
        
        # Check convergence
        converged = np.abs(diff) < tolerance
        
        # Update active mask - remove converged options
        active_indices = np.where(active)[0]
        active[active_indices[converged]] = False
        
        # Newton-Raphson update for non-converged options
        still_active = ~converged
        if still_active.any():
            # Avoid division by zero in vega
            vega_safe = np.where(np.abs(vega[still_active]) < 1e-10, 1e-10, vega[still_active])
            
            # Newton step with damping for stability
            delta = diff[still_active] / vega_safe
            delta = np.clip(delta, -0.5, 0.5)  # Damping
            
            # Update sigma
            new_sigma = sigma[active_indices[still_active]] - delta
            new_sigma = np.clip(new_sigma, MIN_VOL, MAX_VOL)
            sigma[active_indices[still_active]] = new_sigma
    
    # Mark non-converged as NaN
    sigma[active] = np.nan
    
    # Final validation - verify solution
    valid_mask = ~np.isnan(sigma)
    if valid_mask.any():
        final_price = bs_price_vec(S[valid_mask], K[valid_mask], T[valid_mask], 
                                   r[valid_mask], sigma[valid_mask], is_call[valid_mask])
        price_error = np.abs(final_price - price[valid_mask])
        
        # Mark large errors as failed
        bad_solutions = price_error > 0.10
        if bad_solutions.any():
            valid_indices = np.where(valid_mask)[0]
            sigma[valid_indices[bad_solutions]] = np.nan
    
    return sigma


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def enrich_with_iv(
    input_file: Path = INPUT_FILE,
    output_file: Path = OUTPUT_FILE
) -> pd.DataFrame:
    """
    Load options data, calculate IV, and save enriched dataset.
    
    Args:
        input_file: Path to input parquet file
        output_file: Path to output parquet file
    
    Returns:
        Enriched DataFrame with 'vol' column
    """
    print(f"\n{'='*60}")
    print("Volatility Preprocessing (Vectorized)")
    print(f"{'='*60}")
    
    # 1. Load data
    print(f"\n1. Loading options data...")
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    df = pd.read_parquet(input_file)
    print(f"   ✓ Loaded {len(df):,} options")
    print(f"   Date range: {df['date'].min()} to {df['date'].max()}")
    
    # 2. Prepare arrays for vectorized calculation
    print(f"\n2. Calculating implied volatility...")
    start_time = time.time()
    
    # Extract arrays
    price = df['mid_price'].values
    S = df['underlying_price'].values
    K = df['strike'].values
    T = df['days_to_expiry'].values / 365.0  # Convert to years
    r = df['risk_free_rate'].values
    is_call = (df['call_put'] == 'call').values
    
    # Calculate IV (vectorized - all at once!)
    iv = implied_volatility_vec(price, S, K, T, r, is_call)
    
    elapsed = time.time() - start_time
    print(f"   ✓ Calculated {len(df):,} IVs in {elapsed:.2f} seconds")
    print(f"   ✓ Speed: {len(df)/elapsed:,.0f} options/second")
    
    # Add to dataframe
    df['vol'] = iv
    
    # 3. Report statistics
    valid_iv = df['vol'].notna()
    n_valid = valid_iv.sum()
    n_failed = (~valid_iv).sum()
    
    print(f"\n3. IV Calculation Results:")
    print(f"   ✓ Successful: {n_valid:,} ({100*n_valid/len(df):.1f}%)")
    print(f"   ✗ Failed: {n_failed:,} ({100*n_failed/len(df):.1f}%)")
    
    if n_valid > 0:
        print(f"\n   IV Statistics (valid options):")
        print(f"   - Mean: {df.loc[valid_iv, 'vol'].mean():.4f} ({df.loc[valid_iv, 'vol'].mean()*100:.2f}%)")
        print(f"   - Median: {df.loc[valid_iv, 'vol'].median():.4f} ({df.loc[valid_iv, 'vol'].median()*100:.2f}%)")
        print(f"   - Std: {df.loc[valid_iv, 'vol'].std():.4f}")
        print(f"   - Min: {df.loc[valid_iv, 'vol'].min():.4f}")
        print(f"   - Max: {df.loc[valid_iv, 'vol'].max():.4f}")
    
    # 4. Save enriched data
    print(f"\n4. Saving enriched data...")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_file, compression='snappy', index=False)
    print(f"   ✓ Saved to: {output_file}")
    
    # Summary
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"  Total options: {len(df):,}")
    print(f"  Options with valid IV: {n_valid:,}")
    print(f"  Processing time: {elapsed:.2f}s")
    print(f"  Output file: {output_file}")
    
    return df


def load_options_with_iv(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """
    Convenience function to load the options data with IV.
    
    Args:
        data_dir: Directory containing the processed data
    
    Returns:
        DataFrame with options data including 'vol' column
    """
    file_path = data_dir / "spy_options.parquet"
    df = pd.read_parquet(file_path)
    
    if 'vol' not in df.columns:
        print("Warning: 'vol' column not found. Run enrich_with_iv() first.")
    
    return df


# =============================================================================
# SINGLE OPTION CALCULATION (for testing/debugging)
# =============================================================================

def calculate_iv_single(row: dict) -> float:
    """
    Calculate IV for a single option (wrapper for compatibility).
    
    Args:
        row: Dictionary with option data
    
    Returns:
        Implied volatility as float, or np.nan if calculation fails
    """
    price = np.array([row['mid_price']])
    S = np.array([row['underlying_price']])
    K = np.array([row['strike']])
    T = np.array([row['days_to_expiry'] / 365.0])
    r = np.array([row['risk_free_rate']])
    is_call = np.array([row['call_put'] == 'call'])
    
    result = implied_volatility_vec(price, S, K, T, r, is_call)
    return result[0]


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate IV for options data (vectorized)")
    parser.add_argument("--input", type=str, default=str(INPUT_FILE),
                       help="Input parquet file path")
    parser.add_argument("--output", type=str, default=str(OUTPUT_FILE),
                       help="Output parquet file path")
    
    args = parser.parse_args()
    
    # Run pipeline
    df = enrich_with_iv(
        input_file=Path(args.input),
        output_file=Path(args.output)
    )
    
    print("\n✓ Done!")
