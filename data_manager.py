"""
Data loading and preprocessing for Rough Volatility engine.
"""

import pandas as pd
import numpy as np
from huggingface_hub import hf_hub_download
import config

def load_master_data() -> pd.DataFrame:
    print(f"Downloading {config.HF_DATA_FILE} from {config.HF_DATA_REPO}...")
    file_path = hf_hub_download(
        repo_id=config.HF_DATA_REPO,
        filename=config.HF_DATA_FILE,
        repo_type="dataset",
        token=config.HF_TOKEN,
        cache_dir="./hf_cache"
    )
    df = pd.read_parquet(file_path)
    if isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index().rename(columns={'index': 'Date'})
    df['Date'] = pd.to_datetime(df['Date'])
    return df

def prepare_returns_series(df_wide: pd.DataFrame, ticker: str) -> pd.Series:
    """Extract log returns for a single ticker."""
    if ticker not in df_wide.columns:
        return pd.Series(dtype=float)
    prices = df_wide.set_index('Date')[ticker].dropna()
    returns = np.log(prices / prices.shift(1)).dropna()
    return returns

def compute_realized_volatility(returns: pd.Series, estimator: str = "parkinson") -> pd.Series:
    """
    Compute realized volatility using the specified estimator.
    Since we only have close prices, we use close-to-close squared returns for Parkinson.
    For a true Parkinson estimator, high/low prices are needed.
    Here we approximate using daily volatility scaled.
    """
    if estimator == "parkinson":
        # Approximate Parkinson: daily vol = sqrt(252) * |return| * sqrt(1 / (4*ln(2)))
        rv = np.abs(returns) * np.sqrt(252 / (4 * np.log(2)))
    else:
        rv = returns.rolling(22).std() * np.sqrt(252)
    return rv.dropna()

def prepare_volatility_series(df_wide: pd.DataFrame, tickers: list) -> pd.DataFrame:
    """Compute realized volatility for all tickers and return as wide DataFrame."""
    vol_data = {}
    for ticker in tickers:
        returns = prepare_returns_series(df_wide, ticker)
        if len(returns) > config.MIN_OBSERVATIONS:
            vol_data[ticker] = compute_realized_volatility(returns, config.RV_ESTIMATOR)
    vol_df = pd.DataFrame(vol_data).dropna()
    return vol_df
