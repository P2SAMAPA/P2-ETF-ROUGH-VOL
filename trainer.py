"""
Main training script for Rough Volatility engine.
"""

import json
import pandas as pd
import numpy as np

import config
import data_manager
from rough_vol_model import RoughVolatilityModel
import push_results

def compute_expected_return_simple(returns: pd.Series) -> float:
    """Simple expected return: recent 21-day annualized."""
    if len(returns) < 21:
        return 0.0
    return returns.iloc[-21:].mean() * 252

def run_rough_vol():
    print(f"=== P2-ETF-ROUGH-VOL Run: {config.TODAY} ===")
    df_master = data_manager.load_master_data()

    all_results = {}
    top_picks = {}

    for universe_name, tickers in config.UNIVERSES.items():
        print(f"\n--- Processing Universe: {universe_name} ---")
        universe_results = {}

        # Fetch volatility matrix once per universe
        vol_df = data_manager.prepare_volatility_series(df_master, tickers)
        if vol_df.empty:
            continue

        for ticker in tickers:
            if ticker not in vol_df.columns:
                continue

            # CRITICAL FIX: Pass the FULL volatility history to the model.
            full_volatility_series = vol_df[ticker].dropna()
            if len(full_volatility_series) < config.MIN_OBSERVATIONS:
                continue

            model = RoughVolatilityModel(
                hurst_method=config.HURST_METHOD,
                roughness_threshold=config.ROUGHNESS_THRESHOLD
            )
            
            print(f"  Estimating rolling Hurst for {ticker} (Length: {len(full_volatility_series)})...")
            success = model.fit(full_volatility_series)
            if not success:
                continue

            vol_forecast = model.forecast_volatility(full_volatility_series)
            
            # Fetch returns for this specific ticker
            returns = data_manager.prepare_returns_series(df_master, ticker)
            recent_returns = returns.iloc[-config.LOOKBACK_WINDOW:]

            exp_ret = model.compute_expected_return(recent_returns, vol_forecast["forecast"])
            exp_ret_simple = compute_expected_return_simple(recent_returns)

            # Cast numpy types to native python floats for JSON serialization
            weights = vol_forecast.get("weights", {})
            clean_weights = {k: float(v) for k, v in weights.items()}

            universe_results[ticker] = {
                "ticker": ticker,
                "hurst_exponent": float(vol_forecast["hurst"]),
                "is_rough": bool(vol_forecast["is_rough"]),
                "vol_forecast": float(vol_forecast["forecast"]) if not np.isnan(vol_forecast["forecast"]) else None,
                "expected_return_raw": float(exp_ret_simple),
                "expected_return_rough_adj": float(exp_ret),
                "weights": clean_weights
            }

        all_results[universe_name] = universe_results
        sorted_tickers = sorted(universe_results.items(),
                                key=lambda x: x[1]["expected_return_rough_adj"], reverse=True)
        top_picks[universe_name] = [
            {"ticker": t, "expected_return": d["expected_return_rough_adj"],
             "hurst": d["hurst_exponent"], "is_rough": d["is_rough"]}
            for t, d in sorted_tickers[:3]
        ]

    output_payload = {
        "run_date": config.TODAY,
        "config": {
            "lookback_window": config.LOOKBACK_WINDOW,
            "hurst_method": config.HURST_METHOD,
            "roughness_threshold": config.ROUGHNESS_THRESHOLD
        },
        "daily_trading": {
            "universes": all_results,
            "top_picks": top_picks
        }
    }

    push_results.push_daily_result(output_payload)
    print("\n=== Run Complete ===")


if __name__ == "__main__":
    run_rough_vol()
