"""
Rough Volatility modeling: Hurst exponent estimation and roughness-adjusted forecasting.
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression

try:
    import fracbm
    FRACBM_AVAILABLE = True
except ImportError:
    FRACBM_AVAILABLE = False
    print("Warning: fracbm not installed. Using DFA fallback for Hurst estimation.")


def compute_hurst_dfa(series: np.ndarray) -> float:
    """
    Compute Hurst exponent using Detrended Fluctuation Analysis (DFA).
    """
    if len(series) < 100:
        return 0.5
    
    # Cumulative sum of deviations from mean
    y = np.cumsum(series - np.mean(series))
    
    scales = np.logspace(1, np.log10(len(series)//4), 20, dtype=int)
    fluct = []
    
    for scale in scales:
        n_segments = len(y) // scale
        if n_segments < 2:
            continue
        f = 0.0
        for i in range(n_segments):
            seg = y[i*scale:(i+1)*scale]
            x = np.arange(len(seg))
            coef = np.polyfit(x, seg, 1)
            trend = np.polyval(coef, x)
            f += np.sum((seg - trend)**2)
        f = np.sqrt(f / (n_segments * scale))
        fluct.append(f)
    
    fluct = np.array(fluct)
    scales = scales[:len(fluct)]
    
    if len(fluct) < 3:
        return 0.5
    
    log_scales = np.log(scales)
    log_fluct = np.log(fluct)
    slope, _, _, _, _ = stats.linregress(log_scales, log_fluct)
    return slope


def compute_hurst_wavelet(series: np.ndarray) -> float:
    """
    Compute Hurst exponent using wavelet method (via fracbm if available).
    """
    if FRACBM_AVAILABLE and len(series) >= 1000:
        try:
            return fracbm.invhurst(series)
        except:
            pass
    return compute_hurst_dfa(series)


class RoughVolatilityModel:
    def __init__(self, hurst_method="wavelet", roughness_threshold=0.45):
        self.hurst_method = hurst_method
        self.roughness_threshold = roughness_threshold
        self.hurst_exponent = None
        self.is_rough = False
        self.fitted = False

    def fit(self, volatility_series: pd.Series):
        """Estimate Hurst exponent from volatility series."""
        if len(volatility_series) < config.MIN_OBSERVATIONS:
            return False
        
        values = volatility_series.values
        if self.hurst_method == "wavelet":
            self.hurst_exponent = compute_hurst_wavelet(values)
        else:
            self.hurst_exponent = compute_hurst_dfa(values)
        
        self.is_rough = self.hurst_exponent < self.roughness_threshold
        self.fitted = True
        return True

    def forecast_volatility(self, volatility_series: pd.Series) -> dict:
        """
        Forecast next-day volatility using roughness-adjusted HAR model.
        """
        if not self.fitted:
            return {"forecast": None, "hurst": None, "is_rough": False}
        
        values = volatility_series.values
        if len(values) < 22:
            return {"forecast": np.nan, "hurst": self.hurst_exponent, "is_rough": self.is_rough}
        
        # HAR features
        daily = values[-1]
        weekly = np.mean(values[-5:]) if len(values) >= 5 else daily
        monthly = np.mean(values[-22:]) if len(values) >= 22 else daily
        
        # Roughness adjustment: if rough (H < 0.45), volatility reverts faster
        # Adjust HAR coefficients based on Hurst exponent
        if self.is_rough:
            # Rough volatility: more weight on recent (daily), less on long memory
            w_daily, w_weekly, w_monthly = 0.6, 0.3, 0.1
        else:
            # Smooth volatility: standard HAR weights
            w_daily, w_weekly, w_monthly = 0.3, 0.4, 0.3
        
        forecast = w_daily * daily + w_weekly * weekly + w_monthly * monthly
        
        return {
            "forecast": forecast,
            "hurst": self.hurst_exponent,
            "is_rough": self.is_rough,
            "weights": {"daily": w_daily, "weekly": w_weekly, "monthly": w_monthly}
        }

    def compute_expected_return(self, returns: pd.Series, volatility_forecast: float) -> float:
        """
        Compute roughness-adjusted expected return.
        Rough assets (H < 0.45) tend to have faster mean reversion, so expected return
        is adjusted based on recent momentum and roughness.
        """
        if volatility_forecast is None or np.isnan(volatility_forecast):
            return 0.0
        
        recent_return = returns.iloc[-21:].mean() * 252  # Annualized
        
        if self.is_rough:
            # Rough assets: mean-reversion dominates, expected return is negative of recent momentum
            expected_return = -0.5 * recent_return
        else:
            # Smooth assets: momentum persists
            expected_return = 0.3 * recent_return
        
        # Scale by volatility forecast (higher vol = lower confidence)
        vol_penalty = volatility_forecast / 0.20  # normalize by ~20% annual vol
        expected_return = expected_return / (1 + vol_penalty)
        
        return expected_return
