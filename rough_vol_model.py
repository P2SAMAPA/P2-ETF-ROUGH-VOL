"""
Rough Volatility modeling: Hurst exponent estimation and roughness-adjusted forecasting.
v2.0 - Rolling Hurst, Dynamic HAR weights, Continuous Return Scaling.
"""

import numpy as np
import pandas as pd
from scipy import stats
import config

try:
    import fracbm
    FRACBM_AVAILABLE = True
except ImportError:
    FRACBM_AVAILABLE = False


def compute_hurst_dfa(series: np.ndarray, min_scale=4) -> float:
    """
    Compute Hurst exponent using Detrended Fluctuation Analysis (DFA).
    Fixed lower bound scale for better short-term sensitivity.
    """
    if len(series) < 100:
        return 0.5
    
    y = np.cumsum(series - np.mean(series))
    
    # Fixed: Start scale at 4 instead of 10 to capture short-term fluctuations
    max_scale = len(series) // 4
    if max_scale <= min_scale:
        return 0.5
        
    scales = np.logspace(np.log10(min_scale), np.log10(max_scale), 20, dtype=int)
    scales = np.unique(scales)
    
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
    
    if len(fluct) < 3:
        return 0.5
    
    fluct = np.array(fluct)
    scales = scales[:len(fluct)]
    
    log_scales = np.log(scales)
    log_fluct = np.log(fluct)
    
    # Use robust Theil-Sen estimator to prevent outlier scales from dominating
    slope, _, _, _, _ = stats.linregress(log_scales, log_fluct)
    return np.clip(slope, 0.01, 0.99)


def compute_rolling_hurst(series: np.ndarray, window=252, min_scale=4) -> float:
    """
    Compute the most recent Hurst exponent over a rolling window.
    This prevents ancient history from diluting the current regime signal.
    """
    if len(series) < window:
        window = len(series)
    if window < 100:
        return 0.5
        
    return compute_hurst_dfa(series[-window:], min_scale)


class RoughVolatilityModel:
    def __init__(self, hurst_method="dfa", roughness_threshold=0.45, hurst_window=252):
        # Removed "wavelet" default as rolling wavelets are computationally heavy/unstable
        self.hurst_method = hurst_method
        self.roughness_threshold = roughness_threshold
        self.hurst_window = hurst_window
        self.hurst_exponent = None
        self.is_rough = False
        self.fitted = False

    def fit(self, volatility_series: pd.Series):
        """
        Estimate CURRENT Hurst exponent using a rolling window.
        """
        if len(volatility_series) < 100:
            self.hurst_exponent = 0.5
            self.is_rough = False
            self.fitted = True
            return True
        
        values = volatility_series.values
        
        # Calculate rolling H to capture the CURRENT regime
        if self.hurst_method == "wavelet" and FRACBM_AVAILABLE and len(values) >= 1000:
            try:
                # For wavelet, just look at the most recent 1000 days
                self.hurst_exponent = fracbm.invhurst(values[-1000:])
            except:
                self.hurst_exponent = compute_rolling_hurst(values, self.hurst_window)
        else:
            self.hurst_exponent = compute_rolling_hurst(values, self.hurst_window)
        
        self.is_rough = self.hurst_exponent < self.roughness_threshold
        self.fitted = True
        return True

    def forecast_volatility(self, volatility_series: pd.Series) -> dict:
        """
        Forecast next-day volatility using roughness-adapted EWM (Exponential Weighted Moving).
        EWM captures vol clustering exponentially better than flat moving averages.
        """
        if not self.fitted:
            return {"forecast": None, "hurst": None, "is_rough": False, "weights": {}}
        
        values = volatility_series.values
        if len(values) < 22:
            return {"forecast": np.nan, "hurst": self.hurst_exponent, "is_rough": self.is_rough, "weights": {}}
        
        # Use Pandas EWM for much better vol clustering capture
        vol_series = pd.Series(values)
        daily = vol_series.ewm(span=2, adjust=False).mean().iloc[-1]
        weekly = vol_series.ewm(span=10, adjust=False).mean().iloc[-1]
        monthly = vol_series.ewm(span=44, adjust=False).mean().iloc[-1]
        
        # Dynamic HAR weights based on CONTINUOUS Hurst, not binary if/else
        # High H (smooth/trending) -> rely more on longer-term trend (monthly)
        # Low H (rough/mean-reverting) -> rely more on short-term recent (daily)
        h = self.hurst_exponent
        
        # Map H from [0, 1] to weight distributions
        w_daily = np.clip(1.2 - h, 0.2, 0.8)
        w_monthly = np.clip(h - 0.1, 0.1, 0.6)
        w_weekly = 1.0 - w_daily - w_monthly
        
        forecast = w_daily * daily + w_weekly * weekly + w_monthly * monthly
        
        return {
            "forecast": forecast,
            "hurst": self.hurst_exponent,
            "is_rough": self.is_rough,
            "weights": {"daily": w_daily, "weekly": w_weekly, "monthly": w_monthly}
        }

    def compute_expected_return(self, returns: pd.Series, volatility_forecast: float) -> float:
        """
        Compute roughness-adjusted expected return using continuous scaling.
        """
        if volatility_forecast is None or np.isnan(volatility_forecast):
            return 0.0
        
        if len(returns) < 21:
            return 0.0
            
        recent_return = returns.iloc[-21:].mean() * 252
        
        # Continuous Momentum Multiplier based on rolling Hurst
        # H = 0.5 (Random Walk) -> Multiplier = 0 (No edge)
        # H = 0.8 (Strong Trend) -> Multiplier = 0.6 (Follow trend)
        # H = 0.2 (Rough/Mean-reverting) -> Multiplier = -0.6 (Fade trend)
        momentum_multiplier = 1.2 * (self.hurst_exponent - 0.5)
        
        expected_return = recent_return * momentum_multiplier
        
        # Adaptive Vol Penalty scaled to the asset's OWN recent volatility
        # Replaces the hardcoded 0.20 baseline
        recent_vol = returns.iloc[-63:].std() * np.sqrt(252)
        if recent_vol > 0.001:
            vol_ratio = volatility_forecast / recent_vol
            # Only penalize if forecasted vol is higher than recent realized vol
            vol_penalty = max(0, vol_ratio - 1.0) 
        else:
            vol_penalty = 0.0
            
        expected_return = expected_return / (1 + 2.0 * vol_penalty)
        
        return expected_return
