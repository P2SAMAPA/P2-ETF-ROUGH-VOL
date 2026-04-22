# P2-ETF-ROUGH-VOL

**Rough Volatility & Fractional Dynamics – Hurst Exponent & Roughness-Adjusted ETF Ranking**

[![Daily Run](https://github.com/P2SAMAPA/P2-ETF-ROUGH-VOL/actions/workflows/daily_run.yml/badge.svg)](https://github.com/P2SAMAPA/P2-ETF-ROUGH-VOL/actions/workflows/daily_run.yml)
[![Hugging Face Dataset](https://img.shields.io/badge/🤗%20Dataset-p2--etf--rough--vol--results-blue)](https://huggingface.co/datasets/P2SAMAPA/p2-etf-rough-vol-results)

## Overview

`P2-ETF-ROUGH-VOL` estimates the **Hurst exponent** of each ETF's realized volatility series to determine whether volatility is **rough** (H < 0.45) or **smooth** (H ≥ 0.45). Rough volatility implies faster mean reversion, so expected returns are adjusted accordingly. The engine ranks ETFs by roughness-adjusted expected return.

## Methodology

1. **Realized Volatility**: Parkinson estimator (or close-to-close).
2. **Hurst Exponent**: Wavelet method (via `fracbm`) or DFA fallback.
3. **Roughness Classification**: H < 0.45 → Rough; H ≥ 0.45 → Smooth.
4. **Volatility Forecast**: Roughness-weighted HAR (daily/weekly/monthly).
5. **Expected Return**: Rough assets: mean-reversion tilt; Smooth assets: momentum tilt.

## Universe
FI/Commodities, Equity Sectors, Combined (23 ETFs)
