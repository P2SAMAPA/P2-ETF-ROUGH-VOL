"""
Streamlit Dashboard for Rough Volatility Engine.
"""

import streamlit as st
import pandas as pd
from huggingface_hub import HfApi, hf_hub_download
import json
import config
from us_calendar import USMarketCalendar

st.set_page_config(page_title="P2Quant Rough Vol", page_icon="📊", layout="wide")

st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: 600; color: #1f77b4; }
    .hero-card { background: linear-gradient(135deg, #1f77b4 0%, #2C5282 100%); border-radius: 16px; padding: 2rem; color: white; text-align: center; }
    .hero-ticker { font-size: 4rem; font-weight: 800; }
    .rough-badge { background: #dc3545; color: white; padding: 0.2rem 0.8rem; border-radius: 20px; font-size: 0.9rem; }
    .smooth-badge { background: #28a745; color: white; padding: 0.2rem 0.8rem; border-radius: 20px; font-size: 0.9rem; }
    .explain-box { background: #f8f9fa; border-radius: 12px; padding: 1.5rem; margin-top: 2rem; }
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=3600)
def load_latest_results():
    try:
        api = HfApi(token=config.HF_TOKEN)
        files = api.list_repo_files(repo_id=config.HF_OUTPUT_REPO, repo_type="dataset")
        json_files = sorted([f for f in files if f.startswith("rough_vol_") and f.endswith('.json')], reverse=True)
        if not json_files:
            return None
        local_path = hf_hub_download(
            repo_id=config.HF_OUTPUT_REPO, filename=json_files[0],
            repo_type="dataset", token=config.HF_TOKEN, cache_dir="./hf_cache"
        )
        with open(local_path) as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        return None

# --- Sidebar ---
st.sidebar.markdown("## ⚙️ Configuration")
calendar = USMarketCalendar()
st.sidebar.markdown(f"**📅 Next Trading Day:** {calendar.next_trading_day().strftime('%Y-%m-%d')}")
data = load_latest_results()
if data:
    st.sidebar.markdown(f"**Run Date:** {data.get('run_date', 'Unknown')}")

st.markdown('<div class="main-header">📊 P2Quant Rough Volatility</div>', unsafe_allow_html=True)
st.markdown('<div>Fractional Dynamics – Hurst Exponent & Roughness-Adjusted Ranking</div>', unsafe_allow_html=True)

if data is None:
    st.warning("No data available.")
    st.stop()

daily = data['daily_trading']
universes = daily['universes']
top_picks = daily['top_picks']

tabs = st.tabs(["📊 Combined", "📈 Equity Sectors", "💰 FI/Commodities"])
universe_keys = ["COMBINED", "EQUITY_SECTORS", "FI_COMMODITIES"]

for tab, key in zip(tabs, universe_keys):
    with tab:
        top = top_picks.get(key, [])
        universe_data = universes.get(key, {})
        if top:
            pick = top[0]
            ticker = pick['ticker']
            ret = pick['expected_return']
            hurst = pick['hurst']
            is_rough = pick['is_rough']
            badge = '<span class="rough-badge">ROUGH</span>' if is_rough else '<span class="smooth-badge">SMOOTH</span>'
            st.markdown(f"""
            <div class="hero-card">
                <div style="font-size: 1.2rem; opacity: 0.8;">📊 TOP PICK (Roughness-Adjusted)</div>
                <div class="hero-ticker">{ticker}</div>
                <div style="font-size: 1.5rem;">Expected Return: {ret*100:.2f}%</div>
                <div style="margin-top: 1rem;">Hurst: {hurst:.3f} {badge}</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("### Top 3 Predictions")
            rows = []
            for p in top:
                rows.append({
                    "Ticker": p['ticker'],
                    "Exp Return": f"{p['expected_return']*100:.2f}%",
                    "Hurst": f"{p['hurst']:.3f}",
                    "Type": "Rough" if p['is_rough'] else "Smooth"
                })
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True, hide_index=True)

            st.markdown("### All ETFs")
            all_rows = []
            for t, d in universe_data.items():
                all_rows.append({
                    "Ticker": t,
                    "Exp Return (Raw)": f"{d['expected_return_raw']*100:.2f}%",
                    "Exp Return (Rough Adj)": f"{d['expected_return_rough_adj']*100:.2f}%",
                    "Hurst": f"{d['hurst_exponent']:.3f}",
                    "Vol Forecast": f"{d['vol_forecast']*100:.2f}%" if d['vol_forecast'] else "N/A"
                })
            df_all = pd.DataFrame(all_rows).sort_values("Exp Return (Rough Adj)", ascending=False)
            st.dataframe(df_all, use_container_width=True, hide_index=True)

        # --- Explanation expander at the bottom of each tab ---
        with st.expander("📘 What does 'Roughness-Adjusted' mean?"):
            st.markdown("""
            ### Hurst Exponent & Rough Volatility
            
            The **Hurst exponent (H)** measures the long‑memory of a time series:
            - **H = 0.5**: Random walk (no memory)
            - **H > 0.5**: Persistent / trending (smooth volatility)
            - **H < 0.5**: Anti‑persistent / mean‑reverting (**rough volatility**)
            
            This engine estimates H for each ETF's realized volatility series.
            
            ### Roughness-Adjusted Expected Return
            
            | Volatility Type | Hurst Range | Expected Return Adjustment |
            |-----------------|-------------|----------------------------|
            | **Rough** | H < {threshold} | Mean‑reversion dominates: `Exp Return = -0.5 × Recent Return` |
            | **Smooth** | H ≥ {threshold} | Momentum persists: `Exp Return = +0.3 × Recent Return` |
            
            Both are scaled by `1 / (1 + Vol_Forecast / 0.20)` to penalize high‑volatility regimes.
            
            **Why this matters:** Rough volatility implies faster mean reversion, so chasing recent momentum is dangerous. The engine tilts away from recent winners when volatility is rough.
            """.format(threshold=config.ROUGHNESS_THRESHOLD))
