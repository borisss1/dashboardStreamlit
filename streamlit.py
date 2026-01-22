import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pycoingecko import CoinGeckoAPI
from scipy.optimize import curve_fit

cg = CoinGeckoAPI()

st.set_page_config(page_title="BTC Merton Portfolio & Vol Surface", layout="wide")

st.title("BTC Volatility Surface & Merton Allocation")
st.markdown("""
This dashboard uses **CoinGecko** data to model BTC's volatility surface and calculate the 
**Merton Optimal Portfolio**. It breaks down your position into **Myopic Demand** (immediate risk/reward) and **Hedging Demand** (protection against volatility shifts).
""")

st.sidebar.header("Model Settings")
lookback_days = st.sidebar.slider("Lookback period (days)", 60, 365, 180)

st.sidebar.subheader("Merton Parameters")
risk_free_rate = st.sidebar.number_input("Risk-Free Rate", 0.0, 0.1, 0.04)
expected_return = st.sidebar.number_input("Expected Return (μ)", 0.0, 1.0, 0.15)
investor_gamma = st.sidebar.slider("Risk Aversion (γ)", 1.0, 10.0, 3.0)

st.sidebar.subheader("Surface Tuning")
manual_gamma_toggle = st.sidebar.checkbox("Override Estimated Gamma?")
manual_gamma = st.sidebar.slider("Manual Gamma", -2.0, 2.0, 0.0)

@st.cache_data
def fetch_btc_data(days):
    data = cg.get_coin_market_chart_by_id(id='bitcoin', vs_currency='usd', days=days)
    df = pd.DataFrame(data['prices'], columns=['ts', 'close'])
    df['ts'] = pd.to_datetime(df['ts'], unit='ms')
    df.set_index('ts', inplace=True)
    return df

def run_calculations(df):
    df['ret'] = np.log(df['close'] / df['close'].shift(1))
    df = df.dropna().copy()
    
    daily_std = df['ret'].std()
    sigma_0 = daily_std * np.sqrt(365)
    
    df['vol_7d'] = df['ret'].rolling(7).std() * np.sqrt(365)
    vol_clean = df['vol_7d'].dropna()
    
    y_fit = vol_clean.iloc[-30:].values
    t_fit = np.arange(1, len(y_fit) + 1)
    
    def vol_decay_func(t, a, b):
        return sigma_0 + a * np.exp(-b * t)

    try:
        popt, _ = curve_fit(vol_decay_func, t_fit, y_fit, p0=[0.5, 0.5])
        a, b = popt
    except:
        a, b = 0.5, 0.5
        
    S = df['close'].values
    y_vol = df['vol_7d'].values
    mask = ~np.isnan(y_vol)
    S0 = df['close'].mean()
    
    log_x = np.log(S[mask] / S0)
    log_y = np.log(y_vol[mask])
    gamma_est = np.polyfit(log_x, log_y, 1)[0]
    
    df['vol_change'] = df['vol_7d'].diff()
    rho = df['ret'].corr(df['vol_change'])
    
    return df, sigma_0, a, b, gamma_est, rho, S0

try:
    btc_df, sigma_0, a_fit, b_fit, gamma_est, rho, S0 = run_calculations(fetch_btc_data(lookback_days))
except Exception as e:
    st.error(f"Error processing data: {e}")
    st.stop()

gamma = manual_gamma if manual_gamma_toggle else gamma_est

current_vol = btc_df['vol_7d'].iloc[-1]
variance = current_vol**2

myopic_demand = (expected_return - risk_free_rate) / (investor_gamma * variance)

hedging_demand = -(rho * a_fit * b_fit) / (investor_gamma * current_vol) 

total_allocation = myopic_demand + hedging_demand

m1, m2, m3, m4 = st.columns(4)
m1.metric("Current Annual Vol", f"{current_vol:.2%}")
m2.metric("Gamma (Elasticity)", f"{gamma:.2f}")
m3.metric("Rho (Price-Vol Corr)", f"{rho:.2f}")
m4.metric("Total Weight", f"{total_allocation:.2%}")

st.divider()

col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Merton Allocation Breakdown")
    
    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        x=['Myopic Demand', 'Hedging Demand', 'Total Weight'],
        y=[myopic_demand, hedging_demand, total_allocation],
        marker_color=['#1f77b4', '#ff7f0e', '#2ca02c'],
        text=[f"{myopic_demand:.2%}", f"{hedging_demand:.2%}", f"{total_allocation:.2%}"],
        textposition='outside'
    ))
    fig_bar.update_layout(yaxis_title="Portfolio Weight", height=450)
    st.plotly_chart(fig_bar, use_container_width=True)

with col_right:
    st.subheader("3D Volatility Surface")

    t_grid = np.linspace(0, 1, 50)
    s_grid = np.linspace(btc_df['close'].min(), btc_df['close'].max(), 50)
    T_mesh, S_mesh = np.meshgrid(t_grid, s_grid)
    
    sigma_surface = (sigma_0 + a_fit * np.exp(-b_fit * T_mesh)) * (S_mesh / S0)**gamma
    
    fig_surf = go.Figure(data=[go.Surface(z=sigma_surface, x=T_mesh, y=S_mesh, colorscale='Viridis')])
    fig_surf.update_layout(
        scene=dict(xaxis_title='Time', yaxis_title='BTC Price', zaxis_title='Vol'),
        height=450, margin=dict(l=0, r=0, b=0, t=0)
    )
    st.plotly_chart(fig_surf, use_container_width=True)

st.subheader("BTC Price vs 7-Day Volatility")
fig_history = go.Figure()
fig_history.add_trace(go.Scatter(x=btc_df.index, y=btc_df['close'], name="BTC Price", yaxis="y1"))
fig_history.add_trace(go.Scatter(x=btc_df.index, y=btc_df['vol_7d'], name="7D Vol", yaxis="y2", line=dict(dash='dot')))
fig_history.update_layout(
    yaxis=dict(title="Price (USD)"),
    yaxis2=dict(title="Volatility (Annualized)", overlaying="y", side="right"),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)
st.plotly_chart(fig_history, use_container_width=True)