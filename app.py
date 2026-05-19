import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import joblib
from datetime import datetime, timedelta
from database import (init_db, get_portfolio, update_portfolio, add_trade, 
                      get_trade_history, reset_db, log_performance, 
                      get_performance_history, add_alert, get_active_alerts, deactivate_alert)
from sentiment import get_gold_sentiment

# Page config
st.set_page_config(page_title="Gold Price Predictor PRO", page_icon="💰", layout="wide")

# Initialize Persistent Database
init_db()

# Load portfolio from DB into session state if not already done
if 'balance' not in st.session_state or 'holdings' not in st.session_state:
    db_balance, db_holdings = get_portfolio()
    st.session_state.balance = db_balance
    st.session_state.holdings = db_holdings

st.title("💰 Gold Price Predictor (Professional Grade)")
st.markdown("""
This is a comprehensive Gold market analysis and trading platform. It combines **Macro-Economic data**, 
**AI-driven Sentiment Analysis**, and a **Robust ML Model** to give you the most accurate investment signals.
""")

# Load model and features
@st.cache_resource
def load_model():
    model = joblib.load("models/gold_model_v2.pkl")
    features = joblib.load("models/features_v2.pkl")
    return model, features

model, feature_names = load_model()

def calculate_indicators_macro(df):
    # Relabel columns for internal consistency if needed
    # (Assuming we use the macro data format)
    df['SMA_7_Rel'] = df['Gold'] / df['Gold'].rolling(window=7).mean()
    df['SMA_20_Rel'] = df['Gold'] / df['Gold'].rolling(window=20).mean()
    df['SMA_50_Rel'] = df['Gold'] / df['Gold'].rolling(window=50).mean()
    
    delta = df['Gold'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    df['Volatility'] = df['Gold'].pct_change().rolling(window=7).std()
    df['Gold_vs_SP500'] = df['Gold'] / df['SP500']
    df['Gold_vs_USD'] = df['Gold'] / df['USD_Index']
    
    assets = ['Gold', 'USD_Index', 'SP500', '10Y_Yield', 'Oil', 'Silver']
    for asset in assets:
        df[f'{asset}_Ret'] = df[asset].pct_change()
        for i in range(1, 4):
            df[f'Gold_Lag_{i}'] = df['Gold'].pct_change(periods=i)
            df[f'USD_Index_Lag_{i}'] = df['USD_Index'].pct_change(periods=i)
            
    return df

# Fetch latest data (Macro)
@st.cache_data(ttl=60)
def get_latest_data_pro():
    symbols = {
        "Gold": "GC=F",
        "USD_Index": "DX-Y.NYB",
        "SP500": "^GSPC",
        "10Y_Yield": "^TNX",
        "Oil": "CL=F",
        "Silver": "SI=F"
    }
    
    data_frames = []
    for name, ticker in symbols.items():
        df = yf.download(ticker, period="3mo", interval="1d", progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[['Close']].rename(columns={'Close': name})
        data_frames.append(df)
        
    final_df = data_frames[0]
    for df in data_frames[1:]:
        final_df = final_df.join(df, how='inner')
    
    # Also get exchange rates
    mad_rate = yf.download("USDMAD=X", period="1d", progress=False)['Close'].iloc[-1]
    eur_rate = yf.download("USDEUR=X", period="1d", progress=False)['Close'].iloc[-1]
    
    if isinstance(mad_rate, pd.Series): mad_rate = mad_rate.iloc[0]
    if isinstance(eur_rate, pd.Series): eur_rate = eur_rate.iloc[0]
        
    return final_df, {"MAD": mad_rate, "EUR": eur_rate, "USD": 1.0}

# Sidebar Selectors
st.sidebar.header("Trading Configuration")
selected_currency = st.sidebar.selectbox("Currency", ["MAD", "USD", "EUR"])
selected_karat = st.sidebar.selectbox("Karat Type", ["24K", "22K", "21K", "18K"])

karat_multipliers = {"24K": 1.0, "22K": 22/24, "21K": 21/24, "18K": 18/24}
currency_symbols = {"MAD": "DH", "USD": "$", "EUR": "€"}

# Constants
TROY_OUNCE_TO_GRAMS = 31.1035

data, ex_rates = get_latest_data_pro()
current_ex_rate = ex_rates[selected_currency]
karat_mult = karat_multipliers[selected_karat]
sym = currency_symbols[selected_currency]

if not data.empty:
    current_gold_price_usd = data['Gold'].iloc[-1]
    
    def convert_price(price_usd_oz):
        return (price_usd_oz * current_ex_rate * karat_mult) / TROY_OUNCE_TO_GRAMS

    current_price_final = convert_price(current_gold_price_usd)
    
    # Header Metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(f"Gold Price ({selected_currency}/g)", f"{current_price_final:,.2f} {sym}/g")
    
    # Predict
    df_proc = calculate_indicators_macro(data.copy())
    last_row = df_proc.iloc[-1:]
    X = last_row[feature_names]
    pred_return = model.predict(X)[0]
    pred_price_final = current_price_final * (1 + pred_return)
    
    col2.metric("Tomorrow Prediction", f"{pred_price_final:,.2f} {sym}", f"{pred_return:+.2%}")
    
    # Sentiment
    with st.spinner("Analyzing Market Sentiment..."):
        sent_label, sent_score, headlines = get_gold_sentiment()
    
    col3.metric("Market Sentiment", sent_label, f"{sent_score:+.2f}")
    
    # Portfolio Value
    portfolio_val = st.session_state.balance + (st.session_state.holdings * current_price_final)
    log_performance(portfolio_val)
    col4.metric("Demo Portfolio Value", f"{portfolio_val:,.2f} {sym}")

    st.divider()

    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Market Chart", "Pro Indicators", "Market News", "Demo Trading", "Price Alerts"])

    with tab1:
        st.subheader("Interactive Market Visualization")
        # Macro Comparison
        comparison = st.checkbox("Show Macro Correlation (SP500 & USD Index)")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data.index, y=convert_price(data['Gold']), name="Gold (MAD/g)"))
        if comparison:
            fig.add_trace(go.Scatter(x=data.index, y=data['SP500']/data['SP500'].iloc[0]*current_price_final, name="S&P 500 (Normalized)", opacity=0.5))
            fig.add_trace(go.Scatter(x=data.index, y=data['USD_Index']/data['USD_Index'].iloc[0]*current_price_final, name="USD Index (Normalized)", opacity=0.5))
        
        fig.update_layout(yaxis_title=f"Price ({sym}/g)", height=600)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Macro-Economic Indicators")
        c1, c2, c3 = st.columns(3)
        c1.metric("USD Index", f"{data['USD_Index'].iloc[-1]:.2f}", f"{data['USD_Index'].pct_change().iloc[-1]:+.2%}")
        c2.metric("10Y Treasury Yield", f"{data['10Y_Yield'].iloc[-1]:.2f}%", f"{data['10Y_Yield'].pct_change().iloc[-1]:+.2%}")
        c3.metric("S&P 500", f"{data['SP500'].iloc[-1]:,.2f}", f"{data['SP500'].pct_change().iloc[-1]:+.2%}")
        
        st.write("### Technical Strength")
        st.progress(min(max((sent_score + 1) / 2, 0.0), 1.0), text=f"AI Sentiment Score: {sent_score:+.2f}")

    with tab3:
        st.subheader("Real-Time Gold News & AI Sentiment")
        for h in headlines:
            st.write(f"📰 {h}")
        st.info("Sentiment is analyzed using Natural Language Processing (VADER) on live financial RSS feeds.")

    with tab4:
        st.subheader("🚀 Advanced Trading Simulator")
        c1, c2 = st.columns(2)
        with c1:
            st.write(f"**Cash:** {st.session_state.balance:,.2f} {sym}")
            st.write(f"**Gold:** {st.session_state.holdings:.3f} g")
            
            # Action controls
            action = st.radio("Action", ["Buy", "Sell"], horizontal=True)
            amount = st.number_input("Amount (g)", min_value=0.0, step=1.0)
            cost = amount * current_price_final
            
            if st.button("Execute Trade"):
                if action == "Buy":
                    if cost > st.session_state.balance: st.error("Insufficient Cash")
                    else:
                        st.session_state.balance -= cost
                        st.session_state.holdings += amount
                        update_portfolio(st.session_state.balance, st.session_state.holdings)
                        add_trade("BUY", amount, current_price_final, cost, selected_currency, selected_karat, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                        st.success("Buy successful!")
                        st.rerun()
                else:
                    if amount > st.session_state.holdings: st.error("Insufficient Gold")
                    else:
                        st.session_state.balance += cost
                        st.session_state.holdings -= amount
                        update_portfolio(st.session_state.balance, st.session_state.holdings)
                        add_trade("SELL", amount, current_price_final, cost, selected_currency, selected_karat, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                        st.success("Sell successful!")
                        st.rerun()

        with c2:
            st.write("**Performance History**")
            perf_df = get_performance_history()
            if not perf_df.empty:
                fig_perf = go.Figure()
                fig_perf.add_trace(go.Scatter(x=perf_df['time'], y=perf_df['total_value'], fill='tozeroy', name="Net Wealth"))
                fig_perf.update_layout(height=300, margin=dict(l=0,r=0,t=0,b=0))
                st.plotly_chart(fig_perf, use_container_width=True)

    with tab5:
        st.subheader("🔔 Intelligent Price Alerts")
        with st.form("alert_form"):
            target = st.number_input(f"Target Price ({sym}/g)", min_value=0.0)
            email = st.text_input("Email for notification")
            dir_alert = st.selectbox("Trigger when price goes", ["Above", "Below"])
            if st.form_submit_button("Set Alert"):
                add_alert(target, selected_currency, dir_alert, email)
                st.success(f"Alert set for {target} {sym}/g")
        
        st.write("### Active Alerts")
        st.dataframe(get_active_alerts())

else:
    st.error("Market data unavailable.")
