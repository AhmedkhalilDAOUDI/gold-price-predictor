import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import joblib
from datetime import datetime, timedelta

# Page config
st.set_page_config(page_title="Gold Price Predictor", page_icon="💰", layout="wide")

# Initialize Demo Trading Session State
if 'balance' not in st.session_state:
    st.session_state.balance = 100000.0  # Initial 100,000 in selected currency (default MAD)
if 'holdings' not in st.session_state:
    st.session_state.holdings = 0.0      # Grams of gold
if 'history' not in st.session_state:
    st.session_state.history = []

st.title("💰 Gold Price Predictor (24K Gold)")
st.markdown("""
This application uses an XGBoost model to predict **24K Gold Futures (GC=F)** price movements.
*Note: Small variances (~0.2%) may exist between sources due to the bid-ask spread and the difference between Spot and Futures prices.*
""")

# Load model and features
@st.cache_resource
def load_model():
    model = joblib.load("models/gold_model.pkl")
    features = joblib.load("models/features.pkl")
    return model, features

model, feature_names = load_model()

def calculate_indicators(df):
    # Moving Averages relative to price
    df['SMA_7_Rel'] = df['Close'] / df['Close'].rolling(window=7).mean()
    df['SMA_20_Rel'] = df['Close'] / df['Close'].rolling(window=20).mean()
    df['SMA_50_Rel'] = df['Close'] / df['Close'].rolling(window=50).mean()
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Volatility
    df['Volatility'] = df['Close'].pct_change().rolling(window=7).std()
    
    # MACD
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD_Rel'] = (exp1 - exp2) / df['Close']
    
    # Price Returns (History)
    for i in range(1, 6):
        df[f'Return_Lag_{i}'] = df['Close'].pct_change(periods=i)
        
    return df

# Fetch latest data
@st.cache_data(ttl=60)
def get_latest_data():
    gold = yf.Ticker("GC=F")
    # Fetch 1-minute interval data for the last 5 days
    df = gold.history(period="5d", interval="1m")
    
    # Get exchange rates
    try:
        mad_rate = yf.Ticker("USDMAD=X").history(period="1d")["Close"].iloc[-1]
        eur_rate = yf.Ticker("USDEUR=X").history(period="1d")["Close"].iloc[-1]
    except:
        mad_rate = 10.0 
        eur_rate = 0.92
        
    return df, {"MAD": mad_rate, "EUR": eur_rate, "USD": 1.0}

# Sidebar Selectors
st.sidebar.header("Options")
selected_currency = st.sidebar.selectbox("Currency", ["MAD", "USD", "EUR"])
selected_karat = st.sidebar.selectbox("Karat Type", ["24K", "22K", "21K", "18K"])

karat_multipliers = {
    "24K": 1.0,
    "22K": 22/24,
    "21K": 21/24,
    "18K": 18/24
}

currency_symbols = {
    "MAD": "DH",
    "USD": "$",
    "EUR": "€"
}

# Constants
TROY_OUNCE_TO_GRAMS = 31.1035

data, ex_rates = get_latest_data()
current_ex_rate = ex_rates[selected_currency]
karat_mult = karat_multipliers[selected_karat]
sym = currency_symbols[selected_currency]

if not data.empty:
    # Last update time
    last_update = data.index[-1].strftime("%Y-%m-%d %H:%M:%S")
    st.sidebar.info(f"Last Market Update: {last_update}")
    st.sidebar.write(f"**Source:** Yahoo Finance (GC=F)")
    if selected_currency != "USD":
        st.sidebar.write(f"**Exchange Rate:** 1 USD = {current_ex_rate:.4f} {selected_currency}")
    st.sidebar.write(f"**Unit:** Price per Gram ({selected_karat})")

    # Convert prices to selected currency and karat per gram
    def convert_price(price_usd_oz):
        return (price_usd_oz * current_ex_rate * karat_mult) / TROY_OUNCE_TO_GRAMS

    current_price_final = convert_price(data['Close'].iloc[-1])
    prev_price_final = convert_price(data['Close'].iloc[-2])
    
    change_final = current_price_final - prev_price_final
    pct_change = (change_final / prev_price_final) * 100

    col1, col2, col3 = st.columns(3)
    col1.metric(f"Current Price ({selected_currency}/g)", f"{current_price_final:,.2f} {sym}/g", f"{change_final:+.2f} ({pct_change:+.2f}%)")
    
    # Prepare data for prediction
    df_proc = calculate_indicators(data.copy())
    last_row = df_proc.iloc[-1:]
    X = last_row[feature_names]
    
    # Prediction (Returns are the same regardless of currency/unit)
    pred_return = model.predict(X)[0]
    pred_price_final = current_price_final * (1 + pred_return)
    
    col2.metric("Predicted Price (Tomorrow)", f"{pred_price_final:,.2f} {sym}/g", f"{pred_return:+.2%}")
    
    # Signal
    threshold = 0.002 # 0.2% threshold for action
    if pred_return > threshold:
        signal = "INVEST (BUY)"
        color = "green"
        recommendation = "The model predicts a significant upward movement. High potential for gains."
    elif pred_return < -threshold:
        signal = "SELL / SHORT"
        color = "red"
        recommendation = "The model predicts a downward trend. Consider securing profits or selling."
    else:
        signal = "HOLD"
        color = "orange"
        recommendation = "Market conditions are stable. No strong signal to buy or sell."

    col3.markdown(f"### Signal: <span style='color:{color}'>{signal}</span>", unsafe_allow_html=True)

    st.divider()

    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["Chart", "Technical Indicators", "Model Confidence", "Demo Trading"])

    with tab1:
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=data.index,
                        open=convert_price(data['Open']),
                        high=convert_price(data['High']),
                        low=convert_price(data['Low']),
                        close=convert_price(data['Close']),
                        name=f'Market Data ({selected_currency}/g)'))
        fig.update_layout(title=f"Gold Price History ({selected_karat} in {selected_currency})", yaxis_title=f"Price ({sym}/g)", height=600)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Key Technical Indicators")
        c1, c2, c3 = st.columns(3)
        
        rsi = last_row['RSI'].iloc[0]
        c1.metric("RSI (14)", f"{rsi:.2f}")
        if rsi > 70: c1.warning("Overbought")
        elif rsi < 30: c1.success("Oversold")
        else: c1.info("Neutral")
        
        macd = last_row['MACD_Rel'].iloc[0]
        c2.metric("MACD (Relative)", f"{macd:.4f}")
        
        vol = last_row['Volatility'].iloc[0]
        c3.metric("Volatility (7d)", f"{vol:.2%}")

    with tab3:
        st.subheader("Recommendation Details")
        st.write(recommendation)
        st.info("""
        **Disclaimer:** Trading gold involves significant risk. This model is for informational purposes only 
        and does not constitute financial advice. Always perform your own research and never invest money you cannot afford to lose.
        """)
        
        st.write("### Prediction Confidence Factors")
        # Show top contributing features for this specific prediction (SHAP-like if we had SHAP, but we'll show values)
        st.write("Current Feature Values:")
        st.dataframe(X.T.rename(columns={X.index[0]: "Value"}))

    with tab4:
        st.subheader("🚀 Demo Trading Simulator")
        
        # Portfolio Summary
        portfolio_value = st.session_state.balance + (st.session_state.holdings * current_price_final)
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Cash Balance", f"{st.session_state.balance:,.2f} {sym}")
        c2.metric("Gold Holdings", f"{st.session_state.holdings:.3f} g")
        c3.metric("Total Portfolio Value", f"{portfolio_value:,.2f} {sym}")
        
        st.divider()
        
        # Trading Controls
        col_trade1, col_trade2 = st.columns(2)
        
        with col_trade1:
            st.write("### Buy Gold")
            buy_amount = st.number_input("Amount to buy (grams)", min_value=0.0, step=0.1, key="buy_input")
            cost = buy_amount * current_price_final
            st.write(f"Estimated Cost: **{cost:,.2f} {sym}**")
            
            if st.button("Confirm Purchase"):
                if cost > st.session_state.balance:
                    st.error("Insufficient balance!")
                elif buy_amount > 0:
                    st.session_state.balance -= cost
                    st.session_state.holdings += buy_amount
                    st.session_state.history.append({
                        "Type": "BUY",
                        "Amount": buy_amount,
                        "Price": current_price_final,
                        "Total": cost,
                        "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    })
                    st.success(f"Bought {buy_amount:.3f}g of gold!")
                    st.rerun()

        with col_trade2:
            st.write("### Sell Gold")
            sell_amount = st.number_input("Amount to sell (grams)", min_value=0.0, max_value=st.session_state.holdings, step=0.1, key="sell_input")
            revenue = sell_amount * current_price_final
            st.write(f"Estimated Revenue: **{revenue:,.2f} {sym}**")
            
            if st.button("Confirm Sale"):
                if sell_amount > st.session_state.holdings:
                    st.error("Not enough gold to sell!")
                elif sell_amount > 0:
                    st.session_state.balance += revenue
                    st.session_state.holdings -= sell_amount
                    st.session_state.history.append({
                        "Type": "SELL",
                        "Amount": sell_amount,
                        "Price": current_price_final,
                        "Total": revenue,
                        "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    })
                    st.success(f"Sold {sell_amount:.3f}g of gold!")
                    st.rerun()

        st.divider()
        st.write("### Trade History")
        if st.session_state.history:
            history_df = pd.DataFrame(st.session_state.history)
            st.table(history_df.iloc[::-1]) # Show latest first
        else:
            st.write("No trades yet.")
            
        if st.button("Reset Portfolio"):
            st.session_state.balance = 100000.0
            st.session_state.holdings = 0.0
            st.session_state.history = []
            st.rerun()

else:
    st.error("Could not fetch market data. Please check your internet connection.")
