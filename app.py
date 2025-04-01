import streamlit as st
import pandas as pd
import joblib
import numpy as np
from oandapyV20 import API
from oandapyV20.endpoints.orders import OrderCreate

# === Load model and data ===
model = joblib.load('xgb_t5_model.pkl')
df = pd.read_csv('EURUSD_with_SR_enriched.csv')

# === Extract latest row ===
features = ['wma_wma', 'ema_ema', 'sar_sar', 'rsi_rsi', 'cci_cci', 'macd_signal']
latest = df.iloc[[-1]].copy()
X_latest = latest[features]
prob_up = model.predict_proba(X_latest)[0][1]

# === Signal Logic ===
latest['confident_signal'] = np.where(prob_up > 0.55, 1, np.where(prob_up < 0.45, 0, np.nan))
trend_long = (latest['close'].values[0] > latest['ema_ema'].values[0]) and (latest['close'].values[0] > latest['sar_sar'].values[0])
trend_short = (latest['close'].values[0] < latest['ema_ema'].values[0]) and (latest['close'].values[0] < latest['sar_sar'].values[0])
sr_zone = latest['sr_zone'].values[0]

signal = '⚪ NO TRADE'
if latest['confident_signal'].values[0] == 1 and trend_long and sr_zone == 'near_support':
    signal = '📗 LONG Signal'
elif latest['confident_signal'].values[0] == 0 and trend_short and sr_zone == 'near_resistance':
    signal = '📕 SHORT Signal'

# === Streamlit Dashboard ===
st.title("📈 EUR/USD Signal Dashboard")
st.write(f"**Date**: {latest['date'].values[0]}")
st.write(f"**Close**: {latest['close'].values[0]}")
st.write(f"**Model Confidence (Up)**: {prob_up:.2%}")
st.write(f"**Trend**: {'Long' if trend_long else 'Short' if trend_short else 'Neutral'}")
st.write(f"**S/R Zone**: {sr_zone}")
st.subheader(f"📍 Final Signal: {signal}")

# === Execute Trade ===
if st.button("🚀 Place Trade on OANDA"):
    OANDA_API_KEY = st.secrets["OANDA_API_KEY"]
    OANDA_ACCOUNT_ID = st.secrets["OANDA_ACCOUNT_ID"]
    client = API(access_token=OANDA_API_KEY, environment="practice")

    direction = 'buy' if signal == '📗 LONG Signal' else 'sell'
    units = 1000 if direction == 'buy' else -1000

    order = {
        "order": {
            "instrument": "EUR_USD",
            "units": str(units),
            "type": "MARKET",
            "positionFill": "DEFAULT"
        }
    }

    try:
        r = OrderCreate(accountID=OANDA_ACCOUNT_ID, data=order)
        client.request(r)
        st.success(f"✅ Trade sent: {direction.upper()} {units} EUR/USD")
    except Exception as e:
        st.error(f"❌ Trade failed: {e}")
