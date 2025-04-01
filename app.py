import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
import os
from datetime import datetime
from oandapyV20 import API
from oandapyV20.endpoints.orders import OrderCreate

# === CONFIG ===
TRADE_LOG_PATH = "eurusd_trade_log.csv"
MAX_UNITS = 1000
TP_PIPS = 0.0050
SL_PIPS = 0.0030
REFRESH_INTERVAL = 900  # seconds (15 min)

# === AUTO REFRESH ===
if "last_refresh" not in st.session_state:
    st.session_state.last_refresh = time.time()
if time.time() - st.session_state.last_refresh > REFRESH_INTERVAL:
    st.session_state.last_refresh = time.time()
    st.experimental_rerun()

# === Load model and data ===
model = joblib.load("xgb_t5_model.pkl")
df = pd.read_csv("EURUSD_with_SR_enriched.csv")

# === Extract latest signal ===
features = ['wma_wma', 'ema_ema', 'sar_sar', 'rsi_rsi', 'cci_cci', 'macd_signal']
latest = df.iloc[[-1]].copy()
X_latest = latest[features]
prob_up = model.predict_proba(X_latest)[0][1]

# === Signal logic ===
latest['confident_signal'] = np.where(prob_up > 0.55, 1, np.where(prob_up < 0.45, 0, np.nan))
trend_long = (latest['close'].values[0] > latest['ema_ema'].values[0]) and (latest['close'].values[0] > latest['sar_sar'].values[0])
trend_short = (latest['close'].values[0] < latest['ema_ema'].values[0]) and (latest['close'].values[0] < latest['sar_sar'].values[0])
sr_zone = latest['sr_zone'].values[0]
latest_close = float(latest['close'].values[0])
today = datetime.now().strftime("%Y-%m-%d")

signal = "NO TRADE"
if latest['confident_signal'].values[0] == 1 and trend_long and sr_zone == 'near_support':
    signal = "BUY"
elif latest['confident_signal'].values[0] == 0 and trend_short and sr_zone == 'near_resistance':
    signal = "SELL"

# === Display UI ===
st.title("📈 EUR/USD Signal Dashboard")
st.write(f"**Date**: {latest['date'].values[0]}")
st.write(f"**Close**: {latest_close}")
st.write(f"**Model Confidence (Up)**: {prob_up:.2%}")
st.write(f"**Trend**: {'Long' if trend_long else 'Short' if trend_short else 'Neutral'}")
st.write(f"**S/R Zone**: {sr_zone}")
st.subheader(f"📍 Final Signal: {signal}")

# === Load Trade Log ===
if os.path.exists(TRADE_LOG_PATH):
    trade_log = pd.read_csv(TRADE_LOG_PATH)
else:
    trade_log = pd.DataFrame(columns=["date", "signal", "units", "price", "tp", "sl", "pnl"])

already_traded = today in trade_log["date"].values

# === Trade execution ===
if signal in ["BUY", "SELL"] and not already_traded:
    try:
        # Price targets
        if signal == "BUY":
            tp = round(latest_close + TP_PIPS, 5)
            sl = round(latest_close - SL_PIPS, 5)
            units = MAX_UNITS
        else:
            tp = round(latest_close - TP_PIPS, 5)
            sl = round(latest_close + SL_PIPS, 5)
            units = -MAX_UNITS

        # OANDA credentials from secrets
        OANDA_API_KEY = st.secrets["OANDA_API_KEY"]
        OANDA_ACCOUNT_ID = st.secrets["OANDA_ACCOUNT_ID"]
        client = API(access_token=OANDA_API_KEY, environment="practice")

        order_data = {
            "order": {
                "instrument": "EUR_USD",
                "units": str(units),
                "type": "MARKET",
                "positionFill": "DEFAULT",
                "takeProfitOnFill": {"price": str(tp)},
                "stopLossOnFill": {"price": str(sl)}
            }
        }

        r = OrderCreate(accountID=OANDA_ACCOUNT_ID, data=order_data)
        client.request(r)

        # Log trade
        trade_log.loc[len(trade_log)] = {
            "date": today,
            "signal": signal,
            "units": units,
            "price": latest_close,
            "tp": tp,
            "sl": sl,
            "pnl": None
        }
        trade_log.to_csv(TRADE_LOG_PATH, index=False)

        st.success(f"✅ Auto-Traded: {signal} {abs(units)} units @ {latest_close}, TP: {tp}, SL: {sl}")

    except Exception as e:
        st.error(f"❌ Trade execution failed: {e}")

elif already_traded:
    st.info("📛 Trade already placed today. Limit: 1/day.")
else:
    st.info("⚪ No trade signal at this time.")

# === Show Trade Log ===
if not trade_log.empty:
    st.subheader("📊 Trade Log")
    st.dataframe(trade_log.tail(10))
