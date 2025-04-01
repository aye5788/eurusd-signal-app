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

        # OANDA credentials from Streamlit secrets
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

# === Update PnL for previous trades ===
enriched_df = pd.read_csv("EURUSD_with_SR_enriched.csv")
enriched_df["date"] = pd.to_datetime(enriched_df["date"])
open_trades = trade_log[trade_log["pnl"].isna()].copy()

for idx, row in open_trades.iterrows():
    trade_date = pd.to_datetime(row["date"])
    direction = 1 if row["signal"] == "BUY" else -1
    tp = row["tp"]
    sl = row["sl"]
    entry_price = row["price"]

    next_day = enriched_df[enriched_df["date"] == trade_date + pd.Timedelta(days=1)]
    if next_day.empty:
        continue

    high = next_day["high"].values[0]
    low = next_day["low"].values[0]

    if direction == 1:
        if high >= tp:
            pnl = (tp - entry_price) * row["units"]
        elif low <= sl:
            pnl = (sl - entry_price) * row["units"]
        else:
            pnl = 0
    else:
        if low <= tp:
            pnl = (entry_price - tp) * abs(row["units"])
        elif high >= sl:
            pnl = (entry_price - sl) * abs(row["units"])
        else:
            pnl = 0

    trade_log.at[idx, "pnl"] = round(pnl, 2)

# Save PnL-updated trade log
trade_log.to_csv(TRADE_LOG_PATH, index=False)

# === Show Trade Log ===
if not trade_log.empty:
    st.subheader("📊 Trade Log (Last 10)")
    st.dataframe(trade_log.tail(10))
