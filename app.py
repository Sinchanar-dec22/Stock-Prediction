import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from model import train_and_predict

st.set_page_config(page_title="Stock Price Predictor", layout="wide")

st.title("📈 Stock Price Predictor")
st.caption("Uses Linear Regression with technical indicators to predict next-day closing price.")

# Sidebar controls
with st.sidebar:
    st.header("Settings")
    ticker   = st.text_input("Ticker Symbol", value="AAPL").upper()
    period   = st.selectbox("Historical Data", ["3mo", "6mo", "1y", "2y"], index=1)
    run_btn  = st.button("Run Prediction", use_container_width=True)

# Initial welcome message
if not run_btn:
    st.info("👈 Enter a ticker symbol and click 'Run Prediction' to start forecasting!")

if run_btn:
    with st.spinner(f"Fetching data for {ticker}..."):
        df = yf.download(ticker, period=period)

    if df.empty:
        st.error("No data found. Check the ticker symbol.")
    else:
        df.index = pd.to_datetime(df.index)
        y_test, y_pred, rmse, tomorrow, test_dates = train_and_predict(df)

        current_price = float(df['Close'].iloc[-1])
        change_pct    = ((tomorrow - current_price) / current_price) * 100

        # Metric cards
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Current Price",    f"${current_price:.2f}")
        col2.metric("Predicted Tomorrow", f"${float(tomorrow):.2f}", f"{float(change_pct):+.2f}%")
        col3.metric("Model RMSE",       f"${float(rmse):.2f}")
        col4.metric("Data Points",      len(df))

        # Chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=test_dates, y=y_test,
            name="Actual", line=dict(color="#185FA5", width=2)
        ))
        fig.add_trace(go.Scatter(
            x=test_dates, y=y_pred,
            name="Predicted", line=dict(color="#639922", width=2, dash="dash")
        ))
        fig.update_layout(
            title=f"{ticker} — Actual vs Predicted (Test Set)",
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            template="plotly_white",
            height=420
        )
        st.plotly_chart(fig, use_container_width=True)

        # Raw data preview
        with st.expander("View Raw Data"):
            st.dataframe(df.tail(20), use_container_width=True)