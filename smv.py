import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.linear_model import LinearRegression
import plotly.graph_objs as go
from datetime import datetime, timedelta

st.set_page_config(
    page_title="Stock Market",
    page_icon="📈",
    layout="wide"
)

st.title("Stock Market Visualizer")

st.markdown("**[Find Stock Symbols on Yahoo Finance](https://finance.yahoo.com/lookup/)**")
st.markdown("### Popular Tickers:")
st.markdown("**US Stocks:** AAPL(Apple), MSFT(Microsoft), GOOGL(Google), TSLA(Tesla), AMZN(Amazon)")
st.markdown("**Indian Stocks:** TCS.NS(TCS), INFY.NS(Infosys), RELIANCE.NS(Reliance), HDFCBANK.NS(HDFC Bank)")

popular_stocks = {
    "Apple (AAPL)": "AAPL",
    "Microsoft (MSFT)": "MSFT",
    "Google (GOOGL)": "GOOGL",
    "Amazon (AMZN)": "AMZN",
    "Tesla (TSLA)": "TSLA",
    "Meta (META)": "META",
    "NVIDIA (NVDA)": "NVDA",
    "Netflix (NFLX)": "NFLX",
    "Tata Consultancy (TCS.NS)": "TCS.NS",
    "Reliance Industries (RELIANCE.NS)": "RELIANCE.NS",
    "Infosys (INFY.NS)": "INFY.NS",
    "HDFC Bank (HDFCBANK.NS)": "HDFCBANK.NS",
    "Custom...": "CUSTOM"
}

stock_choice = st.sidebar.selectbox("**Select Stock**", options=list(popular_stocks.keys()))

if popular_stocks[stock_choice] == "CUSTOM":
    ticker = st.sidebar.text_input("Enter Stock Symbol").strip().upper()
else:
    ticker = popular_stocks[stock_choice]
    st.sidebar.info(f"Selected: **{ticker}**")

period_options = {
    "1mo": "1 Month",
    "6mo": "6 Months",
    "1y": "1 Year",
    "5y": "5 Years"
}

period = st.sidebar.selectbox(
    "**Historical Period**",
    options=list(period_options.keys()),
    index=2,
    format_func=lambda x: period_options[x]
)

predict_days = st.sidebar.slider(
    "**Prediction Days**",
    min_value=1,
    max_value=30,
    value=7
)

analyze_button = st.sidebar.button("Analyze Stock", type="primary", use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.markdown("All stock values automatically converted to Indian Rupees (₹)")
st.sidebar.markdown("***Educational purposes only***")
st.sidebar.markdown("---")
st.sidebar.markdown("Done by : *Tanya Roy, Tanvi Thakare, V Monika, Thanmayi R Kashyap, Vaishnavi Kumarak*")

def indian_stock(ticker_symbol):
    ticker_upper = ticker_symbol.upper()
    indian_suffixes = [".NS", ".BO", ".BSE", ".NSX"]
    for suffix in indian_suffixes:
        if ticker_upper.endswith(suffix):
            return True
    return False

def usd_to_inr():
    try:
        fx_data = yf.download("USDINR=X", period="10d", progress=False)
        if fx_data is not None and not fx_data.empty:
            rate = float(fx_data["Close"].iloc[-1])
            return rate
        else:
            st.warning("Could not fetch USD→INR rate. Using default: 89.0")
            return 89.0
    except:
        st.warning("Error fetching USD→INR rate. Using default: 89.0")
        return 89.0

def fetch_stock_data(ticker_symbol, time_period):
    try:
        data = yf.download(
            ticker_symbol,
            period=time_period,
            interval="1d",
            progress=False,
            auto_adjust=True
        )

        if data is None or data.empty:
            return None

        data = data.reset_index()
        data['Date'] = pd.to_datetime(data['Date']).dt.tz_localize(None)

        return data
    except:
        return None

def moving_avg(dataframe):
    dataframe = dataframe.copy()
    dataframe = dataframe.sort_values('Date')
    dataframe['MA20'] = dataframe['Close'].rolling(window=20).mean()
    dataframe['MA50'] = dataframe['Close'].rolling(window=50).mean()
    return dataframe

def returns_volatility(dataframe):
    dataframe = dataframe.copy()
    dataframe['Daily_Return'] = dataframe['Close'].pct_change() * 100
    dataframe['Volatility_10d'] = dataframe['Close'].pct_change().rolling(window=10).std() * 100
    return dataframe

def future_prices(dataframe, days_ahead):

    dataframe_sorted = dataframe.sort_values('Date').copy()
    close_prices = dataframe_sorted['Close'].values

    if len(close_prices) < 10:
        return None

    window = min(60, len(close_prices))
    recent_prices = close_prices[-window:]

    X = np.arange(len(recent_prices)).reshape(-1, 1)
    y = recent_prices

    model = LinearRegression()
    model.fit(X, y)

    future_X = np.arange(len(recent_prices), len(recent_prices) + days_ahead).reshape(-1, 1)
    predicted_prices = model.predict(future_X)

    last_date = dataframe_sorted['Date'].iloc[-1]

    future_dates = []
    for i in range(days_ahead):
        future_dates.append(last_date + timedelta(days=i+1))

    prediction_df = pd.DataFrame({
        'Date': future_dates,
        'Predicted_Close': predicted_prices.flatten()
    })

    return prediction_df

def trend(dataframe):

    window = min(30, len(dataframe))

    if window < 10:
        return "N/A", 0.0

    recent_data = dataframe.tail(window).copy()
    close_prices = recent_data['Close'].values

    X = np.arange(len(close_prices)).reshape(-1, 1)
    y = close_prices

    model = LinearRegression()
    model.fit(X, y)

    slope = float(model.coef_[0])

    if slope > 0.5:
        trend = "UPTREND"
    elif slope < -0.5:
        trend = "DOWNTREND"
    else:
        trend = "SIDEWAYS"

    return trend, slope

def latest_value(series):
    try:
        values = series.values
        for i in range(len(values)-1, -1, -1):
            if pd.notna(values[i]):
                return float(values[i])
        return 0.0
    except:
        return 0.0


if analyze_button or 'stored_ticker' in st.session_state:

    if analyze_button:
        st.session_state['stored_ticker'] = ticker
        st.session_state['stored_period'] = period
        st.session_state['stored_predict_days'] = predict_days

    ticker = st.session_state.get('stored_ticker', ticker)
    period = st.session_state.get('stored_period', period)
    predict_days = st.session_state.get('stored_predict_days', predict_days)

    with st.spinner(f"Fetching data for {ticker}..."):
        stock_df = fetch_stock_data(ticker, period)

    if stock_df is None or stock_df.empty:
        st.error(f"No data found for '{ticker}'. Please check the symbol and try again.")

    else:

        stock_df = moving_avg(stock_df)
        stock_df = returns_volatility(stock_df)

        is_indian = indian_stock(ticker)

        if is_indian:
            conversion_rate = 1.0
        else:
            conversion_rate = usd_to_inr()

        stock_df['Close_INR'] = stock_df['Close'] * conversion_rate
        stock_df['MA20_INR'] = stock_df['MA20'] * conversion_rate
        stock_df['MA50_INR'] = stock_df['MA50'] * conversion_rate

        latest_close_inr = latest_value(stock_df['Close_INR'])
        latest_return = latest_value(stock_df['Daily_Return'])
        latest_volatility = latest_value(stock_df['Volatility_10d'])

        trend_text, trend_slope = trend(stock_df)

        st.markdown("---")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Latest Close", f"₹{latest_close_inr:.2f}")

        with col2:
            st.metric("Daily Return", f"{latest_return:.2f}%", delta=f"{latest_return:.2f}%")

        with col3:
            st.metric("Volatility (10d)", f"{latest_volatility:.2f}%")

        with col4:
            st.metric("Trend", trend_text)

        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "**Price Chart**",
            "**Moving Averages**",
            "**Returns & Volatility**",
            "**Prediction**",
            "**Data Table**"
        ])

        with tab1:

            fig1 = go.Figure()

            fig1.add_trace(go.Scatter(
                x=stock_df['Date'],
                y=stock_df['Close_INR'],
                mode='lines',
                name='Close Price'
            ))

            fig1.update_layout(
                xaxis_title='Date',
                yaxis_title='Price (₹)',
                template='plotly_white'
            )

            st.plotly_chart(fig1, use_container_width=True)

        with tab4:

            prediction_data = future_prices(stock_df, predict_days)

            if prediction_data is None:
                st.warning("Not enough data for prediction")

            else:

                prediction_data['Predicted_INR'] = prediction_data['Predicted_Close'] * conversion_rate

                fig4 = go.Figure()

                fig4.add_trace(go.Scatter(
                    x=stock_df['Date'],
                    y=stock_df['Close_INR'],
                    mode='lines',
                    name='Historical Price'
                ))

                fig4.add_trace(go.Scatter(
                    x=prediction_data['Date'],
                    y=prediction_data['Predicted_INR'],
                    mode='lines+markers',
                    name='Predicted Price'
                ))

                st.plotly_chart(fig4, use_container_width=True)

                final_predicted_price = prediction_data['Predicted_INR'].iloc[-1]

                st.info(f"Predicted price in {predict_days} days: ₹{final_predicted_price:.2f}")