# %%
import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from datetime import timedelta

# --- Sidebar Inputs ---
st.sidebar.title("📊 Stock Indicator Dashboard")
ticker = st.sidebar.text_input("Enter stock ticker", value="AAPL").upper()
start_date = st.sidebar.date_input("Start date", pd.to_datetime("2022-01-01"))
end_date = st.sidebar.date_input("End date", pd.to_datetime("today"))

# --- Adjusted Start Date (for indicator calculation buffer) ---
buffer_days = 300
adjusted_start = start_date - timedelta(days=buffer_days)
adjusted_end = end_date + timedelta(days=1)

# --- Load Data ---
@st.cache_data
def load_data(ticker, start, end):
    data = yf.download(ticker, start=start, end=end)
    # Flatten multi-index columns (e.g., ('Volume', 'AAPL') → 'Volume')
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data.index = pd.to_datetime(data.index).tz_localize(None)

    # Indicators
    data['SMA50'] = data['Close'].rolling(window=50).mean()
    data['EMA200'] = data['Close'].ewm(span=200, adjust=False).mean()

    # RSI
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp12 = data['Close'].ewm(span=12, adjust=False).mean()
    exp26 = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = exp12 - exp26
    data['MACD_Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
    data['MACD_Hist'] = data['MACD'] - data['MACD_Signal']

    # Crossover Signals
    data['Signal'] = (data['SMA50'] > data['EMA200']).astype(int)
    data['Crossover'] = data['Signal'].diff()

    return data

# --- Load and validate data ---
data = load_data(ticker, adjusted_start, adjusted_end)

if data.empty or 'Close' not in data.columns:
    st.error("No data returned. Check the ticker or date range.")
    st.stop()

# --- Trim to user-selected range ---
data_display = data[data.index >= pd.to_datetime(start_date)].copy()

# --- Remove rows missing critical indicators ---
#required_cols = ['SMA50', 'EMA200', 'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist']
#existing_cols = [col for col in required_cols if col in data_display.columns]
#data_display.dropna(subset=existing_cols, inplace=True)

# --- Remove rows with missing Volume if present ---
#if 'Volume' in data_display.columns:
#    data_display = data_display[data_display['Volume'].notnull()]

# --- Buy/Sell markers ---
buy_signals = data_display[data_display['Crossover'] == 1]
sell_signals = data_display[data_display['Crossover'] == -1]

# Ensure indicator columns exist before dropping NaNs
expected_cols = ['Close', 'SMA50', 'EMA200', 'Volume']
safe_cols = [col for col in expected_cols if col in data_display.columns]

#st.write("NaN counts in data_display:")
#st.write(data_display[['Close', 'SMA50', 'EMA200', 'Volume']].isna().sum())

# Now drop rows where any of these are missing — only if they're all present
if len(safe_cols) == len(expected_cols):
    plot_data = data_display.dropna(subset=safe_cols).copy()
else:
    st.warning("Some required columns are missing from the data. Unable to plot indicators.")
    st.stop()

# --- Prepare plot data (remove rows with NaNs just for the price chart) ---
plot_data = data_display.dropna(subset=['Close', 'SMA50', 'EMA200', 'Volume'])

# --- Price + Volume Subplot ---
fig = make_subplots(
    rows=2, cols=1, shared_xaxes=True,
    row_heights=[0.7, 0.3], vertical_spacing=0.05,
    subplot_titles=(f"{ticker} Price with SMA/EMA and Signals", f"{ticker} Daily Volume")
)

# --- Price traces ---
fig.add_trace(go.Scatter(x=plot_data.index, y=plot_data['Close'], name='Close', line=dict(color='gray')), row=1, col=1)
fig.add_trace(go.Scatter(x=plot_data.index, y=plot_data['SMA50'], name='SMA 50', line=dict(color='blue', dash='dot')), row=1, col=1)
fig.add_trace(go.Scatter(x=plot_data.index, y=plot_data['EMA200'], name='EMA 200', line=dict(color='orange', dash='dash')), row=1, col=1)
fig.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals['Close'], mode='markers',
                         marker=dict(symbol='triangle-up', color='green', size=10), name='Buy Signal'), row=1, col=1)
fig.add_trace(go.Scatter(x=sell_signals.index, y=sell_signals['Close'], mode='markers',
                         marker=dict(symbol='triangle-down', color='red', size=10), name='Sell Signal'), row=1, col=1)

# --- Volume ---
fig.add_trace(go.Bar(x=plot_data.index, y=plot_data['Volume'], name='Volume', marker_color='lightblue'), row=2, col=1)

# --- Layout ---
fig.update_layout(
    height=600,
    showlegend=True,
    margin=dict(t=40, b=40),
    xaxis=dict(rangeslider=dict(visible=False))
)

st.plotly_chart(fig, use_container_width=True)

# --- RSI Chart ---
fig_rsi = go.Figure()
fig_rsi.add_trace(go.Scatter(x=data_display.index, y=data_display['RSI'], name='RSI', line=dict(color='green')))
fig_rsi.add_shape(type="line", x0=data_display.index[0], x1=data_display.index[-1], y0=70, y1=70, line=dict(color='red', dash='dash'))
fig_rsi.add_shape(type="line", x0=data_display.index[0], x1=data_display.index[-1], y0=30, y1=30, line=dict(color='blue', dash='dash'))
fig_rsi.update_layout(title=f"{ticker} RSI (14-day)", xaxis_title="Date", yaxis_title="RSI", height=300, margin=dict(t=40, b=20))
st.plotly_chart(fig_rsi, use_container_width=True)

# --- MACD Chart ---
fig_macd = go.Figure()
fig_macd.add_trace(go.Scatter(x=data_display.index, y=data_display['MACD'], name='MACD Line', line=dict(color='purple')))
fig_macd.add_trace(go.Scatter(x=data_display.index, y=data_display['MACD_Signal'], name='Signal Line', line=dict(color='orange')))
fig_macd.add_trace(go.Bar(x=data_display.index, y=data_display['MACD_Hist'], name='Histogram', marker=dict(color='gray')))
fig_macd.update_layout(title=f"{ticker} MACD", xaxis_title="Date", yaxis_title="MACD", height=300, margin=dict(t=40, b=20))
st.plotly_chart(fig_macd, use_container_width=True)

# %%
