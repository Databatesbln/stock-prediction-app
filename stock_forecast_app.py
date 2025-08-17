# Import necessary packages
import streamlit as st
import pandas as pd
from datetime import date
import yfinance as yf
from prophet import Prophet
import plotly.express as px
import plotly.graph_objs as go
import matplotlib.pyplot as plt

st.set_page_config(page_title="📈 Stock Price Prediction", page_icon="📈", layout="wide")

# ------------------ Config ------------------
START = "2020-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

# ------------------ UI ------------------
st.title("📈 Stock Price Prediction")
st.markdown(
    """
This app predicts the stock price using historical data. 
Select one of the tech stocks and forecast up to 4 years.
"""
)

stocks = ("AAPL", "GOOG", "MSFT", "AMZN", "META")
selected_stock = st.selectbox("Select a stock for prediction", stocks)
n_years = st.slider("Select the number of years for the prediction:", 1, 4, 1)
period = n_years * 365  # days

# ------------------ Data ------------------
@st.cache_data
def load_data(ticker: str) -> pd.DataFrame:
    try:
        df = yf.download(ticker, START, TODAY, progress=False)
        
        # Handle MultiIndex columns if they exist
        if isinstance(df.columns, pd.MultiIndex):
            # Flatten MultiIndex columns - take the first level
            df.columns = df.columns.get_level_values(0)
        
        df = df.reset_index()
        
        # Ensure columns exist even if yf changes schema
        if "Date" not in df.columns and "Datetime" in df.columns:
            df = df.rename(columns={"Datetime": "Date"})
        
        # Enforce dtypes
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        
        # Handle numeric columns more robustly
        for col in ("Open", "Close", "High", "Low", "Volume"):
            if col in df.columns:
                # Ensure we're working with a Series and handle any edge cases
                series_data = df[col]
                if hasattr(series_data, 'squeeze'):
                    series_data = series_data.squeeze()
                df[col] = pd.to_numeric(series_data, errors="coerce")
        
        return df
        
    except Exception as e:
        st.error(f"Error loading data for {ticker}: {str(e)}")
        return pd.DataFrame()

st.text("Loading data...")
data = load_data(selected_stock)

if not data.empty:
    st.success("Data loaded successfully!")
else:
    st.error("Failed to load data. Please try again.")
    st.stop()

if data.empty or "Close" not in data.columns:
    st.error("No price data returned. Try a different ticker or date range.")
    st.stop()

# ------------------ Preview ------------------
st.subheader("Recent Data Snapshot")
st.write(data.tail(5))

# ------------------ Raw Plot (use long form) ------------------
def plot_raw_data(df: pd.DataFrame):
    plot_df = df[["Date", "Open", "Close"]].copy()
    plot_df = plot_df.dropna(subset=["Date"])
    # Drop rows where both Open and Close are NaN
    plot_df = plot_df.dropna(subset=["Open", "Close"], how="all")
    plot_df = plot_df.melt(
        id_vars="Date", var_name="Price Type", value_name="Stock Price ($)"
    )
    fig = px.line(
        plot_df,
        x="Date",
        y="Stock Price ($)",
        color="Price Type",
        title="Stock Opening and Closing Prices",
    )
    fig.update_xaxes(rangeslider_visible=True)
    st.plotly_chart(fig, use_container_width=True)

plot_raw_data(data)

# ------------------ Prepare for Prophet ------------------
df_train = data.loc[:, ["Date", "Close"]].copy()
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

# Ensure 1-D numeric y
if hasattr(df_train["y"], "ndim") and df_train["y"].ndim > 1:
    # Squeeze (handles cases where y became (n,1))
    df_train["y"] = df_train["y"].squeeze()

# Coerce types and clean
df_train["ds"] = pd.to_datetime(df_train["ds"], errors="coerce")

# Make sure datetimes are tz-naive (Prophet requirement)
if getattr(df_train["ds"].dt, "tz", None) is not None:
    df_train["ds"] = df_train["ds"].dt.tz_localize(None)

df_train["y"] = pd.to_numeric(df_train["y"], errors="coerce")
df_train = df_train.dropna(subset=["ds", "y"])

# Prophet needs at least 2 rows
if len(df_train) < 2:
    st.error("Not enough clean data to train a forecast (need at least 2 valid rows).")
    st.stop()

# ------------------ Forecast ------------------
prophet_model = Prophet()
prophet_model.fit(df_train)

future = prophet_model.make_future_dataframe(periods=period, freq="D", include_history=True)
forecast = prophet_model.predict(future)

# User-friendly columns
fc = forecast.rename(
    columns={"yhat": "Predicted Value", "yhat_lower": "Lower Bound", "yhat_upper": "Upper Bound", "ds": "Date"}
)

# ------------------ Forecast Table ------------------
st.subheader("Forecast Data Overview")
st.write(fc[["Date", "Predicted Value", "Lower Bound", "Upper Bound"]].tail())

# ------------------ Forecast Plot ------------------
st.subheader("Forecast Chart")
fig1 = px.line(
    fc,
    x="Date",
    y="Predicted Value",
    labels={"Date": "Date", "Predicted Value": "Forecasted Price"},
    title="Forecasted Stock Prices",
)

# Add uncertainty band
fig1.add_trace(
    go.Scatter(
        x=fc["Date"], y=fc["Lower Bound"],
        mode="lines", line=dict(width=0.5), name="Lower Bound"
    )
)
fig1.add_trace(
    go.Scatter(
        x=fc["Date"], y=fc["Upper Bound"],
        mode="lines", line=dict(width=0.5), name="Upper Bound", fill="tonexty"
    )
)

fig1.update_layout(
    hovermode="x unified",
    hoverlabel=dict(bgcolor="white", font_size=12),
    xaxis=dict(
        showspikes=True,
        spikethickness=1,
        spikedash="dot",
        spikecolor="#999999",
        spikesnap="cursor",
        tickformat="%Y-%m-%d",
        rangeslider=dict(visible=True),
    ),
)
fig1.update_traces(hoverinfo="text", hovertemplate="<b>Date</b>: %{x}<br><b>Price</b>: $%{y:.2f}")
st.plotly_chart(fig1, use_container_width=True)

# ------------------ Components ------------------
st.subheader("Forecast Components")
fig2 = prophet_model.plot_components(forecast)
st.pyplot(fig2)
