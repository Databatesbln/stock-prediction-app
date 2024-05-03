# Import necessary packages
import streamlit as st
from datetime import date
import yfinance as yf
from prophet import Prophet
import plotly.graph_objs as go
from plotly import express as px
import matplotlib.pyplot as plt  # Import for Matplotlib

# Define the date range
START = "2020-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

# App title
st.title("📈 Stock Price Prediction")

# Description
st.markdown("""
This app predicts the stock price using historical data. You can select from major technology stocks and forecast up to 4 years into the future.
""")

# List of stocks
stocks = ("AAPL", "GOOG", "MSFT", "AMZN", "META")
# Create a selection option
selected_stock = st.selectbox("Select a stock for prediction", stocks)

# Create a slider for prediction years
n_years = st.slider("Select the number of years for the prediction:", 1, 4, 1)
period = n_years * 365  # days

# Load & cache stock data
@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

# Load state
data_load_state = st.text("Loading data...")
data = load_data(selected_stock)
data_load_state.text("Data loaded successfully!")

# Display the latest stock data
st.subheader('Recent Data Snapshot')
st.write(data.tail(5))

# Plotting the raw stock data
def plot_raw_data():
    fig = px.line(data, x='Date', y=['Open', 'Close'], labels={'value': 'Stock Price ($)', 'variable': 'Price Type'}, title="Stock Opening and Closing Prices")
    fig.update_xaxes(rangeslider_visible=True)
    st.plotly_chart(fig, use_container_width=True)

plot_raw_data()

# Forecasting
df_train = data[['Date', 'Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

prophet_model = Prophet()
prophet_model.fit(df_train)
future = prophet_model.make_future_dataframe(periods=period)
forecast = prophet_model.predict(future)

# Rename forecast columns for user-friendliness
forecast.rename(columns={'yhat': 'Predicted Value', 'yhat_lower': 'Lower Bound', 'yhat_upper': 'Upper Bound'}, inplace=True)

# Display forecast data with renamed columns
st.subheader('Forecast Data Overview')
st.write(forecast[['ds', 'Predicted Value', 'Lower Bound', 'Upper Bound']].tail())



# Display forecast plot
st.subheader('Forecast Chart')
fig1 = px.line(forecast, x='ds', y='Predicted Value', labels={'ds': 'Date', 'Predicted Value': 'Forecasted Price'}, title="Forecasted Stock Prices")
fig1.add_trace(go.Scatter(x=forecast['ds'], y=forecast['Lower Bound'], mode='lines', line=dict(color='gray'), name='Lower Bound'))
fig1.add_trace(go.Scatter(x=forecast['ds'], y=forecast['Upper Bound'], mode='lines', line=dict(color='gray'), name='Upper Bound', fill='tonexty'))

# Specify date format for hover
fig1.update_layout(hovermode="x unified", hoverlabel=dict(bgcolor="white", font_size=12),
                   xaxis=dict(
                       showspikes=True,
                       spikethickness=1,
                       spikedash="dot",
                       spikecolor="#999999",
                       spikesnap="cursor",
                       tickformat="%Y-%m-%d"
                   ))

fig1.update_traces(hoverinfo='text', hovertemplate='<b>Date</b>: %{x} <br><b>Price</b>: $%{y:.2f}')
st.plotly_chart(fig1, use_container_width=True)



# Display forecast components using Matplotlib
st.subheader('Forecast Components')
fig2 = prophet_model.plot_components(forecast)
st.pyplot(fig2)
