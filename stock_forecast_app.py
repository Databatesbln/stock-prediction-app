#import necessary packages
import streamlit as st
from datetime import date
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

#Define Date Range
START = "2020-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

#App Title
st.title("Stock Price Prediction")

#stocks we want to use - Apple, Alphabet, Microsoft, Amazon,Facebook
stocks =("AAPL", "GOOG","MSFT", "AMZN", "META")

#create selection option
selected_stocks = st.selectbox("Please select Stock for Prediction", stocks)

#create slider to predict up to 4 years out in days
n_years = st.slider("Years for Predicion", 1 , 4)
period = n_years * 365

#load & cache stock data
@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

data_load_state = st.text("Load data...")
data = load_data(selected_stocks)
data_load_state.text("Loading data...done!")

#display subheader 'data" and display the last few rows
st.subheader('Data')
st.write(data.tail())

# Define a function to plot raw data of stock prices
def plot_raw_data(): 
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name='stock_open'))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='stock_close'))
    fig.layout.update(title_text="Time Series Data", xaxis_rangeslider_visible =True)
    st.plotly_chart(fig)

# Call the function to plot the data
plot_raw_data()

# Prepare the data for forecasting by selecting and renaming columns
df_train = data[['Date', 'Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

# Initialize the Prophet model, fit it with training data and create future dates for predictions
m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods = period)

# Use the model to make predictions for the specified future dates
forecast = m.predict(future)

# Display the subheader 'Forecast Data' in the app and display the last few rows of forecast
st.subheader('Forecast Data')
st.write(forecast.tail())

#Display title "Forecast", plot forecast and show
st.write('Forecast')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

#Plot Forecast Components
st.write('Forecast Components')
fig2 = m.plot_components(forecast)
st.write(fig2)

