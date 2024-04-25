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

#analyze and plot last 5 days of stock data

st.subheader('Data')
st.write(data.tail())

def plot_raw_data(): 
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name='stock_open'))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='stock_close'))
    fig.layout.update(title_text="Time Series Data", xaxis_rangeslider_visible =True)
    st.plotly_chart(fig)

plot_raw_data()

#Forecasting - change
df_train = data[['Date', 'Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods = period)
forecast = m.predict(future)

st.subheader('Forecast Data')
st.write(forecast.tail())

st.write('Forecast')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

st.write('Forecast Components')
fig2 = m.plot_components(forecast)
st.write(fig2)

