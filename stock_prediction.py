import streamlit as st
from datetime import date
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

# start of data
START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

# title of streamlit app
st.title("Stock Prediction App")

stocks = ("AAPL", "GME", "TSLA", "SHOP")

# create select box for selected stocks
selected_stocks = st.selectbox("Select dataset for prediction", stocks)

# slider to get years
n_years = st.slider("Years of prediction:", 1, 4)
period = n_years * 365

# function to load stock data


@st.cache  # keeps laoded data in memory
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)  # puts date in the first column
    return data


# load data in stream lit and let user know when data is being loaded and done loading
data_load_state = st.text("Load data ...")
data = load_data(selected_stocks)
data_load_state.text("Loading data... done!")

# Show preview of data
st.subheader("Raw data")
st.write(data.tail())

# function to plot data


def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'],
                  y=data["Open"], name="stock_open"))
    fig.add_trace(go.Scatter(x=data['Date'],
                  y=data["Close"], name="stock_close"))
    fig.layout.update(title_text="Time Series Data",
                      xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)


plot_raw_data()

# forecasting
df_train = data[["Date", "Close"]]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

# model
m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

st.subheader("Forecast data")
st.write(forecast.tail())

# plot forecast
st.write("forecast data")
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

st.write('forecast components')
fig2 = m.plot_components(forecast)
st.write(fig2)
