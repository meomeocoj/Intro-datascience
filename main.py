import streamlit as st 
import matplotlib.pyplot as plt 
import yfinance as yf
import datetime

start = "2010-01-01"
end = "2022-12-31"

st.title('Stock Trend Prediction')

user_input = st.text_input("Enter Stock Ticker:", "AAPL")
df = yf.download(user_input, start, end)

# Describing Data
st.subheader('Data from 2010 - 2022')
st.write(df.describe())
