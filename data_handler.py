import pandas as pd
import yfinance as yf

def load_data(name,start='2018-01-01', end='2025-01-01', interval='1d', auto_adjust=False):
    data = yf.download(name, start=start, end=end, interval=interval, auto_adjust=auto_adjust)
    return data

def cleandata(data, name):
    data = data.iloc[2:].reset_index(drop=True)
    data['Date'] = pd.to_datetime(data['Price'])
    data = data.set_index('Date')
    data = data[['Adj Close']]
    data = data.rename(columns={'Adj Close': name})
    return data



