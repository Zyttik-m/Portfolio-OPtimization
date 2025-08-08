# Import Libraries
import data_handler
import models
from argparse import ArgumentParser
import os
import pandas as pd




# Load Dataset

arg = ArgumentParser()

arg.add_argument("--tickers", type=str, nargs='+', default=['SPY', 'TLT', 'GLD'])
arg.add_argument("--start", type=str, default='2018-01-01')
arg.add_argument("--end", type=str, default='2025-01-01')
arg.add_argument("--interval", type=str, default='1d')
arg.add_argument("--auto_adjust", type=bool, default=False)
arg.add_argument("--window_size", type=int, default=60)
arg.add_argument("--rebalance_every", type=int, default=20)


args = arg.parse_args()

os.makedirs('data', exist_ok=True)

for name in args.tickers:
    data = data_handler.load_data(name, args.start, args.end, args.interval, args.auto_adjust)
    data.to_csv("data/" + name + '.csv')


# Data Preprocessing

datalist = []

path = './data'

for file in os.listdir(path):
    data = pd.read_csv(path + '/' + file)
    data = data_handler.cleandata(data, file[:-4])
    datalist.append(data)

prices = pd.concat(datalist, axis=1)
prices = prices.apply(pd.to_numeric)
returns = prices.pct_change()


# Model Training - Benchmark Model(Equal Weights)





# Model Training - Markowitz Model

markowitz_weights = models.markowitzloop(returns, args.window_size , args.rebalance_every)


# Model Training - Markowitz with LSTM

X, y, scalers, scaled_data = models.prepare_multivariate_data(prices, args.window_size )
model = models.build_multivariate_lstm(n_assets=X.shape[2], window_size = args.window_size)
model.fit(X, y, epochs=20, batch_size=32)

pred_df = models.predict_on_existing_data(model, X, scaled_data, scalers, prices, args.window_size)
returns_pred = pred_df.pct_change()

lstm_weights = models.markowitzloop(returns_pred, args.window_size , args.rebalance_every)

# Model Training - Reinforcement Learning










# Model Comparison





