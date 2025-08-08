import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


# Markowitz

def portfolio_variance(weights, cov_matrix_window):
    return weights.T @ cov_matrix_window @ weights

def markowitzloop(returns, window_size = 60, rebalance_every = 20):
    
    weights_over_time = []
    dates = []  

    for i in range(0, len(returns) - window_size, rebalance_every):
        window_returns = returns.iloc[i:i + window_size] * 100

        expected_returns_window = window_returns.mean()
        cov_matrix_window = window_returns.cov()

        # Check variance for different initial guesses
        num_assets = len(expected_returns_window)
        init_guess = np.ones(num_assets) / num_assets

        print("Variance at init_guess:", portfolio_variance(init_guess, cov_matrix_window))

        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        bounds = tuple((0, 1) for _ in range(num_assets))

        opt_result = minimize(portfolio_variance,
                              init_guess,
                              args=(cov_matrix_window,),
                              method='SLSQP',
                              bounds=bounds,
                              constraints=constraints)

        print("Optimized weights:", opt_result.x)
        print("Optimized variance:", portfolio_variance(opt_result.x, cov_matrix_window))

        weights_over_time.append(opt_result.x)
        dates.append(returns.index[i + window_size])




# LSTM + Markowitz

def prepare_multivariate_data(data, window_size=60):
    scalers = {}
    scaled_data = pd.DataFrame(index=data.index)

    for col in data.columns:
        scaler = MinMaxScaler()
        scaled_data[col] = scaler.fit_transform(data[[col]])
        scalers[col] = scaler

    X, y = [], []
    for i in range(window_size, len(scaled_data)):
        X.append(scaled_data.iloc[i-window_size:i].values)  
        y.append(scaled_data.iloc[i].values)                
    
    return np.array(X), np.array(y), scalers, scaled_data

# Train LSTM model
def build_multivariate_lstm(n_assets, window_size=60):
    model = Sequential([
        LSTM(64, return_sequences=False, input_shape=(window_size, n_assets)),
        Dense(n_assets)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def predict_on_existing_data(model, X, scaled_data, scalers, original_data, window_size=60):
    predicted_scaled = []

    for i in range(window_size, len(scaled_data)):
        input_seq = X[i - window_size:i - window_size + 1]  
        pred = model.predict(input_seq, verbose=0)[0]       
        predicted_scaled.append(pred)

    pred_df = pd.DataFrame(predicted_scaled, columns=scaled_data.columns)
    
    for col in pred_df.columns:
        pred_df[col] = scalers[col].inverse_transform(pred_df[[col]])

    pred_df.index = original_data.index[window_size:]
    pred_df = pd.concat([original_data.iloc[0:window_size], pred_df], axis=0).dropna()
    return pred_df