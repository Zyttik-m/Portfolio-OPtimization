import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv


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
    
    return weights_over_time, dates




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

# Reinforcement Learning - PPO

class PortfolioEnv(gym.Env):
    def __init__(self, returns_data, window_size=60, initial_balance=1000000, transaction_cost=0.001):
        super(PortfolioEnv, self).__init__()
        
        # Clean the returns data - remove NaN values and fill any remaining with 0
        self.returns = returns_data.dropna()
        self.returns = self.returns.fillna(0)  # Fill any remaining NaNs with 0
        
        # Ensure we have enough data
        if len(self.returns) < window_size + 1:
            raise ValueError(f"Not enough data points. Need at least {window_size + 1}, got {len(self.returns)}")
        
        self.window_size = window_size
        self.n_assets = len(self.returns.columns)
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        
        # Action space: portfolio weights (continuous, sum to 1)
        self.action_space = spaces.Box(
            low=0, high=1, shape=(self.n_assets,), dtype=np.float32
        )
        
        # Observation space: rolling window of returns
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(window_size, self.n_assets), 
            dtype=np.float32
        )
        
        self.reset()
    
    def reset(self, seed=None):
        super().reset(seed=seed)
        self.current_step = self.window_size
        self.balance = self.initial_balance
        self.portfolio_value = self.initial_balance
        self.weights = np.array([1.0/self.n_assets] * self.n_assets)
        self.portfolio_values = [self.initial_balance]
        self.weights_history = [self.weights.copy()]
        
        # Return initial observation
        obs = self._get_observation()
        return obs, {}
    
    def _get_observation(self):
        # Get the last window_size days of returns
        start_idx = self.current_step - self.window_size
        end_idx = self.current_step
        window_returns = self.returns.iloc[start_idx:end_idx].values
        
        # Replace any NaN or inf values with 0
        window_returns = np.nan_to_num(window_returns, nan=0.0, posinf=0.0, neginf=0.0)
        
        return window_returns.astype(np.float32)
    
    def _calculate_reward(self, returns, weights):
        # Calculate portfolio return
        portfolio_return = np.sum(returns * weights)
        
        # Calculate Sharpe ratio as reward (simplified)
        if len(self.portfolio_values) > 1:
            recent_returns = np.diff(self.portfolio_values[-min(20, len(self.portfolio_values)):]) / self.portfolio_values[-min(20, len(self.portfolio_values)):-1]
            if len(recent_returns) > 0 and np.std(recent_returns) > 0:
                sharpe = np.mean(recent_returns) / np.std(recent_returns)
            else:
                sharpe = 0
        else:
            sharpe = 0
        
        # Combine return and Sharpe ratio
        reward = portfolio_return * 100 + sharpe * 0.1
        
        return reward
    
    def step(self, action):
        # Handle edge case where action sum is 0 or contains NaN
        action = np.nan_to_num(action, nan=1.0/self.n_assets)
        
        # Normalize weights to sum to 1
        action_sum = np.sum(action)
        if action_sum == 0 or np.isnan(action_sum):
            weights = np.ones(self.n_assets) / self.n_assets  # Equal weights as fallback
        else:
            weights = action / action_sum
        
        weights = np.clip(weights, 0, 1)
        weights = weights / np.sum(weights)
        
        # Get current returns
        if self.current_step < len(self.returns):
            current_returns = self.returns.iloc[self.current_step].values
        else:
            # Episode ends
            truncated = True
            obs = self._get_observation()
            reward = 0
            return obs, reward, False, truncated, {}
        
        # Calculate transaction costs
        weight_change = np.abs(weights - self.weights)
        transaction_costs = np.sum(weight_change) * self.transaction_cost
        
        # Update portfolio value
        portfolio_return = np.sum(current_returns * weights)
        self.portfolio_value *= (1 + portfolio_return - transaction_costs)
        self.portfolio_values.append(self.portfolio_value)
        
        # Update weights
        self.weights = weights.copy()
        self.weights_history.append(self.weights.copy())
        
        # Calculate reward
        reward = self._calculate_reward(current_returns, weights)
        
        # Move to next step
        self.current_step += 1
        
        # Check if episode is done
        done = self.current_step >= len(self.returns) - 1
        truncated = done
        
        # Get next observation
        if not done:
            obs = self._get_observation()
        else:
            obs = self._get_observation()
        
        info = {
            'portfolio_value': self.portfolio_value,
            'weights': self.weights.copy()
        }
        
        return obs, reward, False, truncated, info


def train_rl_model(returns, window_size=60, total_timesteps=50000, learning_rate=3e-4):
    # Clean returns data
    returns_clean = returns.dropna()
    returns_clean = returns_clean.fillna(0)
    
    # Check if we have enough data
    if len(returns_clean) < window_size + 10:
        print(f"Warning: Very limited data ({len(returns_clean)} points). Consider using more data or smaller window_size.")
    
    print(f"Training RL model with {len(returns_clean)} data points, {len(returns_clean.columns)} assets")
    
    # Create environment
    env = PortfolioEnv(returns_clean, window_size=window_size)
    env = DummyVecEnv([lambda: env])
    
    # Create PPO model
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=1,
        tensorboard_log="./ppo_portfolio_tensorboard/"
    )
    
    # Train the model
    model.learn(total_timesteps=total_timesteps)
    
    return model, env


def predict_rl_weights(model, returns, window_size=60, rebalance_every=20):
    weights_over_time = []
    dates = []
    
    # Clean returns data
    returns_clean = returns.dropna()
    returns_clean = returns_clean.fillna(0)
    
    # Create environment for prediction
    env = PortfolioEnv(returns_clean, window_size=window_size)
    
    # Reset environment
    obs, _ = env.reset()
    
    # Generate predictions
    for i in range(window_size, len(returns_clean) - 1, rebalance_every):
        # Get observation
        start_idx = i - window_size
        end_idx = i
        window_returns = returns_clean.iloc[start_idx:end_idx].values
        
        # Clean the observation
        window_returns = np.nan_to_num(window_returns, nan=0.0, posinf=0.0, neginf=0.0)
        obs = window_returns.astype(np.float32)
        
        # Predict action (weights)
        action, _ = model.predict(obs, deterministic=True)
        
        # Normalize weights
        weights = action / np.sum(action)
        weights = np.clip(weights, 0, 1)
        weights = weights / np.sum(weights)
        
        weights_over_time.append(weights)
        dates.append(returns_clean.index[i])
        
        print(f"RL Weights at {returns_clean.index[i]}: {weights}")
    
    return weights_over_time, dates


def load_rl_model(model_path="ppo_portfolio_model"):
    try:
        model = PPO.load(model_path)
        print(f"Successfully loaded RL model from {model_path}")
        return model
    except:
        print(f"Could not load model from {model_path}")
        return None