import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import torch.nn as nn
import tensorflow as tf
import torch
import random
import os


def set_seeds(seed=42):
    """
    Set seeds for all random number generators to ensure reproducible results.
    
    Args:
        seed (int): Seed value for reproducibility. Default: 42
    """
    print(f"Setting random seeds to {seed} for reproducible results...")
    
    # Python built-in random
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # TensorFlow
    tf.random.set_seed(seed)
    
    # Configure TensorFlow for deterministic operations
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    
    # PyTorch (used by Stable Baselines3)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # PyTorch deterministic mode
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set environment variables for additional reproducibility
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    print(f"✓ All random seeds set to {seed}")


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

def add_technical_indicators(prices_df):
    """
    Add technical indicators to enhance LSTM features.
    """
    data = prices_df.copy()
    
    for col in prices_df.columns:
        # Simple moving averages
        data[f'{col}_MA5'] = prices_df[col].rolling(window=5).mean()
        data[f'{col}_MA10'] = prices_df[col].rolling(window=10).mean()
        data[f'{col}_MA20'] = prices_df[col].rolling(window=20).mean()
        
        # Exponential moving averages
        data[f'{col}_EMA12'] = prices_df[col].ewm(span=12).mean()
        data[f'{col}_EMA26'] = prices_df[col].ewm(span=26).mean()
        
        # MACD
        data[f'{col}_MACD'] = data[f'{col}_EMA12'] - data[f'{col}_EMA26']
        data[f'{col}_MACD_Signal'] = data[f'{col}_MACD'].ewm(span=9).mean()
        
        # Rolling volatility
        returns = prices_df[col].pct_change()
        data[f'{col}_Volatility'] = returns.rolling(window=20).std()
        
        # Price momentum
        data[f'{col}_Momentum'] = prices_df[col] / prices_df[col].shift(10) - 1
        
        # Relative Strength Index (RSI)
        delta = prices_df[col].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data[f'{col}_RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        rolling_mean = prices_df[col].rolling(window=20).mean()
        rolling_std = prices_df[col].rolling(window=20).std()
        data[f'{col}_BB_Upper'] = rolling_mean + (rolling_std * 2)
        data[f'{col}_BB_Lower'] = rolling_mean - (rolling_std * 2)
        data[f'{col}_BB_Position'] = (prices_df[col] - data[f'{col}_BB_Lower']) / (data[f'{col}_BB_Upper'] - data[f'{col}_BB_Lower'])
    
    # Remove NaN values
    data = data.bfill().ffill()
    
    return data


def prepare_multivariate_data(data, window_size=60, use_features=True):
    """
    Prepare data for LSTM training with optional feature engineering.
    """
    if use_features:
        # Add technical indicators
        enhanced_data = add_technical_indicators(data)
        print(f"Enhanced features: {enhanced_data.shape[1]} columns (from {data.shape[1]} original)")
    else:
        enhanced_data = data.copy()
    
    # Use StandardScaler instead of MinMaxScaler for better performance
    scalers = {}
    scaled_data = pd.DataFrame(index=enhanced_data.index)
    
    for col in enhanced_data.columns:
        scaler = StandardScaler()
        scaled_data[col] = scaler.fit_transform(enhanced_data[[col]])
        scalers[col] = scaler
    
    # Create sequences for LSTM
    X, y = [], []
    target_cols = [col for col in data.columns]  # Only predict original asset prices
    
    for i in range(window_size, len(scaled_data)):
        # Use all features for X (input)
        X.append(scaled_data.iloc[i-window_size:i].values)  
        # Only predict original asset prices for y (output)
        y.append(scaled_data[target_cols].iloc[i].values)
    
    return np.array(X), np.array(y), scalers, scaled_data, target_cols

# Enhanced LSTM model
def build_enhanced_lstm(n_features, n_assets, window_size=60):
    """
    Build enhanced LSTM model with dropout, regularization, and multiple layers.
    """
    model = Sequential([
        # First LSTM layer with dropout
        LSTM(128, return_sequences=True, input_shape=(window_size, n_features),
             dropout=0.2, recurrent_dropout=0.2, 
             kernel_regularizer=l1_l2(l1=0.01, l2=0.01)),
        BatchNormalization(),
        
        # Second LSTM layer
        LSTM(64, return_sequences=True,
             dropout=0.2, recurrent_dropout=0.2,
             kernel_regularizer=l1_l2(l1=0.01, l2=0.01)),
        BatchNormalization(),
        
        # Third LSTM layer
        LSTM(32, return_sequences=False,
             dropout=0.2, recurrent_dropout=0.2,
             kernel_regularizer=l1_l2(l1=0.01, l2=0.01)),
        BatchNormalization(),
        
        # Dense layers with dropout
        Dense(64, activation='relu', kernel_regularizer=l1_l2(l1=0.01, l2=0.01)),
        Dropout(0.3),
        Dense(32, activation='relu', kernel_regularizer=l1_l2(l1=0.01, l2=0.01)),
        Dropout(0.2),
        Dense(n_assets, activation='linear')
    ])
    
    # Use adaptive learning rate and better optimizer
    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )
    
    return model


def build_multivariate_lstm(n_assets, window_size=60):
    """
    Legacy function for backward compatibility.
    """
    return build_enhanced_lstm(n_assets, n_assets, window_size)

def predict_on_existing_data(model, X, scaled_data, scalers, original_data, target_cols, window_size=60):
    """
    Enhanced prediction function that works with feature-rich data.
    """
    predicted_scaled = []
    
    # Use the prepared X data directly
    predictions = model.predict(X, verbose=0)
    
    # Convert predictions back to DataFrame
    pred_df = pd.DataFrame(predictions, columns=target_cols)
    
    # Inverse transform only the target columns (original assets)
    for col in target_cols:
        if col in scalers:
            pred_df[col] = scalers[col].inverse_transform(pred_df[[col]]).flatten()
    
    # Set proper index
    pred_df.index = original_data.index[window_size:window_size + len(pred_df)]
    
    # Concatenate with original data to maintain full timeline
    full_pred_df = pd.concat([original_data.iloc[:window_size], pred_df], axis=0)
    
    # Only return columns that exist in original data
    return full_pred_df[original_data.columns].dropna()


def train_enhanced_lstm(prices_df, window_size=60, epochs=100, validation_split=0.2, use_features=True, seed=42):
    """
    Train LSTM model with enhanced features and proper validation.
    
    Args:
        seed (int): Random seed for reproducible training
    """
    print("Preparing enhanced LSTM training data...")
    
    # Ensure consistent seeding for this training session
    set_seeds(seed)
    
    # Prepare data with features
    X, y, scalers, scaled_data, target_cols = prepare_multivariate_data(
        prices_df, window_size=window_size, use_features=use_features
    )
    
    print(f"Training data shape: X={X.shape}, y={y.shape}")
    
    # Build enhanced model
    n_features = X.shape[2]
    n_assets = y.shape[1]
    model = build_enhanced_lstm(n_features, n_assets, window_size)
    
    print(f"Model architecture: {n_features} features -> {n_assets} assets")
    
    # Set up callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=10,
        min_lr=1e-6,
        verbose=1
    )
    
    # Train model
    print("Training enhanced LSTM model...")
    history = model.fit(
        X, y,
        epochs=epochs,
        batch_size=32,
        validation_split=validation_split,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    
    return model, X, scalers, scaled_data, target_cols, history

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
        
        # Calculate prices from returns for technical indicators
        self.prices = (1 + self.returns).cumprod() * 100  # Start at 100
        
        # Add technical indicators to observation space
        self.enhanced_features = self._prepare_enhanced_features()
        
        self.window_size = window_size
        self.n_assets = len(self.returns.columns)
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        
        # Action space: portfolio weights (continuous, sum to 1)
        self.action_space = spaces.Box(
            low=0, high=1, shape=(self.n_assets,), dtype=np.float32
        )
        
        # Enhanced observation space: rolling window of features
        n_features = self.enhanced_features.shape[1]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(window_size, n_features), 
            dtype=np.float32
        )
        
        self.reset()
    
    def _prepare_enhanced_features(self):
        """
        Prepare enhanced features including technical indicators for RL.
        """
        # Start with returns
        features = self.returns.copy()
        
        # Add price-based features
        for col in self.prices.columns:
            # Moving averages ratios
            ma5 = self.prices[col].rolling(5).mean()
            ma10 = self.prices[col].rolling(10).mean()
            ma20 = self.prices[col].rolling(20).mean()
            
            features[f'{col}_MA5_ratio'] = self.prices[col] / ma5 - 1
            features[f'{col}_MA10_ratio'] = self.prices[col] / ma10 - 1
            features[f'{col}_MA20_ratio'] = self.prices[col] / ma20 - 1
            
            # Volatility
            features[f'{col}_volatility'] = self.returns[col].rolling(20).std()
            
            # Momentum
            features[f'{col}_momentum'] = self.prices[col] / self.prices[col].shift(10) - 1
            
            # RSI
            delta = self.prices[col].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            features[f'{col}_rsi'] = (100 - (100 / (1 + rs))) / 100 - 0.5  # Normalize to [-0.5, 0.5]
        
        # Fill NaN values
        features = features.fillna(0)
        
        # Normalize features using StandardScaler
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        features_scaled = pd.DataFrame(
            scaler.fit_transform(features),
            index=features.index,
            columns=features.columns
        )
        
        return features_scaled
    
    def reset(self, seed=None):
        super().reset(seed=seed)
        if seed is not None:
            # Ensure consistent seeding within the environment
            np.random.seed(seed)
        
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
        # Get the last window_size days of enhanced features
        start_idx = self.current_step - self.window_size
        end_idx = self.current_step
        window_features = self.enhanced_features.iloc[start_idx:end_idx].values
        
        # Replace any NaN or inf values with 0
        window_features = np.nan_to_num(window_features, nan=0.0, posinf=0.0, neginf=0.0)
        
        return window_features.astype(np.float32)
    
    def _calculate_reward(self, returns, weights):
        """
        Enhanced reward function combining multiple risk-adjusted metrics.
        """
        # Calculate portfolio return
        portfolio_return = np.sum(returns * weights)
        
        # Base reward from return
        return_reward = portfolio_return * 100
        
        # Risk-adjusted rewards
        risk_penalty = 0
        sharpe_bonus = 0
        diversification_bonus = 0
        drawdown_penalty = 0
        
        if len(self.portfolio_values) >= 20:
            # Calculate recent portfolio returns
            recent_values = self.portfolio_values[-20:]
            recent_returns = np.diff(recent_values) / recent_values[:-1]
            
            # Sharpe ratio bonus
            if np.std(recent_returns) > 0:
                sharpe_ratio = np.mean(recent_returns) / np.std(recent_returns)
                sharpe_bonus = np.clip(sharpe_ratio * 0.5, -2, 2)
            
            # Volatility penalty
            volatility = np.std(recent_returns)
            risk_penalty = -volatility * 10
            
            # Drawdown penalty
            peak = np.maximum.accumulate(recent_values)
            drawdown = (recent_values - peak) / peak
            max_drawdown = np.min(drawdown)
            if max_drawdown < -0.05:  # Penalty for drawdowns > 5%
                drawdown_penalty = max_drawdown * 5
        
        # Diversification bonus (encourage balanced portfolios)
        weight_entropy = -np.sum(weights * np.log(weights + 1e-8))
        max_entropy = np.log(self.n_assets)
        diversification_bonus = (weight_entropy / max_entropy) * 0.2
        
        # Turnover penalty (discourage excessive trading)
        if hasattr(self, 'prev_weights'):
            turnover = np.sum(np.abs(weights - self.prev_weights))
            turnover_penalty = -turnover * 0.5
        else:
            turnover_penalty = 0
        
        self.prev_weights = weights.copy()
        
        # Combined reward
        total_reward = (return_reward + 
                       sharpe_bonus + 
                       risk_penalty + 
                       diversification_bonus + 
                       drawdown_penalty + 
                       turnover_penalty)
        
        return total_reward
    
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


def train_enhanced_rl_model(returns, window_size=60, total_timesteps=100000, learning_rate=1e-4, seed=42):
    """
    Train enhanced RL model with optimized hyperparameters.
    
    Args:
        seed (int): Random seed for reproducible training
    """
    # Ensure consistent seeding for this training session
    set_seeds(seed)
    
    # Clean returns data
    returns_clean = returns.dropna()
    returns_clean = returns_clean.fillna(0)
    
    # Check if we have enough data
    if len(returns_clean) < window_size + 10:
        print(f"Warning: Very limited data ({len(returns_clean)} points). Consider using more data or smaller window_size.")
    
    print(f"Training enhanced RL model with {len(returns_clean)} data points, {len(returns_clean.columns)} assets")
    
    # Create environment
    env = PortfolioEnv(returns_clean, window_size=window_size)
    env = DummyVecEnv([lambda: env])
    
    print(f"Observation space shape: {env.get_attr('observation_space')[0].shape}")
    print(f"Action space shape: {env.get_attr('action_space')[0].shape}")
    
    # Create PPO model with enhanced hyperparameters
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        n_steps=4096,  # Increased for more stable learning
        batch_size=128,  # Larger batch size
        n_epochs=20,  # More epochs for better learning
        gamma=0.995,  # Higher gamma for long-term rewards
        gae_lambda=0.98,  # Higher GAE lambda
        clip_range=0.15,  # Slightly tighter clipping
        ent_coef=0.01,  # Entropy coefficient for exploration
        vf_coef=0.5,  # Value function coefficient
        max_grad_norm=0.5,  # Gradient clipping
        verbose=1,
        seed=seed,  # Set seed for reproducible RL training
        tensorboard_log="./ppo_portfolio_tensorboard/",
        policy_kwargs=dict(
            net_arch=dict(pi=[256, 128, 64], vf=[256, 128, 64]),  # Larger networks
            activation_fn=nn.Tanh
        )
    )
    
    # Train the model
    print("Starting enhanced RL training...")
    model.learn(total_timesteps=total_timesteps)
    
    return model, env


def train_rl_model(returns, window_size=60, total_timesteps=50000, learning_rate=3e-4):
    """
    Legacy function - now uses enhanced version.
    """
    return train_enhanced_rl_model(returns, window_size, total_timesteps, learning_rate)


def predict_rl_weights(model, returns, window_size=60, rebalance_every=20):
    weights_over_time = []
    dates = []
    
    # Clean returns data
    returns_clean = returns.dropna()
    returns_clean = returns_clean.fillna(0)
    
    # Create environment for prediction (this will generate the enhanced features)
    env = PortfolioEnv(returns_clean, window_size=window_size)
    
    # Reset environment
    obs, _ = env.reset()
    
    # Generate predictions using the environment's observation method
    for i in range(window_size, len(returns_clean) - 1, rebalance_every):
        # Set environment to the correct step
        env.current_step = i
        
        # Get enhanced observation from environment
        obs = env._get_observation()
        
        # Predict action (weights)
        action, _ = model.predict(obs, deterministic=True)
        
        # Handle NaN or invalid actions
        action = np.nan_to_num(action, nan=1.0/len(action))
        
        # Ensure positive values
        action = np.abs(action)
        
        # Handle case where all actions are zero
        action_sum = np.sum(action)
        if action_sum == 0 or not np.isfinite(action_sum):
            # Equal weights fallback
            weights = np.ones(len(action)) / len(action)
        else:
            # Normalize weights
            weights = action / action_sum
        
        # Final safety checks
        weights = np.clip(weights, 0, 1)
        weights = weights / np.sum(weights)
        
        # Ensure no NaN values
        weights = np.nan_to_num(weights, nan=1.0/len(weights))
        
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