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


# Enhanced Markowitz with Mean-Variance Optimization

def shrinkage_covariance(returns, shrinkage_intensity=None):
    """
    Compute shrinkage covariance matrix using Ledoit-Wolf method.
    
    Args:
        returns: DataFrame of returns
        shrinkage_intensity: Float between 0-1, if None uses automatic estimation
    
    Returns:
        Shrunk covariance matrix
    """
    sample_cov = returns.cov().values
    n_samples, n_features = returns.shape
    
    if shrinkage_intensity is None:
        # Automatic shrinkage intensity estimation (simplified Ledoit-Wolf)
        # This is a simplified version - full implementation would be more complex
        shrinkage_intensity = min(1.0, max(0.0, (n_features / n_samples) * 0.1))
    
    # Identity matrix scaled by average variance
    identity_scaled = np.eye(n_features) * np.trace(sample_cov) / n_features
    
    # Shrunk covariance matrix
    shrunk_cov = shrinkage_intensity * identity_scaled + (1 - shrinkage_intensity) * sample_cov
    
    return pd.DataFrame(shrunk_cov, index=returns.columns, columns=returns.columns)


def mean_variance_objective(weights, expected_returns, cov_matrix, risk_aversion=1.0, l2_reg=0.01):
    """
    Mean-variance objective function with L2 regularization.
    
    Args:
        weights: Portfolio weights
        expected_returns: Expected returns vector
        cov_matrix: Covariance matrix
        risk_aversion: Risk aversion parameter (higher = more risk averse)
        l2_reg: L2 regularization strength for diversification
    
    Returns:
        Negative utility (minimize this)
    """
    portfolio_return = np.dot(weights, expected_returns)
    portfolio_risk = np.dot(weights, np.dot(cov_matrix, weights))
    l2_penalty = l2_reg * np.sum(weights**2)  # Encourages diversification
    
    # Utility = Return - (risk_aversion/2) * Risk - L2_penalty
    utility = portfolio_return - (risk_aversion / 2.0) * portfolio_risk - l2_penalty
    
    return -utility  # Minimize negative utility = maximize utility


def enhanced_markowitz_optimization(returns, expected_returns, cov_matrix, risk_aversion=1.0, 
                                  bounds_per_asset=None, l2_reg=0.01):
    """
    Enhanced Markowitz optimization with proper mean-variance tradeoff.
    
    Args:
        returns: Historical returns data
        expected_returns: Expected returns vector
        cov_matrix: Covariance matrix (can be shrunk)
        risk_aversion: Risk aversion level (0.5=aggressive, 1.0=moderate, 2.0=conservative)
        bounds_per_asset: Min/max weight per asset (auto-calculated if None)
        l2_reg: L2 regularization strength
    
    Returns:
        Optimized weights array
    """
    num_assets = len(expected_returns)
    init_guess = np.ones(num_assets) / num_assets
    
    # Auto-calculate bounds based on number of assets
    if bounds_per_asset is None:
        min_weight = 0.02  # 2% minimum
        if num_assets <= 3:
            max_weight = 0.70  # 70% max for small portfolios
        elif num_assets <= 5:
            max_weight = 0.50  # 50% max for medium portfolios  
        elif num_assets <= 10:
            max_weight = 0.30  # 30% max for larger portfolios
        else:
            max_weight = 0.20  # 20% max for very large portfolios
        bounds_per_asset = (min_weight, max_weight)
    
    # Constraints: weights sum to 1
    constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
    
    # Bounds: each asset between bounds_per_asset
    bounds = tuple(bounds_per_asset for _ in range(num_assets))
    
    # Optimize
    result = minimize(
        mean_variance_objective,
        init_guess,
        args=(expected_returns.values, cov_matrix.values, risk_aversion, l2_reg),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 1000, 'ftol': 1e-9}
    )
    
    if not result.success:
        print(f"Optimization warning: {result.message}")
        # Fallback to equal weights if optimization fails
        return init_guess
    
    return result.x


def enhanced_markowitz_loop(returns, window_size=60, rebalance_every=20, risk_profile='moderate'):
    """
    Enhanced Markowitz portfolio optimization with proper mean-variance optimization.
    
    Args:
        returns: Price returns DataFrame
        window_size: Lookback window for estimation
        rebalance_every: Rebalancing frequency
        risk_profile: 'conservative', 'moderate', or 'aggressive'
    
    Returns:
        Tuple of (weights_over_time, dates)
    """
    # Risk aversion parameters for different profiles
    risk_aversion_map = {
        'aggressive': 0.5,    # Higher risk tolerance
        'moderate': 1.0,      # Balanced approach  
        'conservative': 2.0   # Lower risk tolerance
    }
    
    risk_aversion = risk_aversion_map.get(risk_profile, 1.0)
    
    weights_over_time = []
    dates = []
    
    print(f"Enhanced Markowitz with {risk_profile} profile (λ={risk_aversion})")
    
    for i in range(0, len(returns) - window_size, rebalance_every):
        # Get window data (no more * 100 scaling)
        window_returns = returns.iloc[i:i + window_size]
        
        # Calculate expected returns and shrunk covariance matrix
        expected_returns = window_returns.mean()
        shrunk_cov = shrinkage_covariance(window_returns, shrinkage_intensity=0.1)
        
        # Enhanced optimization (bounds auto-calculated based on number of assets)
        optimal_weights = enhanced_markowitz_optimization(
            window_returns, 
            expected_returns, 
            shrunk_cov,
            risk_aversion=risk_aversion,
            bounds_per_asset=None,  # Auto-calculate: 2%-30% for 10 assets
            l2_reg=0.01
        )
        
        # Calculate metrics for comparison
        portfolio_return = np.dot(optimal_weights, expected_returns.values)
        portfolio_risk = np.sqrt(np.dot(optimal_weights, np.dot(shrunk_cov.values, optimal_weights)))
        
        print(f"Period {i//rebalance_every + 1}: Weights={optimal_weights.round(3)}, "
              f"Expected Return={portfolio_return:.4f}, Risk={portfolio_risk:.4f}")
        
        weights_over_time.append(optimal_weights)
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


# 20-Day Aligned LSTM Implementation
def prepare_20day_aligned_data(data, window_size=60, forecast_horizon=20, use_features=True, overlap_step=5):
    """
    Prepare data for 20-day aligned LSTM training with overlapping windows for better training data.
    Uses 60-day windows to predict 20-day cumulative returns.
    
    Args:
        data: Price data DataFrame
        window_size: Input sequence length (default 60)
        forecast_horizon: Prediction horizon (default 20) 
        use_features: Whether to add technical indicators
        overlap_step: Step size for overlapping windows (default 5 days)
    
    Returns:
        X, y, scalers, scaled_data, target_cols
    """
    print(f"Preparing 20-day aligned data: {window_size}-day input → {forecast_horizon}-day prediction")
    print(f"Using overlapping windows with {overlap_step}-day steps for more training data")
    
    if use_features:
        # Add technical indicators
        enhanced_data = add_technical_indicators(data)
        print(f"Enhanced features: {enhanced_data.shape[1]} columns (from {data.shape[1]} original)")
    else:
        enhanced_data = data.copy()
    
    # Use StandardScaler for normalization
    scalers = {}
    scaled_data = pd.DataFrame(index=enhanced_data.index)
    
    for col in enhanced_data.columns:
        scaler = StandardScaler()
        scaled_data[col] = scaler.fit_transform(enhanced_data[[col]])
        scalers[col] = scaler
    
    # Create sequences for 20-day aligned training with overlapping windows
    X, y = [], []
    target_cols = [col for col in data.columns]  # Only predict original asset prices
    
    # Generate overlapping samples for better training data
    for i in range(window_size, len(scaled_data) - forecast_horizon, overlap_step):
        # Input: 60-day window of all features
        X.append(scaled_data.iloc[i-window_size:i].values)
        
        # Target: 20-day cumulative return for each asset
        cumulative_returns = []
        for col in target_cols:
            # Get original price data for this asset
            original_col = col
            start_price = data[original_col].iloc[i-1]  # Price at end of input window
            end_price = data[original_col].iloc[i + forecast_horizon - 1]  # Price after 20 days
            
            # Calculate 20-day cumulative return
            if start_price != 0:
                cum_return = (end_price - start_price) / start_price
            else:
                cum_return = 0.0
            
            # Apply bounds to prevent extreme values
            cum_return = np.clip(cum_return, -0.5, 1.0)  # Reasonable 20-day return bounds
            
            cumulative_returns.append(cum_return)
        
        y.append(cumulative_returns)
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"Created {len(X)} training samples: X={X.shape}, y={y.shape}")
    print(f"Improvement: {len(X)} samples vs {len(range(window_size, len(scaled_data) - forecast_horizon, forecast_horizon))} non-overlapping")
    print(f"Each X sample uses {window_size} days to predict {forecast_horizon}-day returns")
    
    # Add data validation
    if len(X) < 10:
        print(f"⚠️  WARNING: Only {len(X)} training samples available. Consider using smaller overlap_step or more data.")
    
    # Check for invalid targets
    y_array = np.array(y)
    invalid_mask = np.isnan(y_array) | np.isinf(y_array)
    if np.any(invalid_mask):
        print(f"⚠️  WARNING: Found {np.sum(invalid_mask)} invalid target values. Replacing with 0.")
        y_array[invalid_mask] = 0.0
        y = y_array.tolist()
    
    return X, np.array(y), scalers, scaled_data, target_cols


def train_enhanced_lstm_20day(prices_df, window_size=60, forecast_horizon=20, epochs=100, 
                             validation_split=0.2, use_features=True, overlap_step=5, seed=42):
    """
    Train LSTM model for 20-day prediction horizon aligned with rebalancing.
    
    Args:
        prices_df: Price data DataFrame
        window_size: Input sequence length
        forecast_horizon: Prediction horizon (should match rebalance_every)
        epochs: Training epochs
        validation_split: Validation split ratio
        use_features: Whether to use technical indicators
        overlap_step: Step size between training windows (affects training data size)
        seed: Random seed for reproducibility
    
    Returns:
        model, X, scalers, scaled_data, target_cols, history
    """
    print("Preparing enhanced LSTM training data for 20-day prediction...")
    
    # Ensure consistent seeding
    set_seeds(seed)
    
    # Prepare 20-day aligned data with specified overlap_step
    X, y, scalers, scaled_data, target_cols = prepare_20day_aligned_data(
        prices_df, window_size=window_size, forecast_horizon=forecast_horizon, 
        use_features=use_features, overlap_step=overlap_step
    )
    
    print(f"Training data shape: X={X.shape}, y={y.shape}")
    
    # Build enhanced model - same architecture as before
    n_features = X.shape[2]
    n_assets = y.shape[1]
    
    # Model architecture optimized for 20-day prediction
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
        
        # Dense layers
        Dense(64, activation='relu', kernel_regularizer=l1_l2(l1=0.01, l2=0.01)),
        Dropout(0.3),
        Dense(32, activation='relu', kernel_regularizer=l1_l2(l1=0.01, l2=0.01)),
        Dropout(0.2),
        Dense(n_assets, activation='linear')  # Predict cumulative returns
    ])
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )
    
    print(f"Model architecture: {n_features} features → {n_assets} assets (20-day cumulative returns)")
    
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
    print("Training enhanced LSTM model for 20-day prediction...")
    history = model.fit(
        X, y,
        epochs=epochs,
        batch_size=32,
        validation_split=validation_split,
        callbacks=[early_stopping, reduce_lr],
        verbose=1,
        shuffle=True
    )
    
    print("Training completed!")
    
    return model, X, scalers, scaled_data, target_cols, history


def predict_20day_returns(model, prices_df, scalers, window_size=60, forecast_horizon=20, 
                         use_features=True, rebalance_every=20):
    """
    Generate 20-day aligned predictions and convert to proper daily returns for portfolio optimization.
    
    Args:
        model: Trained LSTM model
        prices_df: Price data DataFrame
        scalers: Fitted scalers from training
        window_size: Input sequence length
        forecast_horizon: Prediction horizon
        use_features: Whether to use technical indicators
        rebalance_every: Rebalancing frequency (should match forecast_horizon)
    
    Returns:
        DataFrame with daily returns derived from 20-day predictions
    """
    print(f"Generating 20-day aligned predictions...")
    
    # Prepare features if needed
    if use_features:
        enhanced_data = add_technical_indicators(prices_df)
    else:
        enhanced_data = prices_df.copy()
    
    # Scale data using fitted scalers
    scaled_data = pd.DataFrame(index=enhanced_data.index)
    for col in enhanced_data.columns:
        if col in scalers:
            scaled_data[col] = scalers[col].transform(enhanced_data[[col]])
        else:
            # Handle columns that might not have been in training
            scaled_data[col] = enhanced_data[col]
    
    # Generate predictions at rebalancing intervals
    predicted_cumulative_returns = []
    prediction_dates = []
    prediction_start_dates = []
    
    for i in range(window_size, len(scaled_data) - forecast_horizon, rebalance_every):
        # Extract 60-day window
        X_sample = scaled_data.iloc[i-window_size:i].values.reshape(1, window_size, -1)
        
        # Predict 20-day cumulative returns
        pred = model.predict(X_sample, verbose=0)[0]
        
        # Store prediction with start and end dates
        predicted_cumulative_returns.append(pred)
        prediction_start_dates.append(prices_df.index[i])  # Start of prediction period
        prediction_dates.append(prices_df.index[i + forecast_horizon - 1])  # End of prediction period
    
    print(f"Generated {len(predicted_cumulative_returns)} cumulative return predictions")
    
    # Create daily returns DataFrame
    daily_returns = pd.DataFrame(index=prices_df.index, columns=prices_df.columns, dtype=float)
    daily_returns.fillna(0.0, inplace=True)
    
    # Convert 20-day cumulative returns to equivalent daily returns
    for i, (cum_returns, start_date, end_date) in enumerate(zip(predicted_cumulative_returns, prediction_start_dates, prediction_dates)):
        
        # Convert cumulative returns to equivalent daily returns
        # Formula: daily_rate = (1 + cum_return)^(1/days) - 1
        daily_equivalent_returns = []
        for cum_return in cum_returns:
            # Handle extreme values and ensure valid calculations
            cum_return = np.clip(cum_return, -0.9, 5.0)  # Reasonable bounds for 20-day returns
            
            if cum_return <= -1.0:  # Avoid impossible returns
                daily_equiv = -0.05  # Max daily loss of 5%
            else:
                # Convert to daily equivalent
                daily_equiv = (1 + cum_return) ** (1.0 / forecast_horizon) - 1
                # Clip to reasonable daily return range
                daily_equiv = np.clip(daily_equiv, -0.05, 0.05)
            
            daily_equivalent_returns.append(daily_equiv)
        
        # Apply these daily returns for the prediction period
        start_idx = daily_returns.index.get_loc(start_date)
        end_idx = min(start_idx + forecast_horizon, len(daily_returns))
        
        for day_offset in range(end_idx - start_idx):
            date_idx = start_idx + day_offset
            if date_idx < len(daily_returns):
                daily_returns.iloc[date_idx] = daily_equivalent_returns
        
        print(f"Prediction {i+1}: {forecast_horizon}-day cumulative returns {cum_returns} -> daily equivalent {daily_equivalent_returns}")
    
    # For periods without predictions, use actual historical returns (scaled down)
    actual_returns = prices_df.pct_change().fillna(0)
    mask = (daily_returns == 0).all(axis=1)  # Find rows with no predictions
    daily_returns.loc[mask] = actual_returns.loc[mask] * 0.1  # Use 10% of actual returns as conservative estimate
    
    print(f"Created dense daily returns DataFrame: {daily_returns.shape}")
    print(f"Return range: [{daily_returns.min().min():.4f}, {daily_returns.max().max():.4f}]")
    print(f"Mean daily return: {daily_returns.mean().mean():.6f}")
    
    return daily_returns


def validate_lstm_implementation(prices_df, returns_df, window_size=60, forecast_horizon=20):
    """
    Validate the 20-day LSTM implementation by checking data consistency and return magnitudes.
    
    Args:
        prices_df: Original price data
        returns_df: Generated returns from LSTM
        window_size: LSTM input window size
        forecast_horizon: Prediction horizon
    
    Returns:
        Dictionary with validation results
    """
    print("\n" + "="*60)
    print("LSTM IMPLEMENTATION VALIDATION")
    print("="*60)
    
    # Calculate actual historical returns for comparison
    actual_returns = prices_df.pct_change().fillna(0)
    
    # Basic statistics
    print(f"\n📊 RETURN STATISTICS COMPARISON:")
    print(f"{'Metric':<25} {'Historical':<15} {'LSTM Predicted':<15} {'Ratio':<10}")
    print("-" * 70)
    
    hist_mean = actual_returns.mean().mean()
    lstm_mean = returns_df.mean().mean()
    print(f"{'Mean Daily Return':<25} {hist_mean:<15.6f} {lstm_mean:<15.6f} {lstm_mean/hist_mean if hist_mean != 0 else 'N/A':<10.2f}")
    
    hist_std = actual_returns.std().mean()
    lstm_std = returns_df.std().mean()
    print(f"{'Daily Volatility':<25} {hist_std:<15.6f} {lstm_std:<15.6f} {lstm_std/hist_std if hist_std != 0 else 'N/A':<10.2f}")
    
    hist_min = actual_returns.min().min()
    lstm_min = returns_df.min().min()
    print(f"{'Min Daily Return':<25} {hist_min:<15.6f} {lstm_min:<15.6f} {lstm_min/hist_min if hist_min != 0 else 'N/A':<10.2f}")
    
    hist_max = actual_returns.max().max()
    lstm_max = returns_df.max().max()
    print(f"{'Max Daily Return':<25} {hist_max:<15.6f} {lstm_max:<15.6f} {lstm_max/hist_max if hist_max != 0 else 'N/A':<10.2f}")
    
    # Check for reasonable ranges
    print(f"\n🔍 SANITY CHECKS:")
    reasonable_range = (-0.1, 0.1)  # ±10% daily returns
    extreme_returns = ((returns_df < reasonable_range[0]) | (returns_df > reasonable_range[1])).sum().sum()
    total_returns = returns_df.shape[0] * returns_df.shape[1]
    
    print(f"Returns outside ±10% range: {extreme_returns}/{total_returns} ({100*extreme_returns/total_returns:.2f}%)")
    
    # Check for constant values (potential data issues)
    constant_days = (returns_df.std(axis=1) == 0).sum()
    print(f"Days with constant returns across assets: {constant_days}/{len(returns_df)} ({100*constant_days/len(returns_df):.2f}%)")
    
    # Check for missing or infinite values
    nan_count = returns_df.isna().sum().sum()
    inf_count = np.isinf(returns_df).sum().sum()
    print(f"NaN values: {nan_count}, Infinite values: {inf_count}")
    
    # Calculate 20-day cumulative returns for validation
    print(f"\n📈 20-DAY CUMULATIVE RETURN ANALYSIS:")
    
    # Historical 20-day cumulative returns
    hist_20day = []
    for i in range(0, len(actual_returns) - forecast_horizon, forecast_horizon):
        period_returns = actual_returns.iloc[i:i+forecast_horizon]
        cum_return = (1 + period_returns).prod() - 1
        hist_20day.append(cum_return.mean())
    
    # LSTM-derived 20-day cumulative returns
    lstm_20day = []
    for i in range(0, len(returns_df) - forecast_horizon, forecast_horizon):
        period_returns = returns_df.iloc[i:i+forecast_horizon]
        cum_return = (1 + period_returns).prod() - 1
        lstm_20day.append(cum_return.mean())
    
    if hist_20day and lstm_20day:
        hist_20day_mean = np.mean(hist_20day)
        lstm_20day_mean = np.mean(lstm_20day[:len(hist_20day)])  # Align lengths
        
        print(f"Historical 20-day cum return (mean): {hist_20day_mean:.4f}")
        print(f"LSTM-derived 20-day cum return (mean): {lstm_20day_mean:.4f}")
        print(f"Ratio: {lstm_20day_mean/hist_20day_mean if hist_20day_mean != 0 else 'N/A':.2f}")
    
    # Risk assessment
    print(f"\n⚠️  RISK ASSESSMENT:")
    
    # Check for Markowitz compatibility
    returns_for_markowitz = returns_df * 100  # What Markowitz sees
    extreme_markowitz = ((returns_for_markowitz < -50) | (returns_for_markowitz > 50)).sum().sum()
    print(f"Values that become extreme in Markowitz (×100): {extreme_markowitz}/{total_returns}")
    
    # Calculate maximum portfolio loss with equal weights
    equal_weight_returns = returns_df.mean(axis=1)
    max_daily_loss = equal_weight_returns.min()
    max_20day_loss = equal_weight_returns.rolling(20).sum().min()
    print(f"Max daily portfolio loss (equal weights): {max_daily_loss:.4f}")
    print(f"Max 20-day portfolio loss (equal weights): {max_20day_loss:.4f}")
    
    # Final assessment
    print(f"\n✅ VALIDATION SUMMARY:")
    
    validation_results = {
        'returns_reasonable': extreme_returns / total_returns < 0.05,  # <5% extreme
        'no_missing_data': nan_count == 0 and inf_count == 0,
        'low_constant_days': constant_days / len(returns_df) < 0.1,  # <10% constant
        'compatible_with_markowitz': extreme_markowitz / total_returns < 0.01,  # <1% extreme for Markowitz
        'mean_return': lstm_mean,
        'volatility': lstm_std,
        'extreme_return_pct': extreme_returns / total_returns
    }
    
    all_passed = all(validation_results[key] for key in ['returns_reasonable', 'no_missing_data', 'low_constant_days', 'compatible_with_markowitz'])
    
    if all_passed:
        print("🎉 All validation checks PASSED! LSTM implementation looks good.")
    else:
        print("❌ Some validation checks FAILED. Review implementation.")
        failed_checks = [k for k, v in validation_results.items() if k.endswith('reasonable') or k.endswith('_data') or k.endswith('_days') or k.endswith('_markowitz') and not v]
        print(f"Failed checks: {failed_checks}")
    
    print("="*60 + "\n")
    
    return validation_results