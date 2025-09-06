"""
Hyperparameter Optimization for Portfolio Models
Optimizes parameters for best Sharpe ratio performance
"""

import numpy as np
import pandas as pd
from itertools import product
import models
import validation
from typing import Dict, Tuple, List
import json
import os


def optimize_markowitz_params(returns: pd.DataFrame, 
                             window_sizes: List[int] = [20, 25, 30, 35, 40],
                             rebalance_frequencies: List[int] = [5, 7, 10, 15],
                             risk_profiles: List[str] = ['moderate', 'aggressive']) -> Dict:
    """
    Grid search for optimal Markowitz parameters
    """
    print("\n=== Optimizing Enhanced Markowitz Parameters ===")
    best_sharpe = -np.inf
    best_params = {}
    
    total_combinations = len(window_sizes) * len(rebalance_frequencies) * len(risk_profiles)
    current = 0
    
    for window, rebalance, risk in product(window_sizes, rebalance_frequencies, risk_profiles):
        current += 1
        print(f"Testing {current}/{total_combinations}: window={window}, rebalance={rebalance}, risk={risk}", end='\r')
        
        try:
            # Run Markowitz with current parameters
            weights = models.enhanced_markowitz_loop(returns, window, rebalance, risk)
            
            # Calculate Sharpe ratio
            prices = (1 + returns).cumprod()
            initial_value = 1000000
            portfolio_values = []
            
            for i, (weight_set, _) in enumerate(weights):
                if i == 0:
                    portfolio_values.append(initial_value)
                else:
                    # Calculate returns since last rebalance
                    start_idx = i * rebalance
                    end_idx = min(start_idx + rebalance, len(returns))
                    period_returns = returns.iloc[start_idx:end_idx]
                    
                    # Apply weights to get portfolio returns
                    portfolio_return = (period_returns @ weight_set[0]).sum()
                    new_value = portfolio_values[-1] * (1 + portfolio_return)
                    portfolio_values.append(new_value)
            
            # Calculate Sharpe ratio
            portfolio_returns = pd.Series(portfolio_values).pct_change().dropna()
            if len(portfolio_returns) > 0:
                sharpe = (portfolio_returns.mean() / portfolio_returns.std()) * np.sqrt(252)
                
                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_params = {
                        'window_size': window,
                        'rebalance_every': rebalance,
                        'risk_profile': risk,
                        'sharpe_ratio': sharpe
                    }
        except Exception as e:
            continue
    
    print(f"\n\nBest Markowitz parameters: {best_params}")
    return best_params


def optimize_lstm_params(prices: pd.DataFrame,
                         epochs_list: List[int] = [75, 100, 125, 150],
                         overlap_steps: List[int] = [3, 5, 7],
                         window_size: int = 30,
                         rebalance_every: int = 7) -> Dict:
    """
    Optimize LSTM hyperparameters
    """
    print("\n=== Optimizing LSTM Parameters ===")
    best_sharpe = -np.inf
    best_params = {}
    
    total_combinations = len(epochs_list) * len(overlap_steps)
    current = 0
    
    for epochs, overlap in product(epochs_list, overlap_steps):
        current += 1
        print(f"Testing {current}/{total_combinations}: epochs={epochs}, overlap={overlap}", end='\r')
        
        try:
            # Train LSTM with current parameters
            lstm_model, X, scalers, scaled_data, target_cols, history = models.train_enhanced_lstm_20day(
                prices, window_size=window_size, forecast_horizon=rebalance_every,
                epochs=epochs, use_features=True, overlap_step=overlap, seed=42
            )
            
            # Generate predictions
            returns_pred = models.predict_20day_returns(
                lstm_model, prices, scalers, window_size=window_size,
                forecast_horizon=rebalance_every, use_features=True, rebalance_every=rebalance_every
            )
            
            # Apply Markowitz to predictions
            weights = models.enhanced_markowitz_loop(returns_pred, window_size, rebalance_every, 'moderate')
            
            # Calculate Sharpe ratio (simplified)
            returns = prices.pct_change()
            portfolio_returns = []
            
            for i, (weight_set, _) in enumerate(weights):
                if i > 0:
                    start_idx = i * rebalance_every
                    end_idx = min(start_idx + rebalance_every, len(returns))
                    period_returns = returns.iloc[start_idx:end_idx]
                    portfolio_return = (period_returns @ weight_set[0]).mean()
                    portfolio_returns.append(portfolio_return)
            
            if portfolio_returns:
                portfolio_returns = pd.Series(portfolio_returns)
                sharpe = (portfolio_returns.mean() / portfolio_returns.std()) * np.sqrt(252)
                
                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_params = {
                        'epochs': epochs,
                        'overlap_step': overlap,
                        'sharpe_ratio': sharpe
                    }
        except Exception as e:
            continue
    
    print(f"\n\nBest LSTM parameters: {best_params}")
    return best_params


def optimize_rl_params(returns: pd.DataFrame,
                      timesteps_list: List[int] = [150000, 200000, 300000],
                      learning_rates: List[float] = [1e-4, 3e-4, 5e-4],
                      window_size: int = 30) -> Dict:
    """
    Optimize RL hyperparameters
    """
    print("\n=== Optimizing RL Parameters ===")
    best_sharpe = -np.inf
    best_params = {}
    
    total_combinations = len(timesteps_list) * len(learning_rates)
    current = 0
    
    for timesteps, lr in product(timesteps_list, learning_rates):
        current += 1
        print(f"Testing {current}/{total_combinations}: timesteps={timesteps}, lr={lr}", end='\r')
        
        try:
            # Train RL with current parameters
            rl_model, rl_env = models.train_enhanced_rl_model(
                returns, window_size,
                total_timesteps=timesteps,
                learning_rate=lr,
                seed=42
            )
            
            # Generate predictions
            rl_weights, rl_dates = models.predict_rl_weights(rl_model, returns, window_size, 20)
            
            # Calculate Sharpe ratio
            portfolio_returns = []
            for i, weights in enumerate(rl_weights):
                if i > 0:
                    start_idx = i * 20
                    end_idx = min(start_idx + 20, len(returns))
                    period_returns = returns.iloc[start_idx:end_idx]
                    portfolio_return = (period_returns @ weights).mean()
                    portfolio_returns.append(portfolio_return)
            
            if portfolio_returns:
                portfolio_returns = pd.Series(portfolio_returns)
                sharpe = (portfolio_returns.mean() / portfolio_returns.std()) * np.sqrt(252)
                
                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_params = {
                        'total_timesteps': timesteps,
                        'learning_rate': lr,
                        'sharpe_ratio': sharpe
                    }
        except Exception as e:
            continue
    
    print(f"\n\nBest RL parameters: {best_params}")
    return best_params


def run_full_optimization(prices: pd.DataFrame, returns: pd.DataFrame, 
                         output_dir: str = 'optimized_params') -> Dict:
    """
    Run complete hyperparameter optimization for all models
    """
    os.makedirs(output_dir, exist_ok=True)
    
    all_params = {}
    
    # Optimize Enhanced Markowitz
    print("\n" + "="*80)
    print("HYPERPARAMETER OPTIMIZATION")
    print("="*80)
    
    markowitz_params = optimize_markowitz_params(returns)
    all_params['markowitz'] = markowitz_params
    
    # Save intermediate results
    with open(f'{output_dir}/markowitz_params.json', 'w') as f:
        json.dump(markowitz_params, f, indent=2)
    
    # Optimize LSTM (using best Markowitz window/rebalance)
    best_window = markowitz_params.get('window_size', 45)
    best_rebalance = markowitz_params.get('rebalance_every', 10)
    
    lstm_params = optimize_lstm_params(prices, window_size=best_window, rebalance_every=best_rebalance)
    all_params['lstm'] = lstm_params
    
    with open(f'{output_dir}/lstm_params.json', 'w') as f:
        json.dump(lstm_params, f, indent=2)
    
    # Optimize RL
    rl_params = optimize_rl_params(returns, window_size=best_window)
    all_params['rl'] = rl_params
    
    with open(f'{output_dir}/rl_params.json', 'w') as f:
        json.dump(rl_params, f, indent=2)
    
    # Save all parameters
    with open(f'{output_dir}/all_optimized_params.json', 'w') as f:
        json.dump(all_params, f, indent=2)
    
    print("\n" + "="*80)
    print("OPTIMIZATION COMPLETE")
    print("="*80)
    print(f"All optimized parameters saved to {output_dir}/")
    print("\nBest parameters summary:")
    print(f"  Markowitz: window={markowitz_params.get('window_size')}, "
          f"rebalance={markowitz_params.get('rebalance_every')}, "
          f"risk={markowitz_params.get('risk_profile')}, "
          f"Sharpe={markowitz_params.get('sharpe_ratio', 0):.3f}")
    print(f"  LSTM: epochs={lstm_params.get('epochs')}, "
          f"overlap={lstm_params.get('overlap_step')}, "
          f"Sharpe={lstm_params.get('sharpe_ratio', 0):.3f}")
    print(f"  RL: timesteps={rl_params.get('total_timesteps')}, "
          f"lr={rl_params.get('learning_rate')}, "
          f"Sharpe={rl_params.get('sharpe_ratio', 0):.3f}")
    
    return all_params