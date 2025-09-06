"""
Hyperparameter tuning utilities for LSTM and RL models.
"""

import numpy as np
import pandas as pd
import models
import validation
from sklearn.model_selection import TimeSeriesSplit
import itertools
import pickle
import os


def tune_lstm_hyperparameters_20day(prices_df, param_grid=None, n_splits=3, save_results=True, seed=42):
    """
    Tune 20-day aligned LSTM hyperparameters.
    
    Args:
        prices_df: Price data
        param_grid: Dictionary of hyperparameters to tune
        n_splits: Number of CV splits 
        save_results: Whether to save results
        seed: Random seed for reproducibility
    """
    if param_grid is None:
        param_grid = {
            'window_size': [60], 
            'forecast_horizon': [20],  
            'overlap_step': [5, 10, 20],  
            'epochs': [50, 100, 150],
            'use_features': [True],  
            'validation_split': [0.2]
        }
    
    print("Starting 20-day aligned LSTM hyperparameter tuning...")
    print(f"Parameter grid: {param_grid}")
    
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    param_combinations = list(itertools.product(*param_values))
    
    print(f"Total combinations to test: {len(param_combinations)}")
    
    results = []
    best_score = -np.inf
    best_params = None
    
    for i, param_combo in enumerate(param_combinations):
        params = dict(zip(param_names, param_combo))
        print(f"\n{'='*60}")
        print(f"Testing combination {i+1}/{len(param_combinations)}: {params}")
        print(f"{'='*60}")
        
        try:
            models.set_seeds(seed)
            
            X, y, scalers, scaled_data, target_cols = models.prepare_20day_aligned_data(
                prices_df,
                window_size=params['window_size'],
                forecast_horizon=params['forecast_horizon'],
                use_features=params['use_features'],
                overlap_step=params.get('overlap_step', 5)
            )
            
            print(f"Training samples created: {len(X)}")
            
            # Train 20-day aligned model
            model, X_train, scalers_train, scaled_data_train, target_cols_train, history = models.train_enhanced_lstm_20day(
                prices_df,
                window_size=params['window_size'],
                forecast_horizon=params['forecast_horizon'],
                epochs=params['epochs'],
                validation_split=params['validation_split'],
                use_features=params['use_features'],
                seed=seed
            )
            
            # Get validation loss
            val_loss = min(history.history['val_loss'])
            
            # Generate 20-day aligned predictions
            returns_pred = models.predict_20day_returns(
                model, prices_df, scalers_train,
                window_size=params['window_size'],
                forecast_horizon=params['forecast_horizon'],
                use_features=params['use_features'],
                rebalance_every=params['forecast_horizon']
            )
            
            # Quick Markowitz optimization
            weights, dates = models.markowitzloop(
                returns_pred, params['window_size'], params['forecast_horizon']
            )
            
            # Calculate Sharpe ratio as performance metric
            if weights:
                returns_df = prices_df.pct_change()
                portfolio_df = validation.calculate_portfolio_returns(
                    weights, dates, returns_df
                )
                sharpe = validation.calculate_sharpe_ratio(portfolio_df['returns'])
            else:
                sharpe = -10  # Penalty for failed optimization
            
            score = sharpe  # Use Sharpe ratio as optimization target
            
            result = {
                'params': params,
                'val_loss': val_loss,
                'sharpe_ratio': sharpe,
                'score': score
            }
            
            results.append(result)
            
            if score > best_score:
                best_score = score
                best_params = params.copy()
                print(f"New best score: {score:.4f}")
            
            print(f"Val Loss: {val_loss:.4f}, Sharpe: {sharpe:.4f}")
            
        except Exception as e:
            print(f"Error with parameters {params}: {e}")
            results.append({
                'params': params,
                'val_loss': np.inf,
                'sharpe_ratio': -10,
                'score': -10,
                'error': str(e)
            })
    
    print("\n" + "="*60)
    print("20-DAY LSTM HYPERPARAMETER TUNING RESULTS")
    print("="*60)
    print(f"Best parameters: {best_params}")
    print(f"Best score (Sharpe): {best_score:.4f}")
    
    if save_results:
        # Save results
        results_df = pd.DataFrame([{**r['params'], **{k:v for k,v in r.items() if k != 'params'}} 
                                  for r in results])
        results_df.to_csv('lstm_20day_hyperparameter_results.csv', index=False)
        
        # Save best parameters
        with open('best_lstm_20day_params.pkl', 'wb') as f:
            pickle.dump(best_params, f)
        
        print(f"Results saved to lstm_20day_hyperparameter_results.csv")
        print(f"Best parameters saved to best_lstm_20day_params.pkl")
    
    return best_params, results


def tune_rl_hyperparameters(returns_df, param_grid=None, save_results=True):
    """
    Tune RL hyperparameters.
    
    Args:
        returns_df: Returns data
        param_grid: Dictionary of hyperparameters to tune
        save_results: Whether to save results
    """
    if param_grid is None:
        param_grid = {
            'window_size': [30, 60],
            'total_timesteps': [50000, 100000],
            'learning_rate': [1e-4, 3e-4],
            'transaction_cost': [0.001, 0.005]
        }
    
    print("Starting RL hyperparameter tuning...")
    print(f"Parameter grid: {param_grid}")
    
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    param_combinations = list(itertools.product(*param_values))
    
    results = []
    best_score = -np.inf
    best_params = None
    
    for i, param_combo in enumerate(param_combinations):
        params = dict(zip(param_names, param_combo))
        print(f"\nTesting RL combination {i+1}/{len(param_combinations)}: {params}")
        
        try:
            # Create environment with these parameters
            env = models.PortfolioEnv(
                returns_df, 
                window_size=params['window_size'],
                transaction_cost=params['transaction_cost']
            )
            
            # Train RL model
            model, _ = models.train_enhanced_rl_model(
                returns_df,
                window_size=params['window_size'],
                total_timesteps=params['total_timesteps'],
                learning_rate=params['learning_rate']
            )
            
            weights, dates = models.predict_rl_weights(
                model, returns_df, params['window_size'], 20
            )
            
            if weights:
                portfolio_df = validation.calculate_portfolio_returns(
                    weights, dates, returns_df
                )
                sharpe = validation.calculate_sharpe_ratio(portfolio_df['returns'])
                annual_return = validation.calculate_annual_return(portfolio_df['portfolio_value'])
                volatility = validation.calculate_volatility(portfolio_df['returns'])
                
                # Combined score: Sharpe ratio with return bonus
                score = sharpe + (annual_return / 100) * 0.1
            else:
                sharpe = annual_return = volatility = score = -10
            
            result = {
                'params': params,
                'sharpe_ratio': sharpe,
                'annual_return': annual_return,
                'volatility': volatility,
                'score': score
            }
            
            results.append(result)
            
            if score > best_score:
                best_score = score
                best_params = params.copy()
                print(f"New best RL score: {score:.4f}")
            
            print(f"Sharpe: {sharpe:.4f}, Annual Return: {annual_return:.2f}%, Vol: {volatility:.2f}%")
            
        except Exception as e:
            print(f"Error with RL parameters {params}: {e}")
            results.append({
                'params': params,
                'sharpe_ratio': -10,
                'annual_return': 0,
                'volatility': 0,
                'score': -10,
                'error': str(e)
            })
    
    print("\n" + "="*60)
    print("RL HYPERPARAMETER TUNING RESULTS")
    print("="*60)
    print(f"Best RL parameters: {best_params}")
    print(f"Best RL score: {best_score:.4f}")
    
    if save_results:
        # Save results
        results_df = pd.DataFrame([{**r['params'], **{k:v for k,v in r.items() if k != 'params'}} 
                                  for r in results])
        results_df.to_csv('rl_hyperparameter_results.csv', index=False)
        
        # Save best parameters
        with open('best_rl_params.pkl', 'wb') as f:
            pickle.dump(best_params, f)
        
        print(f"RL results saved to rl_hyperparameter_results.csv")
    
    return best_params, results


def load_best_parameters():
    """
    Load best parameters from previous tuning sessions.
    """
    lstm_params = None
    rl_params = None
    
    if os.path.exists('best_lstm_params.pkl'):
        with open('best_lstm_params.pkl', 'rb') as f:
            lstm_params = pickle.load(f)
        print(f"Loaded best LSTM parameters: {lstm_params}")
    
    if os.path.exists('best_rl_params.pkl'):
        with open('best_rl_params.pkl', 'rb') as f:
            rl_params = pickle.load(f)
        print(f"Loaded best RL parameters: {rl_params}")
    
    return lstm_params, rl_params


def quick_hyperparameter_search(prices_df, returns_df):
    """
    Run a quick hyperparameter search for both models.
    """
    print("Starting quick hyperparameter search...")
    
    # Quick LSTM tuning
    lstm_grid = {
        'window_size': [60],
        'epochs': [50, 100],
        'use_features': [True, False],
        'validation_split': [0.2]
    }
    
    best_lstm_params, lstm_results = tune_lstm_hyperparameters(
        prices_df, lstm_grid, save_results=True
    )
    
    # Quick RL tuning
    rl_grid = {
        'window_size': [60],
        'total_timesteps': [50000, 100000],
        'learning_rate': [1e-4, 3e-4],
        'transaction_cost': [0.001]
    }
    
    best_rl_params, rl_results = tune_rl_hyperparameters(
        returns_df, rl_grid, save_results=True
    )
    
    return best_lstm_params, best_rl_params


if __name__ == "__main__":
    # Example usage
    import data_handler
    
    # Load data
    tickers = ['SPY', 'TLT', 'GLD']
    datalist = []
    
    for ticker in tickers:
        data = data_handler.load_data(ticker, '2018-01-01', '2025-01-01')
        data = data_handler.cleandata(data, ticker)
        datalist.append(data)
    
    prices = pd.concat(datalist, axis=1)
    returns = prices.pct_change()
    
    # Run hyperparameter tuning
    best_lstm, best_rl = quick_hyperparameter_search(prices, returns)
    
    print("\nHyperparameter tuning complete!")
    print(f"Best LSTM params: {best_lstm}")
    print(f"Best RL params: {best_rl}")