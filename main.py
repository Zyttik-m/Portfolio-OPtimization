# Import Libraries
import data_handler
import models
import validation
from argparse import ArgumentParser
import os
import pandas as pd
import numpy as np




# Load Dataset

arg = ArgumentParser()

arg.add_argument("--tickers", type=str, nargs='+', default=['SPY', 'TLT', 'GLD', 'QQQ', 'IWM', 'VEA', 'VWO', 'VNQ', 'DBC', 'HYG'])
arg.add_argument("--start", type=str, default='2018-01-01')
arg.add_argument("--end", type=str, default='2025-01-01')
arg.add_argument("--interval", type=str, default='1d')
arg.add_argument("--auto_adjust", type=bool, default=False)
arg.add_argument("--window_size", type=int, default=60)
arg.add_argument("--rebalance_every", type=int, default=20)
arg.add_argument("--train_rl", type=bool, default=True, help="Whether to train RL model")
arg.add_argument("--rl_timesteps", type=int, default=100000, help="Number of timesteps for RL training")
arg.add_argument("--load_rl_model", type=str, default=None, help="Path to load pre-trained RL model")
arg.add_argument("--seed", type=int, default=42, help="Random seed for reproducible results")
arg.add_argument("--use_tuned_params", type=bool, default=False, help="Use hyperparameter tuned parameters if available")
arg.add_argument("--use_enhanced_markowitz", type=bool, default=True, help="Use enhanced Markowitz with mean-variance optimization")
arg.add_argument("--risk_profile", type=str, default='moderate', choices=['conservative', 'moderate', 'aggressive'], help="Risk profile for enhanced Markowitz")


args = arg.parse_args()

# Set random seeds for reproducible results
models.set_seeds(args.seed)

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

print("\n=== Creating Benchmark Model (Equal Weights) ===")
n_assets = len(returns.columns)
equal_weight = 1.0 / n_assets
benchmark_weights = []
benchmark_dates = []

# Create equal weights for same dates as Markowitz
for i in range(0, len(returns) - args.window_size, args.rebalance_every):
    benchmark_weights.append(np.array([equal_weight] * n_assets))
    benchmark_dates.append(returns.index[i + args.window_size])

print(f"Benchmark weights: {[equal_weight] * n_assets}")
print(f"Generated {len(benchmark_weights)} benchmark allocations")


# Model Training - Markowitz Model

if args.use_enhanced_markowitz:
    print(f"\n=== Training Enhanced Markowitz Model ({args.risk_profile} profile) ===")
    markowitz_weights = models.enhanced_markowitz_loop(returns, args.window_size, args.rebalance_every, args.risk_profile)
else:
    print("\n=== Training Classical Markowitz Model ===")
    markowitz_weights = models.markowitzloop(returns, args.window_size , args.rebalance_every)


# Model Training - Enhanced Markowitz with LSTM (20-day aligned)

print("\n=== Training Enhanced Markowitz + LSTM Model (20-day aligned) ===")

# Load tuned parameters if requested
lstm_epochs = 100  # Default
lstm_overlap_step = 5  # Default

if args.use_tuned_params:
    import pickle
    import os
    
    # Try to load tuned LSTM parameters
    if os.path.exists('best_lstm_quick_params.pkl'):
        with open('best_lstm_quick_params.pkl', 'rb') as f:
            lstm_params = pickle.load(f)
            lstm_epochs = lstm_params.get('epochs', 100)
            lstm_overlap_step = lstm_params.get('overlap_step', 5)
            print(f"Using tuned LSTM parameters: epochs={lstm_epochs}, overlap_step={lstm_overlap_step}")
    elif os.path.exists('best_lstm_20day_params.pkl'):
        with open('best_lstm_20day_params.pkl', 'rb') as f:
            lstm_params = pickle.load(f)
            lstm_epochs = lstm_params.get('epochs', 100)
            lstm_overlap_step = lstm_params.get('overlap_step', 5)
            print(f"Using tuned LSTM parameters: epochs={lstm_epochs}, overlap_step={lstm_overlap_step}")
    else:
        print("No tuned LSTM parameters found, using defaults")

# Use the new 20-day aligned LSTM training function with tuned overlap_step
lstm_model, X, scalers, scaled_data, target_cols, history = models.train_enhanced_lstm_20day(
    prices, window_size=args.window_size, forecast_horizon=args.rebalance_every, 
    epochs=lstm_epochs, use_features=True, overlap_step=lstm_overlap_step, seed=args.seed
)

# Generate 20-day aligned predictions
returns_pred = models.predict_20day_returns(
    lstm_model, prices, scalers, window_size=args.window_size, 
    forecast_horizon=args.rebalance_every, use_features=True, rebalance_every=args.rebalance_every
)

print(f"20-day aligned LSTM prediction completed. Returns shape: {returns_pred.shape}")

# Validate the LSTM implementation
print("\n=== Validating LSTM Implementation ===")
validation_results = models.validate_lstm_implementation(prices, returns_pred, args.window_size, args.rebalance_every)

# Apply Markowitz optimization to predicted returns
if args.use_enhanced_markowitz:
    print(f"Applying Enhanced Markowitz optimization to LSTM predictions ({args.risk_profile} profile)")
    lstm_weights = models.enhanced_markowitz_loop(returns_pred, args.window_size, args.rebalance_every, args.risk_profile)
else:
    print("Applying Classical Markowitz optimization to LSTM predictions")
    lstm_weights = models.markowitzloop(returns_pred, args.window_size , args.rebalance_every)

# Model Training - Reinforcement Learning

if args.load_rl_model:
    print(f"\n=== Loading Pre-trained RL Model from {args.load_rl_model} ===")
    rl_model = models.load_rl_model(args.load_rl_model)
    if rl_model:
        rl_weights, rl_dates = models.predict_rl_weights(rl_model, returns, args.window_size, args.rebalance_every)
        print(f"Generated {len(rl_weights)} portfolio allocations using loaded RL model")
    else:
        print("Failed to load model, skipping RL predictions")
        rl_weights, rl_dates = [], []
elif args.train_rl:
    print("\n=== Training Enhanced Reinforcement Learning Model ===")
    
    # Load tuned RL parameters if requested
    rl_timesteps = args.rl_timesteps  # Default from args
    rl_learning_rate = 1e-4  # Default
    
    if args.use_tuned_params:
        # Try to load tuned RL parameters
        if os.path.exists('best_rl_quick_params.pkl'):
            with open('best_rl_quick_params.pkl', 'rb') as f:
                rl_params = pickle.load(f)
                rl_timesteps = rl_params.get('total_timesteps', args.rl_timesteps)
                rl_learning_rate = rl_params.get('learning_rate', 1e-4)
                print(f"Using tuned RL parameters: timesteps={rl_timesteps}, learning_rate={rl_learning_rate}")
        elif os.path.exists('best_rl_params.pkl'):
            with open('best_rl_params.pkl', 'rb') as f:
                rl_params = pickle.load(f)
                rl_timesteps = rl_params.get('total_timesteps', args.rl_timesteps)
                rl_learning_rate = rl_params.get('learning_rate', 1e-4)
                print(f"Using tuned RL parameters: timesteps={rl_timesteps}, learning_rate={rl_learning_rate}")
        else:
            print("No tuned RL parameters found, using defaults")
    
    rl_model, rl_env = models.train_enhanced_rl_model(
        returns, args.window_size, 
        total_timesteps=rl_timesteps, 
        learning_rate=rl_learning_rate,
        seed=args.seed
    )
    
    # Save the trained model
    rl_model.save("ppo_portfolio_model_enhanced")
    print("Enhanced RL model saved to 'ppo_portfolio_model_enhanced'")
    
    # Generate portfolio weights using trained model
    rl_weights, rl_dates = models.predict_rl_weights(rl_model, returns, args.window_size, args.rebalance_every)
    print(f"Generated {len(rl_weights)} portfolio allocations using enhanced RL model")
else:
    print("\n=== Skipping RL Training (--train_rl=False) ===")
    rl_weights, rl_dates = [], []










# Model Comparison

print("\n" + "="*80)
print("PORTFOLIO MODELS COMPARISON")
print("="*80)

# Prepare model weights dictionary
model_weights = {
    'Equal Weights (Benchmark)': (benchmark_weights, benchmark_dates),
    'Markowitz': markowitz_weights,
    'Markowitz + LSTM': lstm_weights,
}

# Add RL model if available
if 'rl_weights' in locals() and rl_weights:
    model_weights['Reinforcement Learning (PPO)'] = (rl_weights, rl_dates)

# Run comprehensive comparison
print("Running comprehensive backtesting and comparison...")
summary_df = validation.compare_models(model_weights, prices, returns, save_plots=True)

print(f"\nComparison complete! Results saved to performance_table.csv")
print("Plots saved to img/ directory:")
print("- img/cumulative_returns.png")
print("- img/portfolio_weights.png") 
print("- img/drawdown.png")
print("- img/performance_metrics.png")
print("- img/rolling_sharpe.png")

# Display final summary
print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
print("All models have been trained, backtested, and compared.")
print("Check the generated plots and CSV file for detailed analysis.")





