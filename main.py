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

arg.add_argument("--tickers", type=str, nargs='+', default=['SPY', 'TLT', 'GLD'])
arg.add_argument("--start", type=str, default='2018-01-01')
arg.add_argument("--end", type=str, default='2025-01-01')
arg.add_argument("--interval", type=str, default='1d')
arg.add_argument("--auto_adjust", type=bool, default=False)
arg.add_argument("--window_size", type=int, default=60)
arg.add_argument("--rebalance_every", type=int, default=20)
arg.add_argument("--train_rl", type=bool, default=True, help="Whether to train RL model")
arg.add_argument("--rl_timesteps", type=int, default=100000, help="Number of timesteps for RL training")
arg.add_argument("--load_rl_model", type=str, default=None, help="Path to load pre-trained RL model")


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

markowitz_weights = models.markowitzloop(returns, args.window_size , args.rebalance_every)


# Model Training - Enhanced Markowitz with LSTM

print("\n=== Training Enhanced Markowitz + LSTM Model ===")

# Use the enhanced LSTM training function
lstm_model, X, scalers, scaled_data, target_cols, history = models.train_enhanced_lstm(
    prices, window_size=args.window_size, epochs=100, use_features=True
)

# Generate predictions using the enhanced model
pred_df = models.predict_on_existing_data(lstm_model, X, scaled_data, scalers, prices, target_cols, args.window_size)
returns_pred = pred_df.pct_change()

print(f"LSTM prediction completed. Predicting returns with shape: {returns_pred.shape}")

# Apply Markowitz optimization to predicted returns
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
    rl_model, rl_env = models.train_enhanced_rl_model(returns, args.window_size, total_timesteps=args.rl_timesteps)
    
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





