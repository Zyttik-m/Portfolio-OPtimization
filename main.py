# Import Libraries
import data_handler
import models
import validation
from argparse import ArgumentParser
import os
import pandas as pd
import numpy as np
import json
from datetime import datetime




# Load Dataset

arg = ArgumentParser()

# Mode selection 
arg.add_argument("--mode", type=str, default='realistic', choices=['realistic', 'optimized', 'changeperiod'], 
                 help="Mode: 'realistic' for conservative real-world settings, 'optimized' for best performance, 'changeperiod' for realistic models with optimized period")

arg.add_argument("--tickers", type=str, nargs='+', default=['SPY', 'TLT', 'GLD', 'QQQ', 'IWM', 'VEA', 'VWO', 'VNQ', 'DBC', 'HYG'])
arg.add_argument("--start", type=str, default=None, help="Start date (auto-set based on mode if not specified)")
arg.add_argument("--end", type=str, default='2025-01-01')
arg.add_argument("--interval", type=str, default='1d')
arg.add_argument("--auto_adjust", type=bool, default=False)
arg.add_argument("--window_size", type=int, default=None, help="Lookback window (auto-set based on mode if not specified)")
arg.add_argument("--rebalance_every", type=int, default=None, help="Rebalancing frequency (auto-set based on mode if not specified)")
arg.add_argument("--train_rl", type=bool, default=True, help="Whether to train RL model")
arg.add_argument("--rl_timesteps", type=int, default=None, help="Number of timesteps for RL training (auto-set based on mode)")
arg.add_argument("--load_rl_model", type=str, default=None, help="Path to load pre-trained RL model")
arg.add_argument("--seed", type=int, default=42, help="Random seed for reproducible results")
arg.add_argument("--use_tuned_params", type=bool, default=False, help="Use hyperparameter tuned parameters if available")
arg.add_argument("--use_enhanced_markowitz", type=bool, default=True, help="Use enhanced Markowitz with mean-variance optimization")
arg.add_argument("--risk_profile", type=str, default=None, help="Risk profile (auto-set based on mode if not specified)")
arg.add_argument("--optimize_hyperparams", type=bool, default=None, help="Run hyperparameter optimization (auto-set for optimized mode)")


args = arg.parse_args()

# Configure parameters based on mode
if args.mode == 'optimized':
    # Optimized mode settings 
    print("\n" + "="*80)
    print("RUNNING IN OPTIMIZED MODE - Academic Best-Case Settings")
    print("="*80)
    
    if args.start is None:
        args.start = '2010-01-01'  # Start of bull market 
    if args.end is None or args.end == '2025-01-01':
        args.end = '2020-01-01'  # End before COVID crash
    if args.window_size is None:
        args.window_size = 30  # Shorter lookback for responsiveness
    if args.rebalance_every is None:
        args.rebalance_every = 7  # Weekly rebalancing
    if args.rl_timesteps is None:
        args.rl_timesteps = 200000  # Proven optimal RL training
    if args.risk_profile is None:
        args.risk_profile = 'aggressive'  # Higher risk tolerance
    if args.optimize_hyperparams is None:
        args.optimize_hyperparams = True
    
    args.target_volatility = 0.15  
    args.risk_free_rate = 0.0  
    args.transaction_cost = 0.0001  
    
    # Set output directory for optimized results
    output_dir = 'results_optimized'
    img_dir = 'img_optimized'
    
elif args.mode == 'changeperiod':
    # Changeperiod mode 
    print("\n" + "="*80)
    print("RUNNING IN CHANGEPERIOD MODE - Realistic Models with Bull Market Period")
    print("="*80)
    
    # Use optimized period but realistic settings
    if args.start is None:
        args.start = '2010-01-01'  # Same as optimized period
    if args.end is None or args.end == '2025-01-01':
        args.end = '2020-01-01'  # Same as optimized period
    if args.window_size is None:
        args.window_size = 60  # Same as realistic
    if args.rebalance_every is None:
        args.rebalance_every = 20  # Same as realistic
    if args.rl_timesteps is None:
        args.rl_timesteps = 100000  # Same as realistic
    if args.risk_profile is None:
        args.risk_profile = 'moderate'  # Same as realistic
    if args.optimize_hyperparams is None:
        args.optimize_hyperparams = True  # Enable optimization
    
    # Set output directory for changeperiod results
    output_dir = 'results_changeperiod'
    img_dir = 'img_changeperiod'
    
else:
    # Realistic mode settings 
    print("\n" + "="*80)
    print("RUNNING IN REALISTIC MODE - Conservative Real-World Settings")
    print("="*80)
    
    # Set realistic defaults if not specified
    if args.start is None:
        args.start = '2018-01-01'  # 7 years of data
    if args.window_size is None:
        args.window_size = 60  # Standard lookback
    if args.rebalance_every is None:
        args.rebalance_every = 20  # Monthly rebalancing
    if args.rl_timesteps is None:
        args.rl_timesteps = 100000  # Standard training
    if args.risk_profile is None:
        args.risk_profile = 'moderate'  # Moderate risk
    if args.optimize_hyperparams is None:
        args.optimize_hyperparams = True  # Enable by default for realistic mode too
    
    # Set output directory for realistic results
    output_dir = 'results_realistic'
    img_dir = 'img_realistic'

# Display configuration
print(f"\nConfiguration:")
print(f"  Start Date: {args.start}")
print(f"  End Date: {args.end}")
print(f"  Window Size: {args.window_size} days")
print(f"  Rebalance Every: {args.rebalance_every} days")
print(f"  RL Timesteps: {args.rl_timesteps}")
print(f"  Risk Profile: {args.risk_profile}")
print(f"  Output Directory: {output_dir}/")
print(f"  Optimize Hyperparameters: {args.optimize_hyperparams}")

# Show advanced settings for optimized mode
if args.mode == 'optimized':
    print(f"  Target Volatility: {args.target_volatility * 100:.1f}%")
    print(f"  Risk-Free Rate: {args.risk_free_rate * 100:.1f}%")
    print(f"  Transaction Cost: {args.transaction_cost * 100:.2f}%")

print("\n" + "="*80)

# Create output directories
os.makedirs(output_dir, exist_ok=True)
os.makedirs(img_dir, exist_ok=True)

# Set random seeds for reproducible results
models.set_seeds(args.seed)

os.makedirs('data', exist_ok=True)

# Load data with mode-specific date range
print(f"\nLoading data from {args.start} to {args.end}...")
for name in args.tickers:
    data = data_handler.load_data(name, args.start, args.end, args.interval, args.auto_adjust)
    # Save to mode-specific data directory
    data_dir = f'data_{args.mode}'
    os.makedirs(data_dir, exist_ok=True)
    data.to_csv(f"{data_dir}/{name}.csv")
    
print(f"Data saved to {data_dir}/ directory")


# Data Preprocessing

datalist = []

# Use mode-specific data directory
path = f'./data_{args.mode}'

for file in os.listdir(path):
    if file.endswith('.csv'):
        data = pd.read_csv(path + '/' + file)
        data = data_handler.cleandata(data, file[:-4])
        datalist.append(data)

prices = pd.concat(datalist, axis=1)
prices = prices.apply(pd.to_numeric)
returns = prices.pct_change()

# Run hyperparameter optimization 
optimized_params = {}
if args.optimize_hyperparams:
    print("\n" + "="*80)
    print(f"RUNNING HYPERPARAMETER OPTIMIZATION FOR {args.mode.upper()} MODE")
    print("This may take 15-30 minutes...")
    print("="*80)
    
    import hyperparameter_optimizer as hp_opt
    
    # Use mode-specific filename for storing parameters
    opt_params_file = f'{output_dir}/{args.mode}_optimized_parameters.json'
    
    # Check if optimization was already done for this mode
    if os.path.exists(opt_params_file):
        print(f"Loading existing optimized parameters from {opt_params_file}")
        with open(opt_params_file, 'r') as f:
            optimized_params = json.load(f)
    else:
        # Run full optimization for this mode
        print(f"Running optimization for {args.mode} mode with data from {args.start} to {args.end}")
        optimized_params = hp_opt.run_full_optimization(prices, returns, output_dir)
        
        # Save for future use with mode-specific filename
        with open(opt_params_file, 'w') as f:
            json.dump(optimized_params, f, indent=2)
        print(f"Saved optimized parameters to {opt_params_file}")
    
    # Override arguments with optimized values
    if 'markowitz' in optimized_params:
        args.window_size = optimized_params['markowitz'].get('window_size', args.window_size)
        args.rebalance_every = optimized_params['markowitz'].get('rebalance_every', args.rebalance_every)
        args.risk_profile = optimized_params['markowitz'].get('risk_profile', args.risk_profile)
        
    print(f"\nUsing optimized parameters for {args.mode} mode:")
    print(f"  Window Size: {args.window_size}")
    print(f"  Rebalance Every: {args.rebalance_every}")
    print(f"  Risk Profile: {args.risk_profile}")


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
    markowitz_weights = models.enhanced_markowitz_loop(returns, args.window_size, args.rebalance_every, args.risk_profile, args.mode)
else:
    print("\n=== Training Classical Markowitz Model ===")
    markowitz_weights = models.markowitzloop(returns, args.window_size , args.rebalance_every)


# Model Training - LSTM Model (Mode-specific)

if args.mode == 'optimized':
    # Use Multi-Horizon LSTM + Kelly model for optimized mode
    print("\n=== Training Multi-Horizon LSTM + Kelly Model (Optimized Mode) ===")
    lstm_weights = models.multi_horizon_lstm_kelly_model(
        prices, returns, window_size=args.window_size, 
        rebalance_every=args.rebalance_every, mode=args.mode
    )
else:
    # Use Enhanced Markowitz + LSTM for realistic mode
    print("\n=== Training Enhanced Markowitz + LSTM Model (20-day aligned) ===")
    
    # Load tuned parameters if requested
    lstm_epochs = 100  # Standard for realistic
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

    # Use the standard 20-day aligned LSTM training for realistic mode
    lstm_model, X, scalers, scaled_data, target_cols, history = models.train_enhanced_lstm_20day(
        prices, window_size=args.window_size, forecast_horizon=args.rebalance_every, 
        epochs=lstm_epochs, use_features=True, overlap_step=lstm_overlap_step, seed=args.seed, mode=args.mode
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

    # Apply Markowitz optimization to predicted returns for realistic mode
    if args.use_enhanced_markowitz:
        print(f"Applying Enhanced Markowitz optimization to LSTM predictions ({args.risk_profile} profile)")
        lstm_weights = models.enhanced_markowitz_loop(returns_pred, args.window_size, args.rebalance_every, args.risk_profile, args.mode)
    else:
        print("Applying Classical Markowitz optimization to LSTM predictions")
        lstm_weights = models.markowitzloop(returns_pred, args.window_size, args.rebalance_every)

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
    
    # Load tuned RL parameters 
    rl_timesteps = args.rl_timesteps  # Default from args
    rl_learning_rate = 1e-4 if args.mode == 'realistic' else 3e-4  # Higher for optimized
    
    # Use optimized params if available
    if args.mode == 'optimized' and 'rl' in optimized_params:
        rl_timesteps = optimized_params['rl'].get('total_timesteps', rl_timesteps)
        rl_learning_rate = optimized_params['rl'].get('learning_rate', rl_learning_rate)
        print(f"Using optimized RL parameters: timesteps={rl_timesteps}, learning_rate={rl_learning_rate}")
    elif args.use_tuned_params:
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

# Prepare model weights dictionary with mode-specific labels
if args.mode == 'optimized':
    model_weights = {
        'Equal Weights (Benchmark)': (benchmark_weights, benchmark_dates),
        'Risk Parity + Momentum': markowitz_weights,
    }
    # Add LSTM model if available
    if lstm_weights is not None:
        model_weights['Multi-Horizon LSTM + Kelly'] = lstm_weights
else:
    # Both realistic and changeperiod use the same traditional models
    model_weights = {
        'Equal Weights (Benchmark)': (benchmark_weights, benchmark_dates),
        'Enhanced Markowitz': markowitz_weights,
        'Enhanced Markowitz + LSTM': lstm_weights,
    }

# Add RL model if available
if 'rl_weights' in locals() and rl_weights:
    model_weights['Reinforcement Learning (PPO)'] = (rl_weights, rl_dates)

# Run comprehensive comparison with mode-specific output
print("Running comprehensive backtesting and comparison...")
summary_df = validation.compare_models(model_weights, prices, returns, save_plots=True, 
                                      output_dir=output_dir, img_dir=img_dir)

# Save results with mode suffix
results_file = f'{output_dir}/performance_table_{args.mode}.csv'
summary_df.to_csv(results_file)

print(f"\nComparison complete! Results saved to {results_file}")
print(f"Plots saved to {img_dir}/ directory:")
print(f"- {img_dir}/cumulative_returns_{args.mode}.png")
print(f"- {img_dir}/portfolio_weights_{args.mode}.png") 
print(f"- {img_dir}/drawdown_{args.mode}.png")
print(f"- {img_dir}/performance_metrics_{args.mode}.png")
print(f"- {img_dir}/rolling_sharpe_{args.mode}.png")

# Display final summary
print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
print("All models have been trained, backtested, and compared.")
print("Check the generated plots and CSV file for detailed analysis.")





