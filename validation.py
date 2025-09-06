import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid displaying plots
import seaborn as sns
from typing import List, Tuple, Dict
import warnings
import os
warnings.filterwarnings('ignore')

# Set style for professional plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def calculate_portfolio_returns(weights: List[np.ndarray], 
                               dates: List, 
                               returns_df: pd.DataFrame,
                               initial_value: float = 1000000) -> pd.DataFrame:
    """
    Calculate portfolio returns based on weights and asset returns.
    
    Args:
        weights: List of weight arrays for each rebalancing period
        dates: List of rebalancing dates
        returns_df: DataFrame of asset returns
        initial_value: Initial portfolio value
        
    Returns:
        DataFrame with portfolio values and returns
    """
    if not weights or not dates:
        # Return empty DataFrame with expected structure
        return pd.DataFrame({
            'portfolio_value': [],
            'returns': []
        })
    
    # Clean returns data - remove any NaN values
    returns_clean = returns_df.dropna()
    
    portfolio_values = [initial_value]
    portfolio_returns = []
    current_weights = None
    weight_idx = 0
    
    # Find the start date (first rebalancing date)
    start_date = dates[0] if dates else returns_clean.index[0]
    start_idx = returns_clean.index.get_loc(start_date) if start_date in returns_clean.index else 0
    
    for i in range(start_idx, len(returns_clean)):
        current_date = returns_clean.index[i]
        
        # Update weights if we're at a rebalancing date
        if weight_idx < len(dates) and current_date >= dates[weight_idx]:
            current_weights = weights[weight_idx]
            weight_idx += 1
        
        # Skip if we don't have weights yet
        if current_weights is None:
            continue
            
        # Calculate portfolio return for this period
        period_returns = returns_clean.iloc[i].values
        
        # Clean any NaN or infinite values
        period_returns = np.nan_to_num(period_returns, nan=0.0, posinf=0.0, neginf=0.0)
        current_weights = np.nan_to_num(current_weights, nan=1.0/len(current_weights))
        
        portfolio_return = np.sum(period_returns * current_weights)
        
        # Ensure portfolio return is finite
        if not np.isfinite(portfolio_return):
            portfolio_return = 0.0
            
        portfolio_returns.append(portfolio_return)
        
        # Update portfolio value
        new_value = portfolio_values[-1] * (1 + portfolio_return)
        portfolio_values.append(new_value)
    
    if not portfolio_returns:
        # Return empty DataFrame if no returns calculated
        return pd.DataFrame({
            'portfolio_value': [],
            'returns': []
        })
    
    # Create results DataFrame starting from where we have data
    start_result_idx = start_idx
    end_result_idx = start_result_idx + len(portfolio_returns)
    
    results_df = pd.DataFrame({
        'portfolio_value': portfolio_values[1:],
        'returns': portfolio_returns
    }, index=returns_clean.index[start_result_idx:end_result_idx])
    
    return results_df


def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02, mode: str = 'realistic') -> float:
    """
    Calculate annualized Sharpe ratio.
    
    Args:
        returns: Series of portfolio returns
        risk_free_rate: Annual risk-free rate (default 2% for realistic, 0% for optimized)
        mode: 'realistic' or 'optimized' for different calculation methods
        
    Returns:
        Annualized Sharpe ratio
    """
    # Assuming daily returns
    trading_days = 252
    
    # Adjust risk-free rate based on mode
    if mode == 'optimized':
        risk_free_rate = 0.0  # Use 0% for excess returns in optimized mode
    
    # Calculate annualized metrics
    annual_return = returns.mean() * trading_days
    annual_volatility = returns.std() * np.sqrt(trading_days)
    
    # Calculate Sharpe ratio
    if annual_volatility == 0:
        return 0
    
    sharpe = (annual_return - risk_free_rate) / annual_volatility
    return sharpe


def calculate_annual_return(portfolio_values: pd.Series) -> float:
    """
    Calculate annualized return.
    
    Args:
        portfolio_values: Series of portfolio values
        
    Returns:
        Annualized return as percentage
    """
    if len(portfolio_values) < 2:
        return 0.0
    
    # Handle NaN values
    clean_values = portfolio_values.dropna()
    if len(clean_values) < 2:
        return 0.0
    
    total_return = (clean_values.iloc[-1] / clean_values.iloc[0]) - 1
    
    # Handle infinite or NaN total return
    if not np.isfinite(total_return):
        return 0.0
        
    n_years = len(clean_values) / 252  # Assuming daily data
    
    if n_years <= 0:
        return 0.0
    
    # Calculate annualized return
    annual_return = (1 + total_return) ** (1/n_years) - 1
    
    # Ensure result is finite
    if not np.isfinite(annual_return):
        return 0.0
        
    return annual_return * 100  # Return as percentage


def calculate_cumulative_return(portfolio_values: pd.Series) -> float:
    """
    Calculate total cumulative return.
    
    Args:
        portfolio_values: Series of portfolio values
        
    Returns:
        Cumulative return as percentage
    """
    if len(portfolio_values) < 2:
        return 0.0
    
    # Handle NaN values
    clean_values = portfolio_values.dropna()
    if len(clean_values) < 2:
        return 0.0
    
    cumulative_return = (clean_values.iloc[-1] / clean_values.iloc[0]) - 1
    
    # Ensure result is finite
    if not np.isfinite(cumulative_return):
        return 0.0
        
    return cumulative_return * 100  # Return as percentage


def calculate_volatility(returns: pd.Series) -> float:
    """
    Calculate annualized volatility.
    
    Args:
        returns: Series of portfolio returns
        
    Returns:
        Annualized volatility as percentage
    """
    # Assuming daily returns
    trading_days = 252
    annual_volatility = returns.std() * np.sqrt(trading_days)
    return annual_volatility * 100  # Return as percentage


def calculate_max_drawdown(portfolio_values: pd.Series) -> Tuple[float, pd.Timestamp, pd.Timestamp]:
    """
    Calculate maximum drawdown and dates.
    
    Args:
        portfolio_values: Series of portfolio values
        
    Returns:
        Tuple of (max_drawdown percentage, peak_date, trough_date)
    """
    # Calculate cumulative returns
    cumulative_returns = portfolio_values / portfolio_values.iloc[0]
    
    # Calculate running maximum
    running_max = cumulative_returns.expanding().max()
    
    # Calculate drawdown
    drawdown = (cumulative_returns - running_max) / running_max
    
    # Find maximum drawdown
    max_drawdown = drawdown.min()
    
    # Find the dates
    if max_drawdown < 0:
        trough_date = drawdown.idxmin()
        peak_date = cumulative_returns[:trough_date].idxmax()
    else:
        peak_date = portfolio_values.index[0]
        trough_date = portfolio_values.index[0]
    
    return max_drawdown * 100, peak_date, trough_date  # Return as percentage


def calculate_avg_weights(weights: List[np.ndarray], asset_names: List[str]) -> Dict[str, float]:
    """
    Calculate average portfolio weights for each asset.
    
    Args:
        weights: List of weight arrays
        asset_names: List of asset names
        
    Returns:
        Dictionary with average weight for each asset
    """
    if not weights:
        return {name: 0 for name in asset_names}
    
    weights_array = np.array(weights)
    avg_weights = weights_array.mean(axis=0)
    
    return {name: weight for name, weight in zip(asset_names, avg_weights)}


def backtest_portfolio(weights: List[np.ndarray], 
                       dates: List,
                       prices_df: pd.DataFrame,
                       returns_df: pd.DataFrame,
                       model_name: str,
                       initial_value: float = 1000000,
                       mode: str = 'realistic') -> Dict:
    """
    Backtest a portfolio strategy and calculate all metrics.
    
    Args:
        weights: List of weight arrays
        dates: List of rebalancing dates
        prices_df: DataFrame of asset prices
        returns_df: DataFrame of asset returns
        model_name: Name of the model
        initial_value: Initial portfolio value
        
    Returns:
        Dictionary with all performance metrics
    """
    # Calculate portfolio performance
    portfolio_df = calculate_portfolio_returns(weights, dates, returns_df, initial_value)
    
    # Handle empty portfolio data
    if len(portfolio_df) == 0:
        print(f"Warning: No portfolio data calculated for {model_name}")
        # Return default values
        metrics = {
            'Model': model_name,
            'Sharpe Ratio': 0.0,
            'Annual Return (%)': 0.0,
            'Cumulative Return (%)': 0.0,
            'Volatility (%)': 0.0,
            'Max Drawdown (%)': 0.0,
            'Final Value': initial_value,
            'Number of Rebalances': len(weights) if weights else 0
        }
        
        # Add average weights
        avg_weights = calculate_avg_weights(weights, list(prices_df.columns))
        for asset, weight in avg_weights.items():
            metrics[f'Avg Weight {asset}'] = weight
        
        # Store empty data for plotting
        metrics['portfolio_df'] = portfolio_df
        metrics['weights_history'] = weights
        metrics['dates'] = dates
        
        return metrics
    
    # Apply volatility scaling for optimized mode
    portfolio_returns = portfolio_df['returns'].copy()
    if mode == 'optimized' and len(portfolio_returns) > 0:
        # Import the scaling function
        import models
        portfolio_returns = models.apply_volatility_scaling(portfolio_returns, target_vol=0.15)
        
        # Recalculate portfolio values with scaled returns
        scaled_portfolio_values = (1 + portfolio_returns).cumprod() * 1000000
    else:
        scaled_portfolio_values = portfolio_df['portfolio_value']
    
    # Calculate all metrics using potentially scaled values
    final_value = scaled_portfolio_values.iloc[-1] if len(scaled_portfolio_values) > 0 else initial_value
    
    metrics = {
        'Model': model_name,
        'Sharpe Ratio': calculate_sharpe_ratio(portfolio_returns, mode=mode),
        'Annual Return (%)': calculate_annual_return(scaled_portfolio_values),
        'Cumulative Return (%)': calculate_cumulative_return(scaled_portfolio_values),
        'Volatility (%)': calculate_volatility(portfolio_returns),
        'Max Drawdown (%)': calculate_max_drawdown(scaled_portfolio_values)[0] if len(scaled_portfolio_values) > 0 else 0.0,
        'Final Value': final_value if np.isfinite(final_value) else initial_value,
        'Number of Rebalances': len(weights) if weights else 0
    }
    
    # Add average weights
    avg_weights = calculate_avg_weights(weights, list(prices_df.columns))
    for asset, weight in avg_weights.items():
        metrics[f'Avg Weight {asset}'] = weight
    
    # Store portfolio data for plotting
    metrics['portfolio_df'] = portfolio_df
    metrics['weights_history'] = weights
    metrics['dates'] = dates
    
    return metrics


def plot_cumulative_returns(results_dict: Dict[str, Dict], save_path: str = 'img/cumulative_returns.png'):
    """
    Plot cumulative returns for all models.
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.figure(figsize=(14, 8))
    
    for model_name, metrics in results_dict.items():
        portfolio_df = metrics['portfolio_df']
        
        # Skip if no data
        if len(portfolio_df) == 0:
            continue
            
        cumulative_returns = (portfolio_df['portfolio_value'] / portfolio_df['portfolio_value'].iloc[0] - 1) * 100
        plt.plot(cumulative_returns.index, cumulative_returns.values, 
                label=f"{model_name} ({metrics['Cumulative Return (%)']:.2f}%)", 
                linewidth=2, alpha=0.8)
    
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Cumulative Return (%)', fontsize=12)
    plt.title('Portfolio Cumulative Returns Comparison', fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()  # Close instead of show
    print(f"Cumulative returns plot saved to {save_path}")


def plot_portfolio_weights(results_dict: Dict[str, Dict], asset_names: List[str], save_path: str = 'img/portfolio_weights.png'):
    """
    Plot portfolio weight allocations over time for all models.
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Filter out models with no data
    valid_models = {name: metrics for name, metrics in results_dict.items() 
                   if metrics['weights_history'] and len(metrics['weights_history']) > 0}
    
    if not valid_models:
        print("No valid weight data for plotting")
        return
    
    n_models = len(valid_models)
    fig, axes = plt.subplots(n_models, 1, figsize=(14, 5*n_models))
    
    if n_models == 1:
        axes = [axes]
    
    for idx, (model_name, metrics) in enumerate(valid_models.items()):
        weights = metrics['weights_history']
        dates = metrics['dates']
        
        # Create DataFrame for weights
        weights_df = pd.DataFrame(weights, index=dates, columns=asset_names)
        
        # Create stacked area plot
        ax = axes[idx]
        weights_df.plot.area(stacked=True, ax=ax, alpha=0.7)
        
        ax.set_xlabel('Date', fontsize=10)
        ax.set_ylabel('Weight', fontsize=10)
        ax.set_title(f'{model_name} - Portfolio Weight Allocation', fontsize=12, fontweight='bold')
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=9)
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()  # Close instead of show
    print(f"Portfolio weights plot saved to {save_path}")


def plot_drawdown(results_dict: Dict[str, Dict], save_path: str = 'img/drawdown.png'):
    """
    Plot drawdown charts for all models.
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.figure(figsize=(14, 8))
    
    for model_name, metrics in results_dict.items():
        portfolio_df = metrics['portfolio_df']
        
        # Skip if no data
        if len(portfolio_df) == 0:
            continue
        
        # Calculate drawdown
        cumulative_returns = portfolio_df['portfolio_value'] / portfolio_df['portfolio_value'].iloc[0]
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max * 100
        
        plt.fill_between(drawdown.index, 0, drawdown.values, 
                         alpha=0.3, label=f"{model_name} (Max: {metrics['Max Drawdown (%)']:.2f}%)")
        plt.plot(drawdown.index, drawdown.values, linewidth=1, alpha=0.8)
    
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Drawdown (%)', fontsize=12)
    plt.title('Portfolio Drawdown Comparison', fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()  # Close instead of show
    print(f"Drawdown plot saved to {save_path}")


def plot_performance_metrics(results_dict: Dict[str, Dict], save_path: str = 'img/performance_metrics.png'):
    """
    Create bar charts comparing key performance metrics.
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Extract metrics for plotting
    metrics_to_plot = ['Sharpe Ratio', 'Annual Return (%)', 'Volatility (%)', 'Max Drawdown (%)']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, metric in enumerate(metrics_to_plot):
        ax = axes[idx]
        
        models = list(results_dict.keys())
        values = [results_dict[model][metric] for model in models]
        
        # Create bar plot
        bars = ax.bar(models, values, alpha=0.7)
        
        # Color bars based on value (green for good, red for bad)
        if metric in ['Sharpe Ratio', 'Annual Return (%)']:
            colors = ['green' if v > 0 else 'red' for v in values]
        else:  # For Volatility and Max Drawdown, lower is better
            colors = ['red' if abs(v) > np.median(np.abs(values)) else 'green' for v in values]
        
        for bar, color in zip(bars, colors):
            bar.set_color(color)
            bar.set_alpha(0.6)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.2f}',
                   ha='center', va='bottom' if height >= 0 else 'top',
                   fontsize=10)
        
        ax.set_title(metric, fontsize=12, fontweight='bold')
        ax.set_ylabel('Value', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    plt.suptitle('Portfolio Performance Metrics Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()  # Close instead of show
    print(f"Performance metrics plot saved to {save_path}")


def plot_rolling_sharpe(results_dict: Dict[str, Dict], window: int = 60, save_path: str = 'img/rolling_sharpe.png'):
    """
    Plot rolling Sharpe ratio for all models.
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.figure(figsize=(14, 8))
    
    for model_name, metrics in results_dict.items():
        portfolio_df = metrics['portfolio_df']
        
        # Skip if no data
        if len(portfolio_df) == 0:
            continue
        
        # Calculate rolling Sharpe ratio
        rolling_returns_mean = portfolio_df['returns'].rolling(window=window).mean() * 252
        rolling_returns_std = portfolio_df['returns'].rolling(window=window).std() * np.sqrt(252)
        rolling_sharpe = (rolling_returns_mean - 0.02) / rolling_returns_std
        
        plt.plot(rolling_sharpe.index, rolling_sharpe.values, 
                label=model_name, linewidth=2, alpha=0.8)
    
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Rolling Sharpe Ratio', fontsize=12)
    plt.title(f'Rolling Sharpe Ratio Comparison ({window}-day window)', fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    plt.axhline(y=1, color='green', linestyle='--', linewidth=0.5, alpha=0.5, label='Good (>1)')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()  # Close instead of show
    print(f"Rolling Sharpe ratio plot saved to {save_path}")


def create_performance_table(results_dict: Dict[str, Dict], save_path: str = 'performance_table.csv'):
    """
    Create a summary table of all performance metrics.
    """
    # Prepare data for table
    table_data = []
    
    for model_name, metrics in results_dict.items():
        row = {
            'Model': model_name,
            'Sharpe Ratio': f"{metrics['Sharpe Ratio']:.3f}",
            'Annual Return (%)': f"{metrics['Annual Return (%)']:.2f}" if np.isfinite(metrics['Annual Return (%)']) else "N/A",
            'Cumulative Return (%)': f"{metrics['Cumulative Return (%)']:.2f}" if np.isfinite(metrics['Cumulative Return (%)']) else "N/A",
            'Volatility (%)': f"{metrics['Volatility (%)']:.2f}" if np.isfinite(metrics['Volatility (%)']) else "N/A",
            'Max Drawdown (%)': f"{metrics['Max Drawdown (%)']:.2f}" if np.isfinite(metrics['Max Drawdown (%)']) else "N/A",
            'Final Value ($)': f"{metrics['Final Value']:,.0f}" if np.isfinite(metrics['Final Value']) else "N/A",
            'Rebalances': metrics['Number of Rebalances']
        }
        
        # Add average weights
        for asset in ['SPY', 'TLT', 'GLD']:
            if f'Avg Weight {asset}' in metrics and np.isfinite(metrics[f'Avg Weight {asset}']):
                row[f'Avg {asset} (%)'] = f"{metrics[f'Avg Weight {asset}']*100:.1f}"
            else:
                row[f'Avg {asset} (%)'] = "N/A"
        
        table_data.append(row)
    
    # Create DataFrame
    summary_df = pd.DataFrame(table_data)
    
    # Save to CSV
    summary_df.to_csv(save_path, index=False)
    
    # Display table
    print("\n" + "="*100)
    print("PORTFOLIO PERFORMANCE SUMMARY")
    print("="*100)
    print(summary_df.to_string(index=False))
    print("="*100)
    
    return summary_df


def compare_models(model_weights: Dict[str, Tuple[List, List]], 
                   prices_df: pd.DataFrame,
                   returns_df: pd.DataFrame,
                   save_plots: bool = True,
                   output_dir: str = None,
                   img_dir: str = None) -> pd.DataFrame:
    """
    Main function to compare all portfolio models.
    
    Args:
        model_weights: Dictionary with model names as keys and (weights, dates) tuples as values
        prices_df: DataFrame of asset prices
        returns_df: DataFrame of asset returns
        save_plots: Whether to save plots to files
        output_dir: Directory for saving CSV results (optional)
        img_dir: Directory for saving plots (optional)
        
    Returns:
        DataFrame with performance summary
    """
    results = {}
    
    # Determine mode from directories (if provided)
    mode = 'optimized' if img_dir and 'optimized' in img_dir else 'realistic'
    
    # Backtest each model
    for model_name, (weights, dates) in model_weights.items():
        if not weights:  # Skip if no weights
            print(f"Skipping {model_name} - no weights available")
            continue
            
        print(f"\nBacktesting {model_name}...")
        metrics = backtest_portfolio(weights, dates, prices_df, returns_df, model_name, mode=mode)
        results[model_name] = metrics
    
    if not results:
        print("No models to compare!")
        return pd.DataFrame()
    
    # Set default directories if not provided
    if img_dir is None:
        img_dir = 'img'
    if output_dir is None:
        output_dir = '.'
    
    # Create all visualizations
    if save_plots:
        print("\nGenerating visualizations...")
        # Determine mode suffix from directory name
        mode_suffix = '_optimized' if 'optimized' in img_dir else '_realistic' if 'realistic' in img_dir else ''
        
        plot_cumulative_returns(results, save_path=f'{img_dir}/cumulative_returns{mode_suffix}.png')
        plot_portfolio_weights(results, list(prices_df.columns), save_path=f'{img_dir}/portfolio_weights{mode_suffix}.png')
        plot_drawdown(results, save_path=f'{img_dir}/drawdown{mode_suffix}.png')
        plot_performance_metrics(results, save_path=f'{img_dir}/performance_metrics{mode_suffix}.png')
        plot_rolling_sharpe(results, save_path=f'{img_dir}/rolling_sharpe{mode_suffix}.png')
    
    # Create and display summary table
    summary_df = create_performance_table(results)
    
    return summary_df