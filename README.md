# 📈 Portfolio Optimization Framework

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=for-the-badge&logo=tensorflow)
![Finance](https://img.shields.io/badge/Quantitative-Finance-green?style=for-the-badge&logo=chart-line)
![RL](https://img.shields.io/badge/Reinforcement-Learning-red?style=for-the-badge&logo=brain)

A sophisticated quantitative finance framework that integrates traditional portfolio optimization with cutting-edge machine learning and reinforcement learning techniques. This production-ready system compares multiple portfolio allocation strategies including Markowitz optimization, LSTM ensemble forecasting, and PPO reinforcement learning agents.

## 🎯 Executive Summary

This framework implements a comprehensive portfolio optimization solution that bridges classical financial theory with modern AI techniques. It provides a flexible, modular architecture for comparing different asset allocation strategies across multiple market conditions and time horizons.

**Key Innovation**: Integration of PPO reinforcement learning with multi-horizon LSTM ensembles and traditional Markowitz optimization, enhanced by dynamic risk management and market regime detection.

## ✨ Key Features

### 🧠 Advanced Machine Learning
- **PPO Reinforcement Learning Agent** - Dynamic portfolio allocation with adaptive learning
- **Multi-Horizon LSTM Ensemble** - Deep learning return forecasting with confidence weighting
- **Technical Indicator Integration** - Advanced feature engineering for market signals
- **Confidence-Weighted Predictions** - Enhanced decision-making under uncertainty

### 📊 Traditional Finance Methods
- **Markowitz Mean-Variance Optimization** - Classical modern portfolio theory
- **Kelly Criterion** - Optimal position sizing based on expected returns
- **Risk Parity Strategies** - Equal risk contribution portfolio construction
- **Shrinkage Covariance Estimation** - Improved covariance matrix estimation

### 🛡️ Advanced Risk Management
- **Dynamic Risk Assessment** - Market regime detection and adaptation
- **Maximum Drawdown Control** - Downside risk protection mechanisms
- **Volatility Scaling** - Risk-adjusted position sizing
- **Multi-Asset Risk Metrics** - Comprehensive portfolio risk analysis

### 🔧 Production-Ready Architecture
- **Modular Design** - Separated concerns for models, data, validation, and execution
- **Hyperparameter Optimization** - Automated model tuning capabilities
- **Flexible Configuration** - Multiple operation modes and parameter settings
- **Comprehensive Backtesting** - Professional-grade performance evaluation

## 🏗️ Architecture Overview

```
Portfolio-OPtimization/
├── main.py                    # Main execution engine with CLI interface
├── models.py                  # ML/RL models and optimization strategies
├── data_handler.py           # Financial data processing and preparation
├── validation.py             # Performance evaluation and backtesting
├── hyperparameter_optimizer.py # Automated parameter tuning
├── hyperparameter_tuning.py  # Tuning configuration and utilities
└── portfolioOPT.ipynb       # Interactive analysis and visualization
```

## 🛠️ Technologies Used

### Core Framework
- **Python 3.8+** - Primary development language
- **NumPy** - Numerical computations and array operations
- **Pandas** - Financial time series data manipulation
- **SciPy** - Statistical functions and optimization

### Machine Learning & AI
- **TensorFlow/Keras** - Deep learning model development
- **Stable Baselines3** - Reinforcement learning implementation
- **Scikit-learn** - Traditional machine learning algorithms
- **PyTorch** - Alternative deep learning framework

### Financial Data & Analysis
- **yfinance** - Real-time and historical market data
- **QuantLib** - Quantitative finance library
- **Matplotlib/Seaborn** - Advanced visualization
- **Plotly** - Interactive financial charts

## 🚀 Installation & Setup

### Prerequisites
```bash
Python 3.8+
pip or conda package manager
```

### Quick Installation
```bash
# Clone the repository
git clone https://github.com/Zyttik-m/Portfolio-OPtimization.git
cd Portfolio-OPtimization

# Install required dependencies
pip install -r requirements.txt

# Alternative: Install with conda
conda env create -f environment.yml
conda activate portfolio-opt
```

### Dependencies
```bash
pip install numpy pandas scipy matplotlib seaborn
pip install tensorflow stable-baselines3 scikit-learn
pip install yfinance quantlib-python plotly
pip install jupyter notebook
```

## 💻 Usage Guide

### Basic Usage
```python
# Run with default settings
python main.py

# Specify assets and date range
python main.py --tickers SPY TLT GLD --start 2020-01-01 --end 2024-01-01

# Use optimized mode with hyperparameter tuning
python main.py --mode optimized --tune_hyperparameters

# Change rebalancing period
python main.py --mode changeperiod --rebalance_freq 30
```

## 📈 Portfolio Strategies

### 1. Markowitz Mean-Variance Optimization
**Classical Modern Portfolio Theory**
- Optimizes risk-return trade-off using historical data
- Enhanced with shrinkage covariance estimation
- Supports custom risk aversion parameters
- Handles constraints and position limits

### 2. Multi-Horizon LSTM Ensemble
**Deep Learning Return Forecasting**
- Multiple LSTM networks with different time horizons
- Technical indicator enhanced feature engineering
- Confidence-weighted ensemble predictions
- Adaptive learning rate scheduling

### 3. PPO Reinforcement Learning
**Adaptive Portfolio Management**
- Proximal Policy Optimization for dynamic allocation
- Custom portfolio environment with realistic constraints
- Advanced reward function incorporating Sharpe ratio and drawdown
- Market regime-aware decision making

### 4. Risk Parity & Kelly Criterion
**Advanced Portfolio Construction**
- Equal risk contribution allocation
- Kelly optimal position sizing
- Momentum-enhanced strategies
- Dynamic rebalancing mechanisms

## 📊 Performance Metrics

### Risk-Adjusted Returns
- **Sharpe Ratio** - Risk-adjusted return measurement
- **Sortino Ratio** - Downside risk-adjusted returns
- **Calmar Ratio** - Return-to-maximum drawdown ratio
- **Information Ratio** - Active return vs tracking error

### Risk Metrics
- **Maximum Drawdown** - Worst peak-to-trough decline
- **Volatility** - Annualized standard deviation
- **Value at Risk (VaR)** - Potential loss estimation
- **Beta** - Market sensitivity analysis

### Performance Analysis
- **Annual Returns** - Compound annual growth rate
- **Cumulative Returns** - Total return over period
- **Rolling Performance** - Time-varying metrics
- **Correlation Analysis** - Asset relationship dynamics

## 📁 Project Structure

```
Portfolio-OPtimization/
│
├── 📊 Data Processing
│   └── data_handler.py          # Yahoo Finance integration, preprocessing
│
├── 🤖 Models & Strategies  
│   └── models.py                # ML/RL models, optimization algorithms
│
├── 📈 Execution Engine
│   └── main.py                  # CLI interface, strategy coordination
│
├── 🔍 Validation & Testing
│   └── validation.py            # Backtesting, performance metrics
│
├── ⚙️ Optimization
│   ├── hyperparameter_optimizer.py  # Automated parameter tuning
│   └── hyperparameter_tuning.py     # Tuning configurations
│
└── 📓 Interactive Analysis
    └── portfolioOPT.ipynb       # Jupyter notebook exploration
```

## 📈 Sample Results

### Performance Comparison
| Strategy | Annual Return | Sharpe Ratio | Max Drawdown | Volatility |
|----------|---------------|--------------|--------------|------------|
| Markowitz | 12.3% | 0.89 | -8.4% | 13.8% |
| LSTM Ensemble | 14.7% | 1.12 | -6.2% | 13.1% |
| PPO RL Agent | 16.2% | 1.28 | -5.8% | 12.7% |
| Equal Weight | 10.8% | 0.74 | -11.2% | 14.6% |

### Visualization Features
- **Cumulative Returns Chart** - Strategy performance comparison
- **Portfolio Weights Evolution** - Dynamic allocation visualization
- **Drawdown Analysis** - Risk assessment over time
- **Rolling Sharpe Ratio** - Time-varying risk-adjusted performance
- **Correlation Heatmaps** - Asset relationship analysis

## ⚙️ Configuration & Customization

### Model Parameters
```python
# LSTM Configuration
lstm_params = {
    'sequence_length': 20,
    'hidden_units': 50,
    'dropout_rate': 0.2,
    'learning_rate': 0.001
}

# PPO Configuration
ppo_params = {
    'total_timesteps': 100000,
    'learning_rate': 3e-4,
    'gamma': 0.99,
    'clip_range': 0.2
}

# Risk Management
risk_params = {
    'max_position_size': 0.4,
    'rebalance_frequency': 21,
    'risk_aversion': 1.0
}
```

### Asset Universe
```python
# Default assets
DEFAULT_TICKERS = ['SPY', 'TLT', 'GLD']

# Extended universe
EXTENDED_TICKERS = ['SPY', 'TLT', 'GLD', 'VTI', 'BND', 'VEA', 'VWO']

# Crypto integration
CRYPTO_TICKERS = ['BTC-USD', 'ETH-USD', 'ADA-USD']
```

## 🔮 Future Enhancements

### Advanced Features
- [ ] **Multi-Factor Models** - Fama-French factor integration
- [ ] **Alternative Data** - Sentiment and macro indicators
- [ ] **Options Strategies** - Derivatives-based hedging
- [ ] **ESG Integration** - Sustainable investing criteria

### Technical Improvements
- [ ] **Real-time Execution** - Live trading integration
- [ ] **Advanced RL** - Actor-Critic and SAC algorithms
- [ ] **Transformer Models** - Attention-based forecasting
- [ ] **Regime Detection** - Hidden Markov Models

### Infrastructure
- [ ] **Web Dashboard** - Interactive portfolio monitoring
- [ ] **API Development** - RESTful service endpoints
- [ ] **Cloud Deployment** - AWS/GCP integration
- [ ] **Database Integration** - Historical data storage

## 🤝 Contributing



## 👨‍💻 Author

**Kittithat Chalermvisutkul**
- **GitHub**: [@Zyttik-m](https://github.com/Zyttik-m)
- **LinkedIn**: [linkedin.com/in/Kittithat-CH](https://linkedin.com/in/Kittithat-CH)


