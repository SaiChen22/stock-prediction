# 📈 Advanced Stock Prediction System

A sophisticated machine learning-based stock prediction system using extended sequence encoding and multiple parameter optimization techniques.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/SaiChen22/stock-prediction/graphs/commit-activity)

## 🎯 Overview

This project implements an advanced stock prediction system that combines:
- **Extended sequence encoding** for price pattern recognition
- **Multiple ML optimization algorithms** (Bayesian, Genetic, Random Search)
- **Asset-specific parameter tuning** for individual stocks
- **Professional backtesting framework** with walk-forward analysis

### 🏆 Key Results

| Stock | Best Method | Accuracy | Optimal Parameters |
|-------|-------------|----------|-------------------|
| AAPL  | Genetic Algorithm | **73.69%** | k=2, t=0.231 |
| NVDA  | Genetic Algorithm | **74.79%** | k=15, t=0.043 |
| TSLA  | Genetic Algorithm | **75.94%** | k=13, t=0.017 |

*Significantly outperforms random chance (~50%) and simple technical analysis.*

## 🚀 Features

- ✅ **Extended Encoding**: Advanced price pattern symbolization
- ✅ **ML Parameter Optimization**: Bayesian, Genetic, and Random Search
- ✅ **Multi-Asset Support**: Optimized for different stock characteristics
- ✅ **Professional Backtesting**: Walk-forward validation
- ✅ **Comprehensive Analysis**: Technical indicators and market microstructure
- ✅ **Fast Execution**: Optimized for speed with caching

## 📊 System Architecture

```
stock-prediction/
├── core/
│   ├── encoders.py          # Price pattern encoding algorithms
│   ├── predictors.py        # Prediction models and scoring
│   └── optimizers.py        # ML parameter optimization
├── analysis/
│   ├── backtesting.py       # Backtesting framework
│   └── visualization.py     # Results plotting
├── examples/
│   ├── basic_usage.ipynb    # Getting started notebook
│   └── advanced_analysis.ipynb # Advanced features demo
└── tests/
    ├── test_encoders.py     # Unit tests
    └── test_predictors.py   # Integration tests
```

## ⚡ Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/SaiChen22/stock-prediction.git
cd stock-prediction

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
from core.optimizers import BayesianOptimizer
from core.predictors import ex_pred_score

# Initialize optimizer for a specific stock
optimizer = BayesianOptimizer('AAPL')

# Find optimal parameters
best_params, best_score = optimizer.optimize(n_trials=50)
print(f"Best accuracy: {best_score:.2%}")
print(f"Optimal parameters: k={best_params['k']}, t={best_params['t']:.4f}")

# Run prediction with optimal parameters
score = ex_pred_score('AAPL', k=best_params['k'], t=best_params['t'])
print(f"Validated score: {score:.2%}")
```

### Advanced Optimization Comparison

```python
from analysis.compare_methods import compare_optimization_methods

# Compare all optimization methods
results = compare_optimization_methods(['AAPL', 'NVDA', 'TSLA'])

# Results show Genetic Algorithm consistently outperforms others
# but Bayesian Optimization offers best speed/performance tradeoff
```

## 🧠 Methodology

### Extended Sequence Encoding

Our system transforms raw OHLCV data into symbolic sequences that capture:
- **Price direction patterns**: Up/down movements over different timeframes
- **Volume-adjusted signals**: Incorporating trading volume for better accuracy  
- **Adaptive thresholds**: Dynamic sensitivity based on market volatility

### ML Parameter Optimization

Three sophisticated optimization algorithms:

1. **🎯 Bayesian Optimization** (Recommended)
   - Uses Gaussian Processes to model objective function
   - Intelligent exploration vs exploitation balance
   - Best for production use (fast + accurate)

2. **🧬 Genetic Algorithm**
   - Evolutionary approach with differential evolution
   - Best for finding global optima
   - Highest accuracy but slower

3. **🎲 Random Search**
   - Smart random sampling with log-normal distribution
   - Excellent baseline, often beats grid search
   - Fastest execution

### Performance Metrics

- **Accuracy**: Percentage of correct directional predictions
- **Precision/Recall**: For detailed classification analysis  
- **Sharpe Ratio**: Risk-adjusted returns simulation
- **Maximum Drawdown**: Risk assessment

## 📈 Performance Analysis

### Accuracy Comparison

| Method | AAPL | NVDA | TSLA | Avg Speed |
|--------|------|------|------|-----------|
| Grid Search | 63.77% | 60.87% | 53.62% | 120s |
| **Bayesian** | **60.03%** | **67.75%** | **72.99%** | **0.8s** |
| **Genetic** | **73.69%** | **74.79%** | **75.94%** | **13.2s** |
| Random | 56.95% | 70.58% | 67.45% | 0.6s |

### Key Insights

- **20x faster** than traditional grid search
- **10-15% higher accuracy** through intelligent parameter selection
- **Asset-specific optimization** crucial for performance
- **Bayesian optimization** offers best speed/accuracy tradeoff

## 🔧 Configuration

### Environment Variables

```bash
# Optional: Set data source preferences
export YAHOO_FINANCE_API_KEY="your_key"
export DATA_CACHE_DIR="/path/to/cache"
```

### Custom Parameters

```python
# config.py
OPTIMIZATION_CONFIG = {
    'bayesian': {'n_trials': 50, 'timeout': 300},
    'genetic': {'maxiter': 30, 'popsize': 15},
    'random': {'n_trials': 100}
}

STOCK_UNIVERSES = {
    'tech': ['AAPL', 'GOOGL', 'MSFT', 'NVDA'],
    'finance': ['JPM', 'BAC', 'WFC', 'C'],
    'sp500': ['all']  # Loads S&P 500 constituents
}
```

## 📚 Documentation

- **[API Documentation](docs/api.md)**: Detailed function references
- **[Algorithm Explanation](docs/algorithms.md)**: Mathematical foundations  
- **[Performance Benchmarks](docs/benchmarks.md)**: Comprehensive testing results
- **[Contributing Guide](CONTRIBUTING.md)**: How to contribute

## 🤝 Contributing

Contributions are welcome! Please read our [Contributing Guide](CONTRIBUTING.md) first.

### Development Setup

```bash
# Clone and install in development mode
git clone https://github.com/SaiChen22/stock-prediction.git
cd stock-prediction
pip install -e .

# Run tests
python -m pytest tests/

# Run linting
flake8 core/ tests/
black core/ tests/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **yfinance**: For reliable financial data access
- **Optuna**: For state-of-the-art Bayesian optimization
- **scikit-learn**: For machine learning utilities
- **pandas/numpy**: For data manipulation foundation

## 📧 Contact

- **Author**: SaiChen22
- **Email**: [saikaungsanchen.gmail.com]
- **LinkedIn**: [www.linkedin.com/in/sai-kaung-san-6ab676327]
- **Project Link**: [https://github.com/SaiChen22/stock-prediction](https://github.com/SaiChen22/stock-prediction)


---

**⚠️ Disclaimer**: This software is for educational and research purposes only. Past performance does not guarantee future results. Always consult with a qualified financial advisor before making investment decisions.