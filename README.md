# 📈 SmartPortfolio AI

**AI-Powered Quantitative Portfolio Management System**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A production-ready quantitative investment platform combining Modern Portfolio Theory, machine learning, and multi-factor analysis to build optimized equity portfolios.

## 🌟 Features

- 📊 **Data Collection**: S&P 500 data from Yahoo Finance
- 🔍 **Multi-Factor Screening**: Fama-French 5-Factor Model
- 🧠 **ML Predictions**: XGBoost for return forecasting
- ⚖️ **Portfolio Optimization**: Markowitz Mean-Variance
- 📉 **Backtesting**: Historical analysis + Monte Carlo simulation
- 📊 **Interactive Dashboard**: Plotly visualizations

## 📊 Performance (6-Month Backtest)

```
Total Return:        +25.09%
Sharpe Ratio:        2.401  ⭐⭐⭐⭐⭐
Annual Volatility:   17.89%
Max Drawdown:        -11.76%
Beta:                0.773 (Defensive)
```

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- SQL Server (Docker or local)
- 4GB+ RAM

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/SmartPortfolio.git
cd SmartPortfolio

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Configuration

1. **Create `.env` file**:
   ```bash
   cp .env.example .env
   ```

2. **Edit `.env` with your credentials**:
   ```env
   DB_HOST=localhost
   DB_PORT=1433
   DB_NAME=SmartPortfolio
   DB_USER=Yourusername
   DB_PASSWORD=YourStrong@Passw0rd
   ```

3. **Start SQL Server (Docker)**:
   ```bash
   docker run -e "ACCEPT_EULA=Y" -e "SA_PASSWORD=YourStrong@Passw0rd" \
     -p 1433:1433 --name sqlserver \
     -d mcr.microsoft.com/mssql/server:2022-latest
   ```

### Run Analysis

```bash
# Run complete pipeline locally
python scripts/run_data_collection.py       # 1. Collect S&P 500 data
python scripts/run_technical_indicators.py  # 2. Calculate indicators
python scripts/run_screening.py             # 3. Screen stocks
python scripts/run_ml_prediction.py         # 4. ML predictions
python scripts/run_optimization.py          # 5. Optimize portfolio
python scripts/run_backtest.py              # 6. Backtest performance
```

## 📁 Project Structure

```
SmartPortfolio/
├── notebooks/              # Analysis notebooks
│   ├── 01_data_collection.ipynb
│   ├── 02_stock_screening.ipynb
│   ├── 03_return_prediction.ipynb
│   ├── 04_portfolio_optimization.ipynb
│   ├── 05_performance_simulation.ipynb
│   └── 06_dashboard.ipynb
│
├── scripts/                # Python automation scripts
├── utils/                  # Utility modules
├── tests/                  # Unit tests
├── database/               # SQL schema
├── data/                   # Data storage (gitignored)
└── models/                 # Trained models (gitignored)
```

## 🛠️ Methodology

### Stock Screening (Fama-French 5-Factor)
- **Fundamental (40%)**: P/E, ROE, Profit Margin
- **Momentum (25%)**: 1M, 3M, 6M Returns
- **Technical (20%)**: RSI, MACD, Trend Signals
- **Risk (15%)**: Volatility, Sharpe, Beta

### Machine Learning (XGBoost)
- **Features**: 34 technical & price-based features
- **Targets**: 1M & 3M forward returns
- **Training**: 75,000+ samples from 503 stocks
- **Performance**: R² = 0.43 (1M), 0.39 (3M)

### Portfolio Optimization (Modern Portfolio Theory)
- **Objective**: Maximize Sharpe Ratio
- **Constraints**: Max 10% per stock, fully invested, long-only

## 🧪 Testing

```bash
# Run tests
pytest tests/ -v

# Format code
black utils/ tests/

# Lint code
flake8 utils/ tests/
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

##  Disclaimer

**This software is for educational and research purposes only.**

- Not financial advice
- Past performance does not guarantee future results
- Investing involves risk of loss
- Consult a financial advisor before investing

## 📄 License

MIT License - See [LICENSE](LICENSE) for details.

