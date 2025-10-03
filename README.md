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
- Git

### 📋 Step-by-Step Setup

#### 1. Clone and Environment Setup
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

#### 2. Database Setup

**Option A: Using Docker (Recommended)**
```bash
# Start SQL Server with Docker
docker run -e "ACCEPT_EULA=Y" -e "SA_PASSWORD=YourStrong@Passw0rd" \
  -p 1433:1433 --name sqlserver \
  -d mcr.microsoft.com/mssql/server:2022-latest

# Wait 30 seconds for SQL Server to initialize
```

**Option B: Local SQL Server**
- Install SQL Server locally
- Update credentials in `.env` file

#### 3. Configuration
```bash
# Copy environment template
cp env.example .env

# Edit .env with your database credentials
# DB_HOST=localhost
# DB_PORT=1433
# DB_NAME=SmartPortfolio
# DB_USER=sa
# DB_PASSWORD=YourStrong@Passw0rd
```

#### 4. Initialize Database
```bash
# Create database and tables
python scripts/setup_database.py
```

### 🎯 Complete Analysis Pipeline

Run these scripts in sequence to perform the entire analysis:

```bash
# Activate virtual environment
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 1. Data Collection
python scripts/run_data_collection.py

# 2. Technical Indicators
python scripts/run_technical_indicators.py

# 3. Stock Screening
python scripts/run_screening.py

# 4. ML Predictions
python scripts/run_ml_prediction.py

# 5. Portfolio Optimization
python scripts/run_optimization.py

# 6. Backtesting
python scripts/run_backtest.py
```

### 📊 View Results

After running the pipeline, open Jupyter Lab to see interactive results:
```bash
jupyter lab
```

Then open `notebooks/06_dashboard.ipynb` to view your portfolio analysis!

### 🐛 Troubleshooting

**Database Connection Issues:**
- Ensure SQL Server is running
- Check credentials in `.env` file
- Verify port 1433 is accessible

**Permission Errors:**
- Run as administrator (Windows)
- Use `sudo` if needed (Linux/Mac)

**Docker Issues:**
- Install Docker Desktop
- Ensure Docker is running
- Check if port 1433 is available

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

