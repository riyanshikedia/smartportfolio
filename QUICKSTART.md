# 🚀 Quick Start Guide

## For Any Laptop - Complete Setup

### Step 1: Clone and Environment Setup
```bash
git clone https://github.com/yourusername/SmartPortfolio.git
cd SmartPortfolio

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Database Setup
```bash
# Start SQL Server with Docker
docker run -e "ACCEPT_EULA=Y" -e "SA_PASSWORD=YourStrong@Passw0rd" \
  -p 1433:1433 --name sqlserver \
  -d mcr.microsoft.com/mssql/server:2022-latest

# Wait 30 seconds for SQL Server to initialize
```

### Step 3: Configuration
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

### Step 4: Initialize Database
```bash
# Create database and tables
python scripts/setup_database.py
```

### Step 5: Run Complete Analysis Pipeline
```bash
# Activate environment
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Run all scripts in sequence
python scripts/run_data_collection.py       # 1. Collect S&P 500 data
python scripts/run_technical_indicators.py  # 2. Calculate indicators
python scripts/run_screening.py             # 3. Screen stocks
python scripts/run_ml_prediction.py         # 4. ML predictions
python scripts/run_optimization.py          # 5. Optimize portfolio
python scripts/run_backtest.py              # 6. Backtest performance
```

## 🐳 Using Docker (Recommended)

If you have Docker installed, the setup script will automatically:
- Download and start SQL Server
- Create the database
- Set up all tables

## 🔧 Manual Database Setup

If you prefer to set up SQL Server manually:

1. **Install SQL Server** (any version 2019+)
2. **Create database**:
   ```sql
   CREATE DATABASE SmartPortfolio;
   ```
3. **Run setup script**:
   ```bash
   python scripts/setup_database.py
   ```

## 🧪 Test Your Setup

```bash
python test_setup.py
```

This will verify:
- ✅ All Python packages are installed
- ✅ Database connection works
- ✅ Data collection functions work

## 📊 View Results

After running the pipeline, open Jupyter Lab:
```bash
jupyter lab
```

Then open `notebooks/06_dashboard.ipynb` to see your results!

## 🆘 Need Help?

1. **Check the logs** - All scripts provide detailed output
2. **Verify .env file** - Make sure database credentials are correct
3. **Test database connection** - Run `python test_setup.py`
4. **Check README.md** - Full documentation available

## 🎯 What You'll Get

- **S&P 500 Analysis**: 500+ stocks analyzed
- **ML Predictions**: XGBoost return forecasting
- **Portfolio Optimization**: Markowitz mean-variance
- **Backtesting**: Historical performance analysis
- **Interactive Dashboard**: Plotly visualizations

Happy investing! 📈
