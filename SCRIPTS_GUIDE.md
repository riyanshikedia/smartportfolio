# 📋 Scripts Execution Guide

## Complete Analysis Pipeline

Run these scripts in the exact order shown to perform the entire SmartPortfolio analysis:

### 🔧 Prerequisites
```bash
# 1. Setup environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. Setup database (Docker)
docker run -e "ACCEPT_EULA=Y" -e "SA_PASSWORD=YourStrong@Passw0rd" \
  -p 1433:1433 --name sqlserver \
  -d mcr.microsoft.com/mssql/server:2022-latest

# 3. Configure environment
cp env.example .env
# Edit .env with your database credentials

# 4. Initialize database
python scripts/setup_database.py
```

### 🎯 Analysis Scripts (Run in Order)

#### 1. Data Collection
```bash
python scripts/run_data_collection.py
```
**What it does:**
- Scrapes S&P 500 tickers from Wikipedia
- Fetches 250 days of historical price data from Yahoo Finance
- Saves data to `market.sp500_tickers` and `market.daily_prices` tables

#### 2. Technical Indicators
```bash
python scripts/run_technical_indicators.py
```
**What it does:**
- Calculates technical indicators (RSI, MACD, Bollinger Bands, etc.)
- Computes price-based features for ML models
- Saves indicators to `market.technical_indicators` table

#### 3. Stock Screening
```bash
python scripts/run_screening.py
```
**What it does:**
- Applies Fama-French 5-Factor screening
- Scores stocks based on fundamental, momentum, technical, and risk factors
- Saves screening results to `fundamental.stock_scores` table

#### 4. ML Predictions
```bash
python scripts/run_ml_prediction.py
```
**What it does:**
- Trains XGBoost models for 1M and 3M return predictions
- Generates predictions for all screened stocks
- Saves predictions to `ml_models.return_predictions` table

#### 5. Portfolio Optimization
```bash
python scripts/run_optimization.py
```
**What it does:**
- Applies Markowitz mean-variance optimization
- Creates optimized portfolios based on ML predictions
- Saves portfolio weights to `portfolio.optimized_weights` table

#### 6. Backtesting
```bash
python scripts/run_backtest.py
```
**What it does:**
- Runs historical backtest of optimized portfolio
- Calculates performance metrics (Sharpe ratio, max drawdown, etc.)
- Generates performance reports and visualizations

### 📊 View Results

After running all scripts:
```bash
# Open Jupyter Lab
jupyter lab

# Then open notebooks/06_dashboard.ipynb to see results
```

### 🔄 Re-running Scripts

- **Safe to re-run:** All scripts can be run multiple times
- **Data refresh:** Re-run `run_data_collection.py` to get latest data
- **Full pipeline:** Re-run all scripts for complete refresh

### ⚠️ Troubleshooting

**If a script fails:**
1. Check the error message
2. Ensure previous scripts completed successfully
3. Verify database connection in `.env` file
4. Check if required tables exist

**Common issues:**
- Database connection: Verify SQL Server is running
- Missing data: Ensure previous scripts completed
- Memory issues: Close other applications

### 📈 Expected Results

After running all scripts, you should have:
- ✅ 500+ S&P 500 stocks analyzed
- ✅ Technical indicators calculated
- ✅ ML predictions generated
- ✅ Optimized portfolio created
- ✅ Backtest results available
- ✅ Interactive dashboard ready

### 🎯 Next Steps

1. **Review Results:** Open the dashboard notebook
2. **Customize Parameters:** Modify script parameters as needed
3. **Schedule Runs:** Set up automated runs for regular updates
4. **Extend Analysis:** Add new features or models
