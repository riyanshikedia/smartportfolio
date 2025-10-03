#!/usr/bin/env python3
"""
Script 5: Portfolio Optimization
Optimize portfolio allocation using Modern Portfolio Theory
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from datetime import datetime
from scipy.optimize import minimize

from utils.database_connector import DatabaseConnector
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
CONFIG = {
    'INITIAL_CAPITAL': 5000,
    'MAX_POSITION': 0.10,  # 10% max per stock
    'MIN_POSITION': 0.005,  # 0.5% min per stock
    'RISK_FREE_RATE': 0.045,  # 4.5% annual
}

def load_ml_predictions():
    """Load latest ML predictions"""
    print("\n📥 Loading ML predictions...")
    
    data_dir = project_root / 'data' / 'processed'
    
    # Find latest predictions file
    pred_files = sorted(data_dir.glob('ml_top_*.csv'), reverse=True)
    
    if not pred_files:
        raise FileNotFoundError("No ML prediction files found. Run 'run_ml_prediction.py' first.")
    
    latest_file = pred_files[0]
    df = pd.read_csv(latest_file)
    
    print(f"✅ Loaded {len(df)} stocks from {latest_file.name}")
    
    return df

def load_historical_returns(db, tickers, days=120):
    """Load historical returns for selected stocks"""
    print(f"\n📥 Loading {days} days of historical returns...")
    
    tickers_str = "', '".join(tickers)
    
    query = f"""
    WITH LatestDate AS (
        SELECT MAX(date) as max_date FROM market.daily_prices
    ),
    PriceData AS (
        SELECT 
            ticker,
            date,
            [close] as price,
            ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY date DESC) as rn
        FROM market.daily_prices
        WHERE ticker IN ('{tickers_str}')
    )
    SELECT ticker, date, price
    FROM PriceData
    WHERE rn <= {days}
    ORDER BY ticker, date
    """
    
    df = db.execute_query(query)
    
    # Convert to numeric
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    
    # Calculate returns
    returns = df.pivot(index='date', columns='ticker', values='price').pct_change().dropna()
    
    print(f"✅ Loaded returns for {len(returns.columns)} stocks, {len(returns)} days")
    
    return returns

def calculate_portfolio_metrics(weights, returns, risk_free_rate):
    """Calculate portfolio return, volatility, and Sharpe ratio"""
    portfolio_return = np.sum(returns.mean() * weights) * 252  # Annualized
    portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_vol
    
    return portfolio_return, portfolio_vol, sharpe_ratio

def optimize_portfolio(returns, method='max_sharpe'):
    """Optimize portfolio weights"""
    print(f"\n⚖️  Optimizing portfolio ({method})...")
    
    n_assets = len(returns.columns)
    
    # Initial guess (equal weight)
    x0 = np.ones(n_assets) / n_assets
    
    # Constraints
    constraints = [
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # Weights sum to 1
    ]
    
    # Bounds
    bounds = tuple((CONFIG['MIN_POSITION'], CONFIG['MAX_POSITION']) for _ in range(n_assets))
    
    if method == 'max_sharpe':
        # Maximize Sharpe ratio (minimize negative Sharpe)
        def objective(weights):
            _, _, sharpe = calculate_portfolio_metrics(weights, returns, CONFIG['RISK_FREE_RATE'])
            return -sharpe
        
    elif method == 'min_variance':
        # Minimize volatility
        def objective(weights):
            _, vol, _ = calculate_portfolio_metrics(weights, returns, CONFIG['RISK_FREE_RATE'])
            return vol
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Optimize
    result = minimize(
        objective,
        x0,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 1000}
    )
    
    if not result.success:
        print(f"⚠️  Optimization warning: {result.message}")
    
    optimal_weights = result.x
    
    # Calculate metrics
    port_return, port_vol, port_sharpe = calculate_portfolio_metrics(
        optimal_weights, returns, CONFIG['RISK_FREE_RATE']
    )
    
    print(f"✅ Optimization complete")
    print(f"   Expected Return: {port_return*100:.2f}%")
    print(f"   Volatility: {port_vol*100:.2f}%")
    print(f"   Sharpe Ratio: {port_sharpe:.3f}")
    
    return optimal_weights, {
        'return': port_return,
        'volatility': port_vol,
        'sharpe': port_sharpe
    }

def create_portfolio_allocation(tickers, weights, predictions, capital):
    """Create portfolio allocation with dollar amounts"""
    print(f"\n💰 Creating portfolio allocation (${capital:,})...")
    
    portfolio = pd.DataFrame({
        'ticker': tickers,
        'weight': weights,
        'allocation': weights * capital
    })
    
    # Merge with predictions
    portfolio = portfolio.merge(
        predictions[['ticker', 'price', 'composite_pred']],
        on='ticker',
        how='left'
    )
    
    # Calculate shares
    portfolio['price'] = pd.to_numeric(portfolio['price'], errors='coerce')
    portfolio['shares'] = (portfolio['allocation'] / portfolio['price']).astype(int)
    portfolio['actual_allocation'] = portfolio['shares'] * portfolio['price']
    portfolio['actual_weight'] = portfolio['actual_allocation'] / portfolio['actual_allocation'].sum()
    
    # Sort by weight
    portfolio = portfolio.sort_values('actual_weight', ascending=False)
    
    print(f"✅ Created allocation for {len(portfolio)} stocks")
    
    return portfolio

def save_results(portfolio, metrics, method):
    """Save portfolio allocation"""
    print("\n💾 Saving results...")
    
    output_dir = project_root / 'data' / 'processed'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    
    # Save portfolio
    portfolio_file = output_dir / f'portfolio_{method}_{timestamp}.csv'
    portfolio.to_csv(portfolio_file, index=False)
    print(f"✅ Saved portfolio: {portfolio_file.name}")
    
    # Save metrics
    metrics_file = output_dir / f'portfolio_metrics_{method}_{timestamp}.txt'
    with open(metrics_file, 'w') as f:
        f.write("PORTFOLIO OPTIMIZATION RESULTS\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Method: {method}\n")
        f.write(f"Capital: ${CONFIG['INITIAL_CAPITAL']:,}\n\n")
        f.write(f"Portfolio Metrics:\n")
        f.write(f"  Expected Annual Return: {metrics['return']*100:.2f}%\n")
        f.write(f"  Annual Volatility: {metrics['volatility']*100:.2f}%\n")
        f.write(f"  Sharpe Ratio: {metrics['sharpe']:.3f}\n\n")
        f.write(f"Actual Invested: ${portfolio['actual_allocation'].sum():,.2f}\n")
        f.write(f"Cash Remaining: ${CONFIG['INITIAL_CAPITAL'] - portfolio['actual_allocation'].sum():,.2f}\n")
    
    print(f"✅ Saved metrics: {metrics_file.name}")
    
    return portfolio_file

def main():
    """Main execution function"""
    print("=" * 80)
    print("⚖️  SMARTPORTFOLIO - PORTFOLIO OPTIMIZATION")
    print("=" * 80)
    
    try:
        # Connect to database
        print("\n📂 Connecting to database...")
        db = DatabaseConnector()
        
        if not db.test_connection():
            raise Exception("Database connection failed!")
        
        print("✅ Database connected")
        
        # Load ML predictions
        predictions = load_ml_predictions()
        tickers = predictions['ticker'].tolist()
        
        # Load historical returns
        returns = load_historical_returns(db, tickers, days=120)
        
        # Filter predictions to match available returns
        available_tickers = returns.columns.tolist()
        predictions = predictions[predictions['ticker'].isin(available_tickers)]
        tickers = predictions['ticker'].tolist()
        returns = returns[tickers]
        
        print(f"\n📊 Optimizing portfolio with {len(tickers)} stocks")
        
        # Optimize portfolio (Maximum Sharpe)
        optimal_weights, metrics = optimize_portfolio(returns, method='max_sharpe')
        
        # Create allocation
        portfolio = create_portfolio_allocation(
            tickers, 
            optimal_weights, 
            predictions,
            CONFIG['INITIAL_CAPITAL']
        )
        
        # Save results
        portfolio_file = save_results(portfolio, metrics, 'max_sharpe')
        
        # Summary
        print("\n" + "=" * 80)
        print("✅ PORTFOLIO OPTIMIZATION COMPLETE!")
        print("=" * 80)
        print(f"\n📊 Portfolio Summary:")
        print(f"  • Stocks: {len(portfolio)}")
        print(f"  • Total Invested: ${portfolio['actual_allocation'].sum():,.2f}")
        print(f"  • Cash Remaining: ${CONFIG['INITIAL_CAPITAL'] - portfolio['actual_allocation'].sum():,.2f}")
        print(f"  • Expected Return: {metrics['return']*100:.2f}%")
        print(f"  • Volatility: {metrics['volatility']*100:.2f}%")
        print(f"  • Sharpe Ratio: {metrics['sharpe']:.3f}")
        
        print(f"\n🏆 Top 10 Holdings:")
        display_cols = ['ticker', 'actual_weight', 'actual_allocation', 'shares', 'price']
        print(portfolio[display_cols].head(10).to_string(index=False))
        
        print(f"\n💾 Results saved to: {portfolio_file.name}")
        print(f"\n🎯 Next step: Run 'python scripts/run_backtest.py'")
        
        db.close()
        return 0
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())

