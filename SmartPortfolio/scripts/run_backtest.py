#!/usr/bin/env python3
"""
Script 6: Portfolio Backtesting
Backtest portfolio performance and calculate risk metrics
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from datetime import datetime

from utils.database_connector import DatabaseConnector
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
CONFIG = {
    'BACKTEST_DAYS': 120,  # ~6 months
    'BENCHMARK': 'SPY',
    'RISK_FREE_RATE': 0.045,
}

def load_portfolio():
    """Load latest optimized portfolio"""
    print("\n📥 Loading portfolio...")
    
    data_dir = project_root / 'data' / 'processed'
    
    # Find latest portfolio file
    portfolio_files = sorted(data_dir.glob('portfolio_*.csv'), reverse=True)
    
    if not portfolio_files:
        raise FileNotFoundError("No portfolio files found. Run 'run_optimization.py' first.")
    
    latest_file = portfolio_files[0]
    df = pd.read_csv(latest_file)
    
    print(f"✅ Loaded {len(df)} stocks from {latest_file.name}")
    
    return df

def load_historical_prices(db, tickers, days):
    """Load historical prices for backtest"""
    print(f"\n📥 Loading {days} days of historical prices...")
    
    tickers_str = "', '".join(tickers)
    
    query = f"""
    WITH RankedPrices AS (
        SELECT 
            ticker,
            date,
            [close] as price,
            ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY date DESC) as rn
        FROM market.daily_prices
        WHERE ticker IN ('{tickers_str}')
    )
    SELECT ticker, date, price
    FROM RankedPrices
    WHERE rn <= {days}
    ORDER BY ticker, date
    """
    
    df = db.execute_query(query)
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    
    print(f"✅ Loaded prices for {df['ticker'].nunique()} stocks")
    
    return df

def calculate_portfolio_value(prices, weights):
    """Calculate portfolio value over time"""
    print("\n📊 Calculating portfolio performance...")
    
    # Pivot to wide format
    price_matrix = prices.pivot(index='date', columns='ticker', values='price')
    
    # Ensure tickers match weights
    common_tickers = list(set(price_matrix.columns) & set(weights.index))
    price_matrix = price_matrix[common_tickers]
    weights = weights[common_tickers]
    weights = weights / weights.sum()  # Renormalize
    
    # Calculate returns
    returns = price_matrix.pct_change().fillna(0)
    
    # Portfolio returns
    portfolio_returns = (returns * weights.values).sum(axis=1)
    
    # Cumulative returns
    portfolio_cumulative = (1 + portfolio_returns).cumprod()
    
    print(f"✅ Calculated {len(portfolio_returns)} days of returns")
    
    return portfolio_returns, portfolio_cumulative

def calculate_metrics(returns, cumulative, risk_free_rate):
    """Calculate performance metrics"""
    print("\n📈 Calculating performance metrics...")
    
    # Total return
    total_return = cumulative.iloc[-1] - 1
    
    # Annualized return
    days = len(returns)
    annual_return = (1 + total_return) ** (252 / days) - 1
    
    # Volatility
    volatility = returns.std() * np.sqrt(252)
    
    # Sharpe ratio
    sharpe = (annual_return - risk_free_rate) / volatility
    
    # Maximum drawdown
    cumulative_max = cumulative.expanding().max()
    drawdown = (cumulative - cumulative_max) / cumulative_max
    max_drawdown = drawdown.min()
    
    # Positive days
    positive_days = (returns > 0).sum()
    total_days = len(returns)
    win_rate = positive_days / total_days
    
    metrics = {
        'total_return': total_return,
        'annual_return': annual_return,
        'volatility': volatility,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'trading_days': total_days
    }
    
    print("✅ Metrics calculated")
    
    return metrics

def calculate_var(returns, confidence=0.95):
    """Calculate Value at Risk and Conditional VaR"""
    var = returns.quantile(1 - confidence)
    cvar = returns[returns <= var].mean()
    
    return var, cvar

def compare_to_benchmark(portfolio_returns, benchmark_returns):
    """Compare portfolio to benchmark"""
    print("\n📊 Comparing to benchmark...")
    
    # Align dates
    common_dates = portfolio_returns.index.intersection(benchmark_returns.index)
    
    if len(common_dates) == 0:
        print("⚠️  No common dates with benchmark")
        return None
    
    port_ret = portfolio_returns[common_dates]
    bench_ret = benchmark_returns[common_dates]
    
    # Calculate beta and alpha
    covariance = np.cov(port_ret, bench_ret)[0][1]
    benchmark_var = np.var(bench_ret)
    
    beta = covariance / benchmark_var if benchmark_var > 0 else 1.0
    
    # Alpha (annualized)
    port_annual = (1 + port_ret.mean()) ** 252 - 1
    bench_annual = (1 + bench_ret.mean()) ** 252 - 1
    alpha = port_annual - (CONFIG['RISK_FREE_RATE'] + beta * (bench_annual - CONFIG['RISK_FREE_RATE']))
    
    # Tracking error
    tracking_error = (port_ret - bench_ret).std() * np.sqrt(252)
    
    # Information ratio
    information_ratio = alpha / tracking_error if tracking_error > 0 else 0
    
    print("✅ Benchmark comparison complete")
    
    return {
        'beta': beta,
        'alpha': alpha,
        'tracking_error': tracking_error,
        'information_ratio': information_ratio
    }

def save_results(metrics, benchmark_metrics, portfolio):
    """Save backtest results"""
    print("\n💾 Saving results...")
    
    output_dir = project_root / 'data' / 'processed'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    
    # Save metrics
    results_file = output_dir / f'backtest_results_{timestamp}.txt'
    
    with open(results_file, 'w') as f:
        f.write("PORTFOLIO BACKTEST RESULTS\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"Backtest Period: {CONFIG['BACKTEST_DAYS']} days (~6 months)\n")
        f.write(f"Stocks: {len(portfolio)}\n")
        f.write(f"Initial Capital: ${portfolio['actual_allocation'].sum():,.2f}\n\n")
        
        f.write("Performance Metrics:\n")
        f.write(f"  Total Return: {metrics['total_return']*100:+.2f}%\n")
        f.write(f"  Annualized Return: {metrics['annual_return']*100:+.2f}%\n")
        f.write(f"  Annual Volatility: {metrics['volatility']*100:.2f}%\n")
        f.write(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.3f}\n")
        f.write(f"  Maximum Drawdown: {metrics['max_drawdown']*100:.2f}%\n")
        f.write(f"  Win Rate: {metrics['win_rate']*100:.2f}%\n")
        f.write(f"  Trading Days: {metrics['trading_days']}\n\n")
        
        if benchmark_metrics:
            f.write("Benchmark Comparison (vs S&P 500):\n")
            f.write(f"  Beta: {benchmark_metrics['beta']:.3f}\n")
            f.write(f"  Alpha: {benchmark_metrics['alpha']*100:+.2f}%\n")
            f.write(f"  Tracking Error: {benchmark_metrics['tracking_error']*100:.2f}%\n")
            f.write(f"  Information Ratio: {benchmark_metrics['information_ratio']:.3f}\n")
    
    print(f"✅ Saved results: {results_file.name}")
    
    return results_file

def main():
    """Main execution function"""
    print("=" * 80)
    print("📉 SMARTPORTFOLIO - PORTFOLIO BACKTESTING")
    print("=" * 80)
    
    try:
        # Connect to database
        print("\n📂 Connecting to database...")
        db = DatabaseConnector()
        
        if not db.test_connection():
            raise Exception("Database connection failed!")
        
        print("✅ Database connected")
        
        # Load portfolio
        portfolio = load_portfolio()
        tickers = portfolio['ticker'].tolist()
        weights = pd.Series(
            portfolio['actual_weight'].values,
            index=portfolio['ticker'].values
        )
        
        # Load historical prices
        prices = load_historical_prices(db, tickers, CONFIG['BACKTEST_DAYS'])
        
        # Calculate portfolio performance
        returns, cumulative = calculate_portfolio_value(prices, weights)
        
        # Calculate metrics
        metrics = calculate_metrics(returns, cumulative, CONFIG['RISK_FREE_RATE'])
        
        # Calculate VaR
        var_95, cvar_95 = calculate_var(returns, confidence=0.95)
        
        # Load benchmark (if available)
        benchmark_metrics = None
        try:
            benchmark_prices = load_historical_prices(db, [CONFIG['BENCHMARK']], CONFIG['BACKTEST_DAYS'])
            benchmark_returns = benchmark_prices.set_index('date')['price'].pct_change().dropna()
            benchmark_metrics = compare_to_benchmark(returns, benchmark_returns)
        except Exception as e:
            print(f"⚠️  Could not load benchmark: {e}")
        
        # Save results
        results_file = save_results(metrics, benchmark_metrics, portfolio)
        
        # Summary
        print("\n" + "=" * 80)
        print("✅ BACKTEST COMPLETE!")
        print("=" * 80)
        
        print(f"\n📊 Performance Summary:")
        print(f"  • Backtest Period: {metrics['trading_days']} days")
        print(f"  • Total Return: {metrics['total_return']*100:+.2f}%")
        print(f"  • Annualized Return: {metrics['annual_return']*100:+.2f}%")
        print(f"  • Annual Volatility: {metrics['volatility']*100:.2f}%")
        print(f"  • Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
        print(f"  • Maximum Drawdown: {metrics['max_drawdown']*100:.2f}%")
        print(f"  • Win Rate: {metrics['win_rate']*100:.2f}%")
        
        print(f"\n📉 Risk Metrics:")
        print(f"  • Value at Risk (95%): {var_95*100:.2f}%")
        print(f"  • Conditional VaR (95%): {cvar_95*100:.2f}%")
        
        if benchmark_metrics:
            print(f"\n📊 vs {CONFIG['BENCHMARK']}:")
            print(f"  • Beta: {benchmark_metrics['beta']:.3f}")
            print(f"  • Alpha: {benchmark_metrics['alpha']*100:+.2f}%")
            print(f"  • Tracking Error: {benchmark_metrics['tracking_error']*100:.2f}%")
            print(f"  • Information Ratio: {benchmark_metrics['information_ratio']:.3f}")
        
        print(f"\n💾 Results saved to: {results_file.name}")
        print(f"\n🎉 Complete pipeline finished! Check notebooks/06_dashboard.ipynb for visualizations.")
        
        db.close()
        return 0
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())

