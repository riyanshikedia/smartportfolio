#!/usr/bin/env python3
"""
Script 3: Stock Screening
Multi-factor screening using Fama-French methodology
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from utils.database_connector import DatabaseConnector
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
CONFIG = {
    'STRATEGY_NAME': 'Universal Multi-Factor',
    'WEIGHTS': {
        'fundamental': 40,
        'momentum': 25,
        'technical': 20,
        'risk': 15,
    },
    'TOP_N': 50,
    'MIN_SCORE': 60,
}

def calculate_scores(df):
    """Calculate multi-factor scores"""
    
    # Fundamental Score (0-100)
    df['fundamental_score'] = 0
    if 'pe_ratio' in df.columns:
        df['fundamental_score'] += np.where(df['pe_ratio'] < 15, 30, 
                                            np.where(df['pe_ratio'] < 25, 15, 0))
    if 'roe' in df.columns:
        df['fundamental_score'] += np.where(df['roe'] > 15, 40, 
                                            np.where(df['roe'] > 10, 20, 0))
    if 'profit_margin' in df.columns:
        df['fundamental_score'] += np.where(df['profit_margin'] > 15, 30, 
                                            np.where(df['profit_margin'] > 10, 15, 0))
    
    # Momentum Score (0-100)
    df['momentum_score'] = 0
    if 'return_1m' in df.columns:
        df['momentum_score'] += np.where(df['return_1m'] > 0.05, 30, 
                                        np.where(df['return_1m'] > 0, 15, 0))
    if 'return_3m' in df.columns:
        df['momentum_score'] += np.where(df['return_3m'] > 0.10, 40, 
                                        np.where(df['return_3m'] > 0, 20, 0))
    if 'return_6m' in df.columns:
        df['momentum_score'] += np.where(df['return_6m'] > 0.15, 30, 
                                        np.where(df['return_6m'] > 0, 15, 0))
    
    # Technical Score (0-100)
    df['technical_score'] = 0
    if 'rsi' in df.columns:
        df['technical_score'] += np.where((df['rsi'] > 30) & (df['rsi'] < 70), 35, 0)
    if 'macd_diff' in df.columns:
        df['technical_score'] += np.where(df['macd_diff'] > 0, 40, 0)
    if 'price_above_sma_50' in df.columns:
        df['technical_score'] += np.where(df['price_above_sma_50'], 25, 0)
    
    # Risk Score (0-100)
    df['risk_score'] = 0
    if 'volatility' in df.columns:
        df['risk_score'] += np.where(df['volatility'] < 0.30, 50, 
                                     np.where(df['volatility'] < 0.50, 25, 0))
    if 'sharpe_ratio' in df.columns:
        df['sharpe_ratio'] += np.where(df['sharpe_ratio'] > 1, 50, 
                                       np.where(df['sharpe_ratio'] > 0.5, 25, 0))
    
    # Weighted composite score
    df['composite_score'] = (
        df['fundamental_score'] * CONFIG['WEIGHTS']['fundamental'] / 100 +
        df['momentum_score'] * CONFIG['WEIGHTS']['momentum'] / 100 +
        df['technical_score'] * CONFIG['WEIGHTS']['technical'] / 100 +
        df['risk_score'] * CONFIG['WEIGHTS']['risk'] / 100
    )
    
    return df

def main():
    """Main execution function"""
    print("=" * 80)
    print("🔍 SMARTPORTFOLIO - STOCK SCREENING")
    print("=" * 80)
    print(f"\nStrategy: {CONFIG['STRATEGY_NAME']}")
    print(f"Selecting: Top {CONFIG['TOP_N']} stocks")
    print(f"Min Score: {CONFIG['MIN_SCORE']}/100")
    
    try:
        # Connect to database
        print("\n📂 Connecting to database...")
        db = DatabaseConnector()
        
        if not db.test_connection():
            raise Exception("Database connection failed!")
        
        print("✅ Database connected")
        
        # Load data
        print("\n📥 Loading market data...")
        
        query = """
        WITH LatestData AS (
            SELECT 
                p.ticker,
                t.company_name,
                t.sector,
                p.[close] as current_price,
                ti.rsi,
                ti.macd_diff,
                ti.sma_50,
                CASE WHEN p.[close] > ti.sma_50 THEN 1 ELSE 0 END as price_above_sma_50,
                LAG(p.[close], 20) OVER (PARTITION BY p.ticker ORDER BY p.date) as price_1m_ago,
                LAG(p.[close], 60) OVER (PARTITION BY p.ticker ORDER BY p.date) as price_3m_ago,
                LAG(p.[close], 125) OVER (PARTITION BY p.ticker ORDER BY p.date) as price_6m_ago,
                ROW_NUMBER() OVER (PARTITION BY p.ticker ORDER BY p.date DESC) as rn
            FROM market.daily_prices p
            LEFT JOIN market.sp500_tickers t ON p.ticker = t.ticker
            LEFT JOIN market.technical_indicators ti ON p.ticker = ti.ticker AND p.date = ti.date
        )
        SELECT 
            ticker,
            company_name,
            sector,
            current_price,
            rsi,
            macd_diff,
            sma_50,
            price_above_sma_50,
            (current_price - price_1m_ago) / NULLIF(price_1m_ago, 0) as return_1m,
            (current_price - price_3m_ago) / NULLIF(price_3m_ago, 0) as return_3m,
            (current_price - price_6m_ago) / NULLIF(price_6m_ago, 0) as return_6m
        FROM LatestData
        WHERE rn = 1
        AND price_6m_ago IS NOT NULL
        """
        
        df = db.execute_query(query)
        print(f"✅ Loaded {len(df)} stocks")
        
        # Calculate scores
        print("\n🎯 Calculating multi-factor scores...")
        df = calculate_scores(df)
        
        # Filter and rank
        df = df[df['composite_score'] >= CONFIG['MIN_SCORE']].copy()
        df = df.sort_values('composite_score', ascending=False)
        df['rank'] = range(1, len(df) + 1)
        
        # Select top N
        top_stocks = df.head(CONFIG['TOP_N'])
        
        # Save results
        print("\n💾 Saving screening results...")
        output_dir = project_root / 'data' / 'processed'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M')
        output_file = output_dir / f'screening_results_{timestamp}.csv'
        top_stocks.to_csv(output_file, index=False)
        
        # Summary
        print("\n" + "=" * 80)
        print("✅ STOCK SCREENING COMPLETE!")
        print("=" * 80)
        print(f"\n📊 Results:")
        print(f"  • Stocks analyzed: {len(df)}")
        print(f"  • Stocks above threshold: {len(df[df['composite_score'] >= CONFIG['MIN_SCORE']])}")
        print(f"  • Top selections: {len(top_stocks)}")
        print(f"\n🏆 Top 10 Stocks:")
        print(top_stocks[['rank', 'ticker', 'company_name', 'composite_score']].head(10).to_string(index=False))
        print(f"\n💾 Results saved to: {output_file.name}")
        print(f"\n🎯 Next step: Run 'python scripts/run_ml_prediction.py'")
        
        db.close()
        return 0
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())

