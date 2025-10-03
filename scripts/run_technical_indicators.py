#!/usr/bin/env python3
"""
Script 2: Technical Indicators
Calculates technical indicators for all stocks
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from ta import add_all_ta_features
from tqdm import tqdm
from utils.database_connector import DatabaseConnector
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def calculate_indicators(df):
    """Calculate technical indicators for a single stock"""
    try:
        # Ensure proper column names
        df = df.rename(columns={
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        })
        
        # Add all technical indicators using ta library
        df = add_all_ta_features(
            df, 
            open="Open", 
            high="High", 
            low="Low", 
            close="Close", 
            volume="Volume",
            fillna=True
        )
        
        # Extract key indicators
        indicators = pd.DataFrame({
            'ticker': df['ticker'],
            'date': df['date'],
            'sma_20': df.get('trend_sma_fast', np.nan),
            'sma_50': df.get('trend_sma_slow', np.nan),
            'ema_12': df.get('trend_ema_fast', np.nan),
            'ema_26': df.get('trend_ema_slow', np.nan),
            'rsi': df.get('momentum_rsi', np.nan),
            'macd': df.get('trend_macd', np.nan),
            'macd_signal': df.get('trend_macd_signal', np.nan),
            'macd_diff': df.get('trend_macd_diff', np.nan),
            'bb_upper': df.get('volatility_bbh', np.nan),
            'bb_middle': df.get('volatility_bbm', np.nan),
            'bb_lower': df.get('volatility_bbl', np.nan),
            'bb_width': df.get('volatility_bbw', np.nan),
            'atr': df.get('volatility_atr', np.nan),
            'adx': df.get('trend_adx', np.nan),
            'obv': df.get('volume_obv', np.nan),
            'stoch': df.get('momentum_stoch', np.nan),
            'stoch_signal': df.get('momentum_stoch_signal', np.nan)
        })
        
        return indicators
        
    except Exception as e:
        print(f"Error calculating indicators: {e}")
        return None

def main():
    """Main execution function"""
    print("=" * 80)
    print("📊 SMARTPORTFOLIO - TECHNICAL INDICATORS")
    print("=" * 80)
    
    try:
        # Connect to database
        print("\n📂 Connecting to database...")
        db = DatabaseConnector()
        
        if not db.test_connection():
            raise Exception("Database connection failed!")
        
        print("✅ Database connected")
        
        # Load tickers
        print("\n📥 Loading stock data...")
        tickers_df = db.execute_query("SELECT ticker FROM market.sp500_tickers")
        tickers = tickers_df['ticker'].tolist()
        print(f"✅ Found {len(tickers)} tickers")
        
        # Load price data
        prices_df = db.execute_query("""
            SELECT ticker, date, [open], high, low, [close], volume
            FROM market.daily_prices
            ORDER BY ticker, date
        """)
        print(f"✅ Loaded {len(prices_df):,} price records")
        
        # Calculate indicators for each stock
        print("\n🔬 Calculating technical indicators...")
        all_indicators = []
        
        for ticker in tqdm(tickers, desc="Processing"):
            stock_data = prices_df[prices_df['ticker'] == ticker].copy()
            
            if len(stock_data) < 50:  # Need minimum data
                continue
            
            indicators = calculate_indicators(stock_data)
            if indicators is not None:
                all_indicators.append(indicators)
        
        # Combine all indicators
        indicators_df = pd.concat(all_indicators, ignore_index=True)
        
        # Remove NaN rows (insufficient data)
        indicators_df = indicators_df.dropna(subset=['sma_20', 'rsi'])
        
        print(f"\n✅ Calculated {len(indicators_df):,} indicator records")
        
        # Save to database
        print("\n💾 Saving indicators to database...")
        db.execute_query("DELETE FROM market.technical_indicators")
        db.insert_data(indicators_df, 'technical_indicators', 'market', chunksize=1000)
        print(f"✅ Saved to market.technical_indicators")
        
        # Summary
        print("\n" + "=" * 80)
        print("✅ TECHNICAL INDICATORS COMPLETE!")
        print("=" * 80)
        print(f"\n📊 Summary:")
        print(f"  • Tickers processed: {len(tickers)}")
        print(f"  • Indicator records: {len(indicators_df):,}")
        print(f"  • Date range: {indicators_df['date'].min()} to {indicators_df['date'].max()}")
        print(f"\n🎯 Next step: Run 'python scripts/run_screening.py'")
        
        db.close()
        return 0
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())

