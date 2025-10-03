#!/usr/bin/env python3
"""
Script 1: Data Collection
Collects S&P 500 tickers and historical price data from Yahoo Finance
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import yfinance as yf
from bs4 import BeautifulSoup
import requests
from datetime import datetime, timedelta
from tqdm import tqdm
from utils.database_connector import DatabaseConnector
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def get_sp500_tickers():
    """Scrape S&P 500 tickers from Wikipedia"""
    print("\n📊 Fetching S&P 500 tickers from Wikipedia...")
    
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    table = soup.find('table', {'id': 'constituents'})
    
    tickers_data = []
    for row in table.find_all('tr')[1:]:
        cols = row.find_all('td')
        if len(cols) >= 4:
            ticker = cols[0].text.strip().replace('.', '-')
            company = cols[1].text.strip()
            sector = cols[2].text.strip()
            industry = cols[3].text.strip()
            tickers_data.append({
                'ticker': ticker,
                'company_name': company,
                'sector': sector,
                'industry': industry
            })
    
    df = pd.DataFrame(tickers_data)
    print(f"✅ Found {len(df)} S&P 500 companies")
    return df

def fetch_stock_data(tickers, days=250):
    """Fetch historical price data for all tickers"""
    print(f"\n📈 Fetching {days} days of historical data...")
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days + 30)  # Extra buffer
    
    all_data = []
    failed = []
    
    for ticker in tqdm(tickers, desc="Downloading"):
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(start=start_date, end=end_date)
            
            if hist.empty:
                failed.append(ticker)
                continue
            
            hist = hist.reset_index()
            hist['ticker'] = ticker
            hist = hist.rename(columns={
                'Date': 'date',
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            })
            
            hist = hist[['ticker', 'date', 'open', 'high', 'low', 'close', 'volume']]
            hist = hist.tail(days)  # Keep only requested days
            all_data.append(hist)
            
        except Exception as e:
            print(f"\n⚠️  Error fetching {ticker}: {e}")
            failed.append(ticker)
    
    if all_data:
        df = pd.concat(all_data, ignore_index=True)
        print(f"\n✅ Successfully fetched {len(all_data)} stocks")
        if failed:
            print(f"⚠️  Failed to fetch {len(failed)} stocks: {', '.join(failed)}")
        return df
    else:
        raise Exception("Failed to fetch any stock data")

def main():
    """Main execution function"""
    print("=" * 80)
    print("🚀 SMARTPORTFOLIO - DATA COLLECTION")
    print("=" * 80)
    
    try:
        # Connect to database
        print("\n📂 Connecting to database...")
        db = DatabaseConnector()
        
        if not db.test_connection():
            raise Exception("Database connection failed!")
        
        print("✅ Database connected")
        
        # Step 1: Get S&P 500 tickers
        tickers_df = get_sp500_tickers()
        
        # Save tickers to database
        print("\n💾 Saving tickers to database...")
        db.execute_query("DELETE FROM market.sp500_tickers")
        db.insert_data(tickers_df, 'sp500_tickers', 'market')
        print(f"✅ Saved {len(tickers_df)} tickers to market.sp500_tickers")
        
        # Step 2: Fetch historical prices
        tickers_list = tickers_df['ticker'].tolist()
        prices_df = fetch_stock_data(tickers_list, days=250)
        
        # Save prices to database
        print("\n💾 Saving prices to database...")
        db.execute_query("DELETE FROM market.daily_prices")
        db.insert_data(prices_df, 'daily_prices', 'market', chunksize=1000)
        print(f"✅ Saved {len(prices_df)} price records to market.daily_prices")
        
        # Summary
        print("\n" + "=" * 80)
        print("✅ DATA COLLECTION COMPLETE!")
        print("=" * 80)
        print(f"\n📊 Summary:")
        print(f"  • Tickers: {len(tickers_df)}")
        print(f"  • Price records: {len(prices_df):,}")
        print(f"  • Date range: {prices_df['date'].min()} to {prices_df['date'].max()}")
        print(f"\n🎯 Next step: Run 'python scripts/run_technical_indicators.py'")
        
        db.close()
        return 0
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())

