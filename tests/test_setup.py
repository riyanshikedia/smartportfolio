#!/usr/bin/env python3
"""
Test Setup Script
Quick test to verify SmartPortfolio setup is working
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Test that all required modules can be imported"""
    print("🧪 Testing imports...")
    
    try:
        import pandas as pd
        import numpy as np
        import yfinance as yf
        import requests
        from bs4 import BeautifulSoup
        from dotenv import load_dotenv
        import sqlalchemy as sa
        print("✅ All imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_database_connection():
    """Test database connection"""
    print("🧪 Testing database connection...")
    
    try:
        from utils.database_connector import DatabaseConnector
        db = DatabaseConnector()
        
        if db.test_connection():
            print("✅ Database connection successful")
            db.close()
            return True
        else:
            print("❌ Database connection failed")
            db.close()
            return False
    except Exception as e:
        print(f"❌ Database test error: {e}")
        return False

def test_data_collection():
    """Test data collection functionality"""
    print("🧪 Testing data collection...")
    
    try:
        from scripts.run_data_collection import get_sp500_tickers
        
        # Test Wikipedia scraping
        tickers_df = get_sp500_tickers()
        if len(tickers_df) > 0:
            print(f"✅ Data collection test successful - found {len(tickers_df)} tickers")
            return True
        else:
            print("❌ Data collection test failed - no tickers found")
            return False
    except Exception as e:
        print(f"❌ Data collection test error: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 80)
    print("🧪 SMARTPORTFOLIO SETUP TEST")
    print("=" * 80)
    
    tests = [
        ("Import Test", test_imports),
        ("Database Connection Test", test_database_connection),
        ("Data Collection Test", test_data_collection)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        if test_func():
            passed += 1
        else:
            print(f"❌ {test_name} failed")
    
    print("\n" + "=" * 80)
    print(f"📊 TEST RESULTS: {passed}/{total} tests passed")
    print("=" * 80)
    
    if passed == total:
        print("🎉 All tests passed! Setup is working correctly.")
        return 0
    else:
        print("⚠️  Some tests failed. Check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
