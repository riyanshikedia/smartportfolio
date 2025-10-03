"""
Tests for Data Collection Module
"""

import pytest
import pandas as pd
import numpy as np
from utils.data_helpers import (
    validate_price_data,
    calculate_returns,
    calculate_volatility,
    calculate_sharpe_ratio,
    detect_outliers,
    calculate_max_drawdown
)


class TestDataValidation:
    """Test data validation functions"""
    
    def test_validate_price_data_valid(self):
        """Test validation with valid data"""
        df = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [105, 106, 107],
            'low': [99, 100, 101],
            'close': [103, 104, 105],
            'volume': [1000000, 1100000, 1200000]
        })
        
        is_valid, errors = validate_price_data(df)
        assert is_valid == True
        assert len(errors) == 0
    
    def test_validate_price_data_negative_prices(self):
        """Test validation with negative prices"""
        df = pd.DataFrame({
            'open': [-100, 101, 102],
            'high': [105, 106, 107],
            'low': [99, 100, 101],
            'close': [103, 104, 105],
            'volume': [1000000, 1100000, 1200000]
        })
        
        is_valid, errors = validate_price_data(df)
        assert is_valid == False
        assert "Negative prices detected" in errors
    
    def test_validate_price_data_invalid_high_low(self):
        """Test validation with high < low"""
        df = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [99, 100, 101],  # High < Low
            'low': [105, 106, 107],
            'close': [103, 104, 105],
            'volume': [1000000, 1100000, 1200000]
        })
        
        is_valid, errors = validate_price_data(df)
        assert is_valid == False
        assert "High < Low detected" in errors


class TestReturnsCalculation:
    """Test returns calculation functions"""
    
    def test_calculate_returns(self):
        """Test returns calculation"""
        prices = pd.Series([100, 105, 103, 108, 110])
        returns = calculate_returns(prices, periods=[1, 2])
        
        assert 'return_1d' in returns.columns
        assert 'return_2d' in returns.columns
        assert len(returns) == len(prices)
        
        # Check first return (5% increase)
        assert abs(returns['return_1d'].iloc[1] - 0.05) < 0.001
    
    def test_calculate_volatility(self):
        """Test volatility calculation"""
        returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02] * 10)
        vol = calculate_volatility(returns, window=5)
        
        assert len(vol) == len(returns)
        assert not pd.isna(vol.iloc[-1])  # Last value should be valid
    
    def test_calculate_sharpe_ratio(self):
        """Test Sharpe ratio calculation"""
        # Positive returns should give positive Sharpe
        returns = pd.Series([0.001] * 252)  # 0.1% daily return
        sharpe = calculate_sharpe_ratio(returns, risk_free_rate=0.045)
        
        assert sharpe > 0
        
        # Zero returns should give negative Sharpe (below risk-free rate)
        returns_zero = pd.Series([0.0] * 252)
        sharpe_zero = calculate_sharpe_ratio(returns_zero, risk_free_rate=0.045)
        
        assert sharpe_zero < 0


class TestOutlierDetection:
    """Test outlier detection"""
    
    def test_detect_outliers(self):
        """Test outlier detection"""
        df = pd.DataFrame({
            'price': [100, 101, 102, 103, 104, 105, 200]  # 200 is outlier
        })
        
        outliers = detect_outliers(df, 'price', n_std=2)
        
        assert outliers.iloc[-1] == True  # Last value is outlier
        assert outliers.iloc[0] == False  # First value is not


class TestMaxDrawdown:
    """Test max drawdown calculation"""
    
    def test_max_drawdown(self):
        """Test max drawdown calculation"""
        # Create returns with a drawdown
        returns = pd.Series([0.01, 0.02, -0.05, -0.03, 0.02, 0.03])
        
        max_dd = calculate_max_drawdown(returns)
        
        assert max_dd < 0  # Drawdown should be negative
        assert max_dd > -0.1  # Should be within reasonable range


class TestDataHelpers:
    """Test general data helper functions"""
    
    def test_split_train_test(self):
        """Test train/test split"""
        from utils.data_helpers import split_train_test
        
        df = pd.DataFrame({
            'feature': range(100),
            'target': range(100)
        })
        
        train, test = split_train_test(df, test_size=0.2, shuffle=False)
        
        assert len(train) == 80
        assert len(test) == 20
        assert train.iloc[-1]['feature'] < test.iloc[0]['feature']  # Time series order preserved


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])

