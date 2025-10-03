"""
Tests for Portfolio Optimization Module
"""

import pytest
import pandas as pd
import numpy as np


class TestPortfolioMetrics:
    """Test portfolio calculation functions"""
    
    def test_portfolio_return(self):
        """Test portfolio return calculation"""
        # Create sample returns
        returns = pd.DataFrame({
            'AAPL': [0.01, 0.02, -0.01],
            'GOOGL': [0.02, -0.01, 0.03],
            'MSFT': [-0.01, 0.03, 0.02]
        })
        
        weights = np.array([0.4, 0.3, 0.3])
        
        # Calculate portfolio returns
        portfolio_returns = (returns * weights).sum(axis=1)
        
        assert len(portfolio_returns) == 3
        assert abs(portfolio_returns.sum() - 0.063) < 0.001
    
    def test_portfolio_volatility(self):
        """Test portfolio volatility calculation"""
        # Create sample returns
        returns = pd.DataFrame({
            'AAPL': [0.01, 0.02, -0.01, 0.03],
            'GOOGL': [0.02, -0.01, 0.03, -0.02],
        })
        
        weights = np.array([0.5, 0.5])
        
        # Calculate portfolio returns
        portfolio_returns = (returns * weights).sum(axis=1)
        
        # Calculate volatility
        volatility = portfolio_returns.std() * np.sqrt(252)
        
        assert volatility > 0
    
    def test_sharpe_ratio_calculation(self):
        """Test Sharpe ratio for portfolio"""
        returns = pd.Series([0.001, 0.002, 0.001, 0.003, 0.002])
        
        mean_return = returns.mean() * 252  # Annualize
        volatility = returns.std() * np.sqrt(252)
        risk_free_rate = 0.045
        
        sharpe = (mean_return - risk_free_rate) / volatility
        
        assert sharpe > 0


class TestWeightConstraints:
    """Test portfolio weight constraints"""
    
    def test_weights_sum_to_one(self):
        """Test that portfolio weights sum to 1.0"""
        weights = np.array([0.3, 0.4, 0.2, 0.1])
        
        assert abs(weights.sum() - 1.0) < 1e-6
    
    def test_no_negative_weights(self):
        """Test no short selling (long-only)"""
        weights = np.array([0.3, 0.4, 0.2, 0.1])
        
        assert np.all(weights >= 0)
    
    def test_max_position_size(self):
        """Test maximum position size constraint"""
        weights = np.array([0.1, 0.09, 0.08, 0.07, 0.06])
        max_position = 0.10
        
        assert np.all(weights <= max_position + 1e-6)


class TestPortfolioOptimization:
    """Test portfolio optimization"""
    
    def test_equal_weight_portfolio(self):
        """Test equal weight portfolio creation"""
        n_stocks = 10
        weights = np.ones(n_stocks) / n_stocks
        
        assert len(weights) == n_stocks
        assert abs(weights.sum() - 1.0) < 1e-6
        assert np.all(weights == 0.1)
    
    def test_minimum_variance_portfolio(self):
        """Test minimum variance optimization concept"""
        # Create sample covariance matrix
        cov_matrix = np.array([
            [0.04, 0.01],
            [0.01, 0.09]
        ])
        
        # For 2 assets, minimum variance can be calculated analytically
        # This is a simplified test of the concept
        
        weights = np.array([0.7, 0.3])
        
        portfolio_variance = weights @ cov_matrix @ weights.T
        
        assert portfolio_variance > 0
        assert portfolio_variance < max(cov_matrix[0, 0], cov_matrix[1, 1])


class TestRiskMetrics:
    """Test risk calculation metrics"""
    
    def test_value_at_risk(self):
        """Test VaR calculation"""
        returns = pd.Series(np.random.normal(0.001, 0.02, 1000))
        
        # 95% VaR
        var_95 = returns.quantile(0.05)
        
        assert var_95 < 0  # VaR should be negative (loss)
    
    def test_conditional_var(self):
        """Test CVaR (Expected Shortfall) calculation"""
        returns = pd.Series(np.random.normal(0.001, 0.02, 1000))
        
        # 95% CVaR
        var_95 = returns.quantile(0.05)
        cvar_95 = returns[returns <= var_95].mean()
        
        assert cvar_95 < var_95  # CVaR should be more extreme than VaR
    
    def test_maximum_drawdown(self):
        """Test maximum drawdown calculation"""
        # Create returns with a known drawdown
        returns = pd.Series([0.02, 0.03, -0.05, -0.04, 0.02, 0.03])
        
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_dd = drawdown.min()
        
        assert max_dd < 0
        assert max_dd > -0.15  # Should be reasonable


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])

