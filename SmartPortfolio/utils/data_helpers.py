"""
Data Helper Functions
Utility functions for data processing, validation, and transformation
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def validate_price_data(df):
    """
    Validate price data for common issues
    
    Args:
        df (pd.DataFrame): Price data with OHLCV columns
        
    Returns:
        tuple: (is_valid, error_messages)
    """
    errors = []
    
    # Check required columns
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        errors.append(f"Missing columns: {missing_cols}")
    
    # Check for negative prices
    if (df[['open', 'high', 'low', 'close']] < 0).any().any():
        errors.append("Negative prices detected")
    
    # Check for zero volumes
    if (df['volume'] == 0).sum() > len(df) * 0.1:  # >10% zero volumes
        errors.append("Too many zero volume days")
    
    # Check for NaN values
    if df[required_cols].isnull().any().any():
        errors.append("NaN values detected")
    
    # Check high >= low
    if (df['high'] < df['low']).any():
        errors.append("High < Low detected")
    
    # Check close within high/low
    if ((df['close'] > df['high']) | (df['close'] < df['low'])).any():
        errors.append("Close outside High/Low range")
    
    return len(errors) == 0, errors


def calculate_returns(prices, periods=[1, 5, 20, 60]):
    """
    Calculate returns for multiple periods
    
    Args:
        prices (pd.Series): Price series
        periods (list): List of periods to calculate returns
        
    Returns:
        pd.DataFrame: Returns for each period
    """
    returns = pd.DataFrame(index=prices.index)
    
    for period in periods:
        returns[f'return_{period}d'] = prices.pct_change(periods=period)
    
    return returns


def calculate_volatility(returns, window=20):
    """
    Calculate rolling volatility (annualized)
    
    Args:
        returns (pd.Series): Daily returns
        window (int): Rolling window size
        
    Returns:
        pd.Series: Annualized volatility
    """
    return returns.rolling(window=window).std() * np.sqrt(252)


def calculate_sharpe_ratio(returns, risk_free_rate=0.045, periods_per_year=252):
    """
    Calculate Sharpe ratio
    
    Args:
        returns (pd.Series): Daily returns
        risk_free_rate (float): Annual risk-free rate
        periods_per_year (int): Trading periods per year
        
    Returns:
        float: Sharpe ratio
    """
    excess_returns = returns.mean() * periods_per_year - risk_free_rate
    volatility = returns.std() * np.sqrt(periods_per_year)
    
    if volatility == 0:
        return 0
    
    return excess_returns / volatility


def detect_outliers(df, column, n_std=3):
    """
    Detect outliers using standard deviation method
    
    Args:
        df (pd.DataFrame): Data
        column (str): Column name to check
        n_std (int): Number of standard deviations
        
    Returns:
        pd.Series: Boolean series of outliers
    """
    mean = df[column].mean()
    std = df[column].std()
    
    return (df[column] < mean - n_std * std) | (df[column] > mean + n_std * std)


def fill_missing_dates(df, date_col='date', ticker_col='ticker'):
    """
    Fill missing trading dates with forward fill
    
    Args:
        df (pd.DataFrame): Price data
        date_col (str): Date column name
        ticker_col (str): Ticker column name
        
    Returns:
        pd.DataFrame: Data with filled dates
    """
    # Create complete date range
    all_dates = pd.date_range(
        start=df[date_col].min(),
        end=df[date_col].max(),
        freq='D'
    )
    
    # Fill for each ticker
    filled_dfs = []
    for ticker in df[ticker_col].unique():
        ticker_df = df[df[ticker_col] == ticker].copy()
        ticker_df = ticker_df.set_index(date_col)
        ticker_df = ticker_df.reindex(all_dates)
        ticker_df = ticker_df.fillna(method='ffill')
        ticker_df[ticker_col] = ticker
        ticker_df = ticker_df.reset_index().rename(columns={'index': date_col})
        filled_dfs.append(ticker_df)
    
    return pd.concat(filled_dfs, ignore_index=True)


def normalize_features(df, columns, method='standard'):
    """
    Normalize features
    
    Args:
        df (pd.DataFrame): Data
        columns (list): Columns to normalize
        method (str): 'standard', 'minmax', or 'robust'
        
    Returns:
        pd.DataFrame: Normalized data
    """
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
    
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    elif method == 'robust':
        scaler = RobustScaler()
    else:
        raise ValueError(f"Unknown method: {method}")
    
    df[columns] = scaler.fit_transform(df[columns])
    
    return df, scaler


def create_lag_features(df, columns, lags=[1, 5, 10]):
    """
    Create lagged features
    
    Args:
        df (pd.DataFrame): Data
        columns (list): Columns to lag
        lags (list): Lag periods
        
    Returns:
        pd.DataFrame: Data with lag features
    """
    for col in columns:
        for lag in lags:
            df[f'{col}_lag_{lag}'] = df[col].shift(lag)
    
    return df


def calculate_max_drawdown(returns):
    """
    Calculate maximum drawdown
    
    Args:
        returns (pd.Series): Daily returns
        
    Returns:
        float: Maximum drawdown (negative value)
    """
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    
    return drawdown.min()


def split_train_test(df, test_size=0.2, shuffle=False):
    """
    Split data into train and test sets
    
    Args:
        df (pd.DataFrame): Data
        test_size (float): Proportion of test data
        shuffle (bool): Whether to shuffle (False for time series)
        
    Returns:
        tuple: (train_df, test_df)
    """
    if shuffle:
        df = df.sample(frac=1).reset_index(drop=True)
    
    split_idx = int(len(df) * (1 - test_size))
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    return train_df, test_df
