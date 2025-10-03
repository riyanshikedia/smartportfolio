"""
Visualization Helper Functions
Utility functions for creating charts and visualizations
"""

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np


# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)


def plot_price_history(df, ticker, title=None):
    """
    Plot price history with volume
    
    Args:
        df (pd.DataFrame): Price data with date, close, volume
        ticker (str): Stock ticker
        title (str): Plot title
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[3, 1])
    
    # Price plot
    ax1.plot(df['date'], df['close'], label='Close Price', linewidth=2)
    ax1.set_ylabel('Price ($)', fontsize=12)
    ax1.set_title(title or f'{ticker} Price History', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Volume plot
    ax2.bar(df['date'], df['volume'], alpha=0.7, color='steelblue')
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_ylabel('Volume', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_returns_distribution(returns, title='Returns Distribution'):
    """
    Plot returns distribution with histogram and KDE
    
    Args:
        returns (pd.Series): Returns series
        title (str): Plot title
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Histogram and KDE
    ax.hist(returns, bins=50, density=True, alpha=0.7, color='steelblue', edgecolor='black')
    returns.plot(kind='kde', ax=ax, linewidth=2, color='darkblue', label='KDE')
    
    # Add mean and median lines
    ax.axvline(returns.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {returns.mean():.4f}')
    ax.axvline(returns.median(), color='green', linestyle='--', linewidth=2, label=f'Median: {returns.median():.4f}')
    
    ax.set_xlabel('Returns', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig


def plot_correlation_matrix(df, title='Correlation Matrix'):
    """
    Plot correlation matrix heatmap
    
    Args:
        df (pd.DataFrame): Data with numeric columns
        title (str): Plot title
    """
    corr = df.corr()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    return fig


def plot_portfolio_allocation(weights_df, top_n=15):
    """
    Plot portfolio allocation as horizontal bar chart
    
    Args:
        weights_df (pd.DataFrame): DataFrame with ticker and weight columns
        top_n (int): Number of top holdings to show
    """
    top_holdings = weights_df.nlargest(top_n, 'weight')
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    ax.barh(top_holdings['ticker'], top_holdings['weight'] * 100, color='steelblue')
    ax.set_xlabel('Weight (%)', fontsize=12)
    ax.set_ylabel('Ticker', fontsize=12)
    ax.set_title(f'Top {top_n} Portfolio Holdings', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add percentage labels
    for i, (ticker, weight) in enumerate(zip(top_holdings['ticker'], top_holdings['weight'])):
        ax.text(weight * 100 + 0.2, i, f'{weight*100:.1f}%', va='center')
    
    plt.tight_layout()
    return fig


def plot_efficient_frontier(returns_list, volatility_list, sharpe_list, 
                            optimal_return=None, optimal_vol=None):
    """
    Plot efficient frontier
    
    Args:
        returns_list (list): Portfolio returns
        volatility_list (list): Portfolio volatilities
        sharpe_list (list): Sharpe ratios
        optimal_return (float): Optimal portfolio return
        optimal_vol (float): Optimal portfolio volatility
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Scatter plot colored by Sharpe ratio
    scatter = ax.scatter(volatility_list, returns_list, c=sharpe_list, 
                        cmap='viridis', s=50, alpha=0.6, edgecolors='black')
    
    # Mark optimal portfolio
    if optimal_return and optimal_vol:
        ax.scatter(optimal_vol, optimal_return, color='red', s=200, 
                  marker='*', edgecolors='black', linewidths=2,
                  label='Optimal Portfolio', zorder=5)
    
    ax.set_xlabel('Volatility (Risk)', fontsize=12)
    ax.set_ylabel('Expected Return', fontsize=12)
    ax.set_title('Efficient Frontier', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Sharpe Ratio', fontsize=12)
    
    plt.tight_layout()
    return fig


def plot_equity_curve(dates, values, benchmark_values=None, title='Portfolio Equity Curve'):
    """
    Plot equity curve vs benchmark
    
    Args:
        dates: Date index
        values (pd.Series): Portfolio values
        benchmark_values (pd.Series): Benchmark values (optional)
        title (str): Plot title
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Portfolio
    ax.plot(dates, values, label='Portfolio', linewidth=2, color='steelblue')
    
    # Benchmark
    if benchmark_values is not None:
        ax.plot(dates, benchmark_values, label='Benchmark (SPY)', 
               linewidth=2, color='orange', linestyle='--')
    
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Value ($)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Format y-axis as currency
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    plt.tight_layout()
    return fig


def create_interactive_portfolio_pie(weights_df, top_n=10):
    """
    Create interactive pie chart for portfolio allocation using Plotly
    
    Args:
        weights_df (pd.DataFrame): DataFrame with ticker and weight columns
        top_n (int): Number of top holdings to show, rest grouped as 'Others'
    
    Returns:
        plotly.graph_objects.Figure
    """
    top_holdings = weights_df.nlargest(top_n, 'weight')
    others_weight = weights_df.iloc[top_n:]['weight'].sum() if len(weights_df) > top_n else 0
    
    if others_weight > 0:
        others_row = pd.DataFrame([{'ticker': 'Others', 'weight': others_weight}])
        plot_data = pd.concat([top_holdings, others_row], ignore_index=True)
    else:
        plot_data = top_holdings
    
    fig = go.Figure(data=[go.Pie(
        labels=plot_data['ticker'],
        values=plot_data['weight'] * 100,
        hole=0.3,
        textinfo='label+percent',
        hovertemplate='<b>%{label}</b><br>Weight: %{value:.2f}%<extra></extra>'
    )])
    
    fig.update_layout(
        title='Portfolio Allocation',
        showlegend=True,
        height=500
    )
    
    return fig


def create_interactive_candlestick(df, ticker):
    """
    Create interactive candlestick chart using Plotly
    
    Args:
        df (pd.DataFrame): OHLC data
        ticker (str): Stock ticker
    
    Returns:
        plotly.graph_objects.Figure
    """
    fig = go.Figure(data=[go.Candlestick(
        x=df['date'],
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name=ticker
    )])
    
    fig.update_layout(
        title=f'{ticker} Price Chart',
        xaxis_title='Date',
        yaxis_title='Price ($)',
        hovermode='x unified',
        height=600
    )
    
    return fig

