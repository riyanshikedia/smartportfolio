#!/usr/bin/env python3
"""
Script 4: ML Return Prediction
Train XGBoost models to predict 1M and 3M stock returns
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm
from xgboost import XGBRegressor
import joblib

from utils.database_connector import DatabaseConnector
from utils.ml_helpers import prepare_features, train_model, evaluate_model
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
CONFIG = {
    'HORIZONS': {
        '1M': 20,   # 1 month = ~20 trading days
        '3M': 60,   # 3 months = ~60 trading days
    },
    'TOP_N': 50,
    'MIN_TRAINING_SAMPLES': 1000,
}

def load_data(db):
    """Load price and technical indicator data"""
    print("\n📥 Loading data from database...")
    
    query = """
    SELECT 
        p.ticker,
        p.date,
        p.[close] as price,
        p.[open],
        p.high,
        p.low,
        p.volume,
        ti.sma_20,
        ti.sma_50,
        ti.ema_12,
        ti.ema_26,
        ti.rsi,
        ti.macd,
        ti.macd_signal,
        ti.macd_diff,
        ti.bb_upper,
        ti.bb_middle,
        ti.bb_lower,
        ti.bb_width,
        ti.atr,
        ti.adx,
        ti.obv,
        ti.stoch,
        ti.stoch_signal
    FROM market.daily_prices p
    LEFT JOIN market.technical_indicators ti 
        ON p.ticker = ti.ticker AND p.date = ti.date
    WHERE ti.sma_20 IS NOT NULL
    ORDER BY p.ticker, p.date
    """
    
    df = db.execute_query(query)
    print(f"✅ Loaded {len(df):,} records for {df['ticker'].nunique()} stocks")
    
    return df

def engineer_features(df):
    """Create additional features"""
    print("\n🔧 Engineering features...")
    
    df = df.sort_values(['ticker', 'date']).copy()
    
    # Price-based features
    df['return_1d'] = df.groupby('ticker')['price'].pct_change(1)
    df['return_5d'] = df.groupby('ticker')['price'].pct_change(5)
    df['return_10d'] = df.groupby('ticker')['price'].pct_change(10)
    
    # Price ratios
    df['price_to_sma20'] = df['price'] / df['sma_20']
    df['price_to_sma50'] = df['price'] / df['sma_50']
    df['sma20_to_sma50'] = df['sma_20'] / df['sma_50']
    
    # Volume features
    df['volume_ma5'] = df.groupby('ticker')['volume'].transform(lambda x: x.rolling(5).mean())
    df['volume_ratio'] = df['volume'] / df['volume_ma5']
    
    # Volatility
    df['volatility_5d'] = df.groupby('ticker')['return_1d'].transform(lambda x: x.rolling(5).std())
    df['volatility_20d'] = df.groupby('ticker')['return_1d'].transform(lambda x: x.rolling(20).std())
    
    # High-Low range
    df['hl_ratio'] = (df['high'] - df['low']) / df['low']
    
    print(f"✅ Created {len(df.columns)} total features")
    
    return df

def create_targets(df, horizons):
    """Create forward return targets"""
    print("\n🎯 Creating target variables...")
    
    df = df.sort_values(['ticker', 'date']).copy()
    
    for name, days in horizons.items():
        df[f'target_{name}'] = df.groupby('ticker')['price'].transform(
            lambda x: x.pct_change(periods=days).shift(-days)
        )
    
    print(f"✅ Created {len(horizons)} target variables")
    
    return df

def train_models(df, feature_cols, horizons):
    """Train XGBoost models for each horizon"""
    print("\n🤖 Training ML models...")
    
    models = {}
    metrics = {}
    
    for name, days in horizons.items():
        target_col = f'target_{name}'
        
        print(f"\n{'='*60}")
        print(f"Training {name} Model ({days} days)")
        print(f"{'='*60}")
        
        # Prepare data
        X, y = prepare_features(df, feature_cols, target_col, dropna=True)
        
        if len(X) < CONFIG['MIN_TRAINING_SAMPLES']:
            print(f"⚠️  Insufficient data: {len(X)} samples")
            continue
        
        print(f"📊 Training samples: {len(X):,}")
        
        # Split train/test (80/20 time-based)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # Train model
        model = train_model(X_train, y_train, model_type='xgboost', params={
            'n_estimators': 100,
            'max_depth': 5,
            'learning_rate': 0.1,
            'random_state': 42
        })
        
        # Evaluate
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        train_metrics = evaluate_model(y_train, y_pred_train, return_dict=True)
        test_metrics = evaluate_model(y_test, y_pred_test, return_dict=True)
        
        print(f"\n📊 Training Performance:")
        for metric, value in train_metrics.items():
            print(f"   {metric}: {value:.4f}")
        
        print(f"\n📊 Test Performance:")
        for metric, value in test_metrics.items():
            print(f"   {metric}: {value:.4f}")
        
        models[name] = model
        metrics[name] = {
            'train': train_metrics,
            'test': test_metrics,
            'samples': len(X)
        }
    
    return models, metrics

def generate_predictions(df, models, feature_cols):
    """Generate predictions for all stocks"""
    print("\n🔮 Generating predictions...")
    
    # Get latest data for each stock
    latest_df = df.sort_values('date').groupby('ticker').tail(1).copy()
    
    # Prepare features
    X = latest_df[feature_cols].copy()
    X = X.fillna(X.median())
    
    # Generate predictions
    predictions = latest_df[['ticker', 'date', 'price']].copy()
    
    for name, model in models.items():
        predictions[f'pred_return_{name}'] = model.predict(X)
    
    print(f"✅ Generated predictions for {len(predictions)} stocks")
    
    return predictions

def rank_and_select(predictions, top_n):
    """Rank stocks by predicted returns"""
    print(f"\n🏆 Ranking and selecting top {top_n} stocks...")
    
    # Calculate composite score (average of predictions)
    pred_cols = [col for col in predictions.columns if col.startswith('pred_return_')]
    predictions['composite_pred'] = predictions[pred_cols].mean(axis=1)
    
    # Rank
    predictions = predictions.sort_values('composite_pred', ascending=False)
    predictions['rank'] = range(1, len(predictions) + 1)
    
    # Select top N
    top_stocks = predictions.head(top_n)
    
    print(f"✅ Selected top {len(top_stocks)} stocks")
    
    return predictions, top_stocks

def save_results(predictions, top_stocks, models, metrics):
    """Save predictions and models"""
    print("\n💾 Saving results...")
    
    # Create output directories
    output_dir = project_root / 'data' / 'processed'
    models_dir = project_root / 'models' / 'return_predictors'
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    
    # Save predictions
    pred_file = output_dir / f'ml_predictions_{timestamp}.csv'
    predictions.to_csv(pred_file, index=False)
    print(f"✅ Saved predictions: {pred_file.name}")
    
    # Save top stocks
    top_n = CONFIG['TOP_N']
    top_file = output_dir / f'ml_top_{top_n}_{timestamp}.csv'
    top_stocks.to_csv(top_file, index=False)
    print(f"✅ Saved top {top_n}: {top_file.name}")
    
    # Save models
    for name, model in models.items():
        model_file = models_dir / f'xgboost_{name}_{timestamp}.pkl'
        joblib.dump(model, model_file)
        print(f"✅ Saved model: {model_file.name}")
    
    # Save metrics
    metrics_file = output_dir / f'ml_metrics_{timestamp}.txt'
    with open(metrics_file, 'w') as f:
        f.write("ML MODEL PERFORMANCE METRICS\n")
        f.write("=" * 60 + "\n\n")
        for name, metric in metrics.items():
            f.write(f"{name} Model:\n")
            f.write(f"  Training samples: {metric['samples']:,}\n")
            f.write(f"  Train Performance:\n")
            for k, v in metric['train'].items():
                f.write(f"    {k}: {v:.4f}\n")
            f.write(f"  Test Performance:\n")
            for k, v in metric['test'].items():
                f.write(f"    {k}: {v:.4f}\n")
            f.write("\n")
    
    print(f"✅ Saved metrics: {metrics_file.name}")
    
    return pred_file, top_file

def main():
    """Main execution function"""
    print("=" * 80)
    print("🤖 SMARTPORTFOLIO - ML RETURN PREDICTION")
    print("=" * 80)
    
    try:
        # Connect to database
        print("\n📂 Connecting to database...")
        db = DatabaseConnector()
        
        if not db.test_connection():
            raise Exception("Database connection failed!")
        
        print("✅ Database connected")
        
        # Load data
        df = load_data(db)
        
        # Engineer features
        df = engineer_features(df)
        
        # Create targets
        df = create_targets(df, CONFIG['HORIZONS'])
        
        # Define feature columns
        feature_cols = [
            'sma_20', 'sma_50', 'ema_12', 'ema_26',
            'rsi', 'macd', 'macd_signal', 'macd_diff',
            'bb_upper', 'bb_middle', 'bb_lower', 'bb_width',
            'atr', 'adx', 'obv', 'stoch', 'stoch_signal',
            'return_1d', 'return_5d', 'return_10d',
            'price_to_sma20', 'price_to_sma50', 'sma20_to_sma50',
            'volume_ratio', 'volatility_5d', 'volatility_20d',
            'hl_ratio'
        ]
        
        # Train models
        models, metrics = train_models(df, feature_cols, CONFIG['HORIZONS'])
        
        if not models:
            raise Exception("No models were trained successfully!")
        
        # Generate predictions
        predictions = generate_predictions(df, models, feature_cols)
        
        # Rank and select
        predictions, top_stocks = rank_and_select(predictions, CONFIG['TOP_N'])
        
        # Save results
        pred_file, top_file = save_results(predictions, top_stocks, models, metrics)
        
        # Summary
        print("\n" + "=" * 80)
        print("✅ ML PREDICTION COMPLETE!")
        print("=" * 80)
        print(f"\n📊 Summary:")
        print(f"  • Models trained: {len(models)}")
        print(f"  • Predictions generated: {len(predictions)}")
        print(f"  • Top stocks selected: {len(top_stocks)}")
        print(f"\n🏆 Top 10 Predicted Stocks:")
        print(top_stocks[['rank', 'ticker', 'price', 'composite_pred']].head(10).to_string(index=False))
        print(f"\n💾 Results saved to:")
        print(f"  • {pred_file.name}")
        print(f"  • {top_file.name}")
        print(f"\n🎯 Next step: Run 'python scripts/run_optimization.py'")
        
        db.close()
        return 0
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())

