"""
Machine Learning Helper Functions
Utility functions for ML model training, evaluation, and prediction
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
from pathlib import Path


def prepare_features(df, feature_cols, target_col=None, dropna=True):
    """
    Prepare features and target for ML
    
    Args:
        df (pd.DataFrame): Raw data
        feature_cols (list): Feature column names
        target_col (str): Target column name (optional)
        dropna (bool): Whether to drop NaN rows
        
    Returns:
        tuple: (X, y) or just X if no target
    """
    X = df[feature_cols].copy()
    
    if target_col:
        y = df[target_col].copy()
        
        if dropna:
            # Only drop rows where target is NaN
            valid_idx = ~y.isna()
            X = X[valid_idx]
            y = y[valid_idx]
            
            # Fill remaining NaN in features with median
            X = X.fillna(X.median())
        
        return X, y
    else:
        if dropna:
            X = X.fillna(X.median())
        return X


def train_model(X_train, y_train, model_type='xgboost', params=None):
    """
    Train ML model
    
    Args:
        X_train: Training features
        y_train: Training target
        model_type (str): 'xgboost', 'lightgbm', or 'randomforest'
        params (dict): Model parameters
        
    Returns:
        Trained model
    """
    if model_type == 'xgboost':
        from xgboost import XGBRegressor
        default_params = {
            'n_estimators': 100,
            'max_depth': 5,
            'learning_rate': 0.1,
            'random_state': 42
        }
        params = {**default_params, **(params or {})}
        model = XGBRegressor(**params)
        
    elif model_type == 'lightgbm':
        from lightgbm import LGBMRegressor
        default_params = {
            'n_estimators': 100,
            'max_depth': 5,
            'learning_rate': 0.1,
            'random_state': 42
        }
        params = {**default_params, **(params or {})}
        model = LGBMRegressor(**params)
        
    elif model_type == 'randomforest':
        from sklearn.ensemble import RandomForestRegressor
        default_params = {
            'n_estimators': 100,
            'max_depth': 10,
            'random_state': 42
        }
        params = {**default_params, **(params or {})}
        model = RandomForestRegressor(**params)
        
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model.fit(X_train, y_train)
    return model


def evaluate_model(y_true, y_pred, return_dict=False):
    """
    Evaluate model performance
    
    Args:
        y_true: True values
        y_pred: Predicted values
        return_dict (bool): Return as dictionary
        
    Returns:
        dict or tuple: Metrics
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # Directional accuracy
    direction_true = np.sign(y_true)
    direction_pred = np.sign(y_pred)
    directional_accuracy = (direction_true == direction_pred).mean()
    
    if return_dict:
        return {
            'RMSE': rmse,
            'MAE': mae,
            'R²': r2,
            'Directional Accuracy': directional_accuracy
        }
    else:
        return rmse, mae, r2, directional_accuracy


def get_feature_importance(model, feature_names, top_n=10):
    """
    Get feature importance from trained model
    
    Args:
        model: Trained model
        feature_names (list): Feature column names
        top_n (int): Number of top features to return
        
    Returns:
        pd.DataFrame: Feature importance
    """
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importance = np.abs(model.coef_)
    else:
        return None
    
    df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    })
    
    df = df.sort_values('importance', ascending=False)
    
    if top_n:
        df = df.head(top_n)
    
    return df


def save_model(model, filepath, scaler=None):
    """
    Save trained model and scaler
    
    Args:
        model: Trained model
        filepath (str): Path to save model
        scaler: Fitted scaler (optional)
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    joblib.dump(model, filepath)
    print(f"✅ Model saved: {filepath.name}")
    
    if scaler:
        scaler_path = filepath.parent / f"scaler_{filepath.stem}.pkl"
        joblib.dump(scaler, scaler_path)
        print(f"✅ Scaler saved: {scaler_path.name}")


def load_model(filepath, load_scaler=False):
    """
    Load trained model and optionally scaler
    
    Args:
        filepath (str): Path to model file
        load_scaler (bool): Whether to load scaler
        
    Returns:
        model or tuple: (model, scaler)
    """
    filepath = Path(filepath)
    model = joblib.load(filepath)
    
    if load_scaler:
        scaler_path = filepath.parent / f"scaler_{filepath.stem}.pkl"
        if scaler_path.exists():
            scaler = joblib.load(scaler_path)
            return model, scaler
        else:
            print(f"⚠️  Scaler not found: {scaler_path}")
            return model, None
    
    return model


def cross_validate_time_series(X, y, n_splits=5, model_type='xgboost'):
    """
    Time series cross-validation
    
    Args:
        X: Features
        y: Target
        n_splits (int): Number of splits
        model_type (str): Model type
        
    Returns:
        list: Scores for each split
    """
    from sklearn.model_selection import TimeSeriesSplit
    
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = []
    
    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        model = train_model(X_train, y_train, model_type=model_type)
        y_pred = model.predict(X_test)
        
        metrics = evaluate_model(y_test, y_pred, return_dict=True)
        scores.append(metrics)
    
    # Average scores
    avg_scores = {k: np.mean([s[k] for s in scores]) for k in scores[0].keys()}
    
    return avg_scores


def predict_with_confidence(model, X, confidence=0.95):
    """
    Make predictions with confidence intervals (for ensemble models)
    
    Args:
        model: Trained model
        X: Features
        confidence (float): Confidence level
        
    Returns:
        tuple: (predictions, lower_bound, upper_bound)
    """
    # This works for RandomForest and similar ensemble models
    if hasattr(model, 'estimators_'):
        # Get predictions from all trees
        all_preds = np.array([tree.predict(X) for tree in model.estimators_])
        
        # Calculate mean and confidence intervals
        predictions = all_preds.mean(axis=0)
        lower = np.percentile(all_preds, (1 - confidence) / 2 * 100, axis=0)
        upper = np.percentile(all_preds, (1 + confidence) / 2 * 100, axis=0)
        
        return predictions, lower, upper
    else:
        # For non-ensemble models, return predictions only
        predictions = model.predict(X)
        return predictions, None, None

