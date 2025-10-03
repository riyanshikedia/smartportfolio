"""
Tests for ML Models Module
"""

import pytest
import pandas as pd
import numpy as np
from utils.ml_helpers import (
    prepare_features,
    train_model,
    evaluate_model,
    get_feature_importance
)


class TestFeaturePreparation:
    """Test feature preparation functions"""
    
    def test_prepare_features_with_target(self):
        """Test feature preparation with target"""
        df = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [10, 20, 30, 40, 50],
            'target': [100, 200, 300, 400, 500]
        })
        
        X, y = prepare_features(
            df,
            feature_cols=['feature1', 'feature2'],
            target_col='target'
        )
        
        assert len(X) == 5
        assert len(y) == 5
        assert list(X.columns) == ['feature1', 'feature2']
    
    def test_prepare_features_with_nan(self):
        """Test feature preparation with NaN handling"""
        df = pd.DataFrame({
            'feature1': [1, 2, np.nan, 4, 5],
            'feature2': [10, 20, 30, 40, 50],
            'target': [100, 200, 300, np.nan, 500]
        })
        
        X, y = prepare_features(
            df,
            feature_cols=['feature1', 'feature2'],
            target_col='target',
            dropna=True
        )
        
        # Should drop row with NaN target
        assert len(X) == 4
        assert len(y) == 4
        
        # Feature NaN should be filled with median
        assert not X['feature1'].isna().any()


class TestModelTraining:
    """Test model training functions"""
    
    def test_train_xgboost_model(self):
        """Test XGBoost model training"""
        # Create sample data
        np.random.seed(42)
        X_train = pd.DataFrame({
            'feature1': np.random.rand(100),
            'feature2': np.random.rand(100)
        })
        y_train = pd.Series(X_train['feature1'] * 2 + X_train['feature2'] + np.random.rand(100) * 0.1)
        
        model = train_model(X_train, y_train, model_type='xgboost')
        
        assert model is not None
        assert hasattr(model, 'predict')
        
        # Test prediction
        y_pred = model.predict(X_train)
        assert len(y_pred) == len(y_train)
    
    def test_train_randomforest_model(self):
        """Test Random Forest model training"""
        np.random.seed(42)
        X_train = pd.DataFrame({
            'feature1': np.random.rand(100),
            'feature2': np.random.rand(100)
        })
        y_train = pd.Series(X_train['feature1'] * 2 + X_train['feature2'])
        
        model = train_model(X_train, y_train, model_type='randomforest')
        
        assert model is not None
        assert hasattr(model, 'predict')


class TestModelEvaluation:
    """Test model evaluation functions"""
    
    def test_evaluate_model(self):
        """Test model evaluation metrics"""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1.1, 2.2, 2.9, 3.8, 5.1])
        
        rmse, mae, r2, dir_acc = evaluate_model(y_true, y_pred)
        
        assert rmse > 0
        assert mae > 0
        assert 0 <= r2 <= 1
        assert 0 <= dir_acc <= 1
    
    def test_evaluate_model_dict(self):
        """Test model evaluation with dict return"""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1.1, 2.2, 2.9, 3.8, 5.1])
        
        metrics = evaluate_model(y_true, y_pred, return_dict=True)
        
        assert 'RMSE' in metrics
        assert 'MAE' in metrics
        assert 'R²' in metrics
        assert 'Directional Accuracy' in metrics


class TestFeatureImportance:
    """Test feature importance extraction"""
    
    def test_get_feature_importance(self):
        """Test feature importance extraction"""
        # Train a simple model
        np.random.seed(42)
        X_train = pd.DataFrame({
            'feature1': np.random.rand(100),
            'feature2': np.random.rand(100),
            'feature3': np.random.rand(100)
        })
        y_train = pd.Series(X_train['feature1'] * 3 + X_train['feature2'] * 0.5)
        
        model = train_model(X_train, y_train, model_type='xgboost')
        
        importance_df = get_feature_importance(
            model,
            feature_names=['feature1', 'feature2', 'feature3'],
            top_n=3
        )
        
        assert len(importance_df) == 3
        assert 'feature' in importance_df.columns
        assert 'importance' in importance_df.columns
        
        # feature1 should have highest importance
        assert importance_df.iloc[0]['feature'] == 'feature1'


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])

