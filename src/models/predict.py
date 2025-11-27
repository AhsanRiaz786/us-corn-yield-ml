"""
Model prediction utilities for production deployment.

Provides functions for loading trained models and generating predictions
on new data.
"""

import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union, Dict, Tuple


class YieldPredictor:
    """Production-ready yield prediction interface."""
    
    def __init__(self, model_name: str = 'xgboost'):
        """
        Initialize predictor with specified model.
        
        Args:
            model_name: Name of model to load ('xgboost', 'randomforest', 
                       'gradientboosting', 'ridge')
        """
        self.model_dir = Path(__file__).parent.parent.parent / "models"
        self.model_name = model_name
        
        self.model = self._load_model()
        self.scaler = self._load_scaler()
        self.feature_columns = self._load_feature_columns()
    
    def _load_model(self):
        """Load trained model from disk."""
        model_path = self.model_dir / f"{self.model_name}_model.pkl"
        
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {model_path}\n"
                f"Available models: xgboost, randomforest, gradientboosting, ridge"
            )
        
        return joblib.load(model_path)
    
    def _load_scaler(self):
        """Load feature scaler."""
        scaler_path = self.model_dir / "scaler.pkl"
        
        if not scaler_path.exists():
            raise FileNotFoundError(f"Scaler file not found: {scaler_path}")
        
        return joblib.load(scaler_path)
    
    def _load_feature_columns(self):
        """Load expected feature column names."""
        columns_path = self.model_dir / "feature_columns.pkl"
        
        if not columns_path.exists():
            raise FileNotFoundError(
                f"Feature columns file not found: {columns_path}"
            )
        
        return joblib.load(columns_path)
    
    def validate_input(self, X: pd.DataFrame) -> Tuple[bool, str]:
        """
        Validate input data format and features.
        
        Args:
            X: Input feature DataFrame
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        missing_cols = set(self.feature_columns) - set(X.columns)
        if missing_cols:
            return False, f"Missing required features: {missing_cols}"
        
        extra_cols = set(X.columns) - set(self.feature_columns)
        if extra_cols:
            return True, f"Warning: Extra columns will be ignored: {extra_cols}"
        
        return True, ""
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray], 
                return_confidence: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Generate yield predictions.
        
        Args:
            X: Input features (DataFrame or array)
            return_confidence: If True, return prediction intervals for tree-based models
            
        Returns:
            Predicted yields (and optionally prediction intervals)
        """
        if isinstance(X, pd.DataFrame):
            is_valid, message = self.validate_input(X)
            if not is_valid:
                raise ValueError(message)
            
            X_subset = X[self.feature_columns].values
        else:
            X_subset = X
        
        X_scaled = self.scaler.transform(X_subset)
        
        predictions = self.model.predict(X_scaled)
        
        if return_confidence and hasattr(self.model, 'estimators_'):
            tree_predictions = np.array([
                estimator.predict(X_scaled) 
                for estimator in self.model.estimators_
            ])
            
            prediction_std = np.std(tree_predictions, axis=0)
            confidence_lower = predictions - 1.96 * prediction_std
            confidence_upper = predictions + 1.96 * prediction_std
            
            return predictions, (confidence_lower, confidence_upper)
        
        return predictions
    
    def predict_single(self, features: Dict[str, float]) -> float:
        """
        Generate prediction for a single observation.
        
        Args:
            features: Dictionary mapping feature names to values
            
        Returns:
            Predicted yield
        """
        X = pd.DataFrame([features])
        return self.predict(X)[0]
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance scores.
        
        Returns:
            DataFrame with features and their importance scores
        """
        if not hasattr(self.model, 'feature_importances_'):
            raise AttributeError(
                f"{self.model_name} model does not support feature importance"
            )
        
        importance_df = pd.DataFrame({
            'Feature': self.feature_columns,
            'Importance': self.model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        return importance_df


def load_model(model_name: str = 'xgboost'):
    """
    Convenience function to load a trained model.
    
    Args:
        model_name: Name of model to load
        
    Returns:
        Loaded model object
    """
    model_dir = Path(__file__).parent.parent.parent / "models"
    model_path = model_dir / f"{model_name}_model.pkl"
    
    return joblib.load(model_path)


def make_prediction(X: pd.DataFrame, model_name: str = 'xgboost') -> np.ndarray:
    """
    Quick prediction function for simple use cases.
    
    Args:
        X: Input features
        model_name: Model to use for prediction
        
    Returns:
        Predicted yields
    """
    predictor = YieldPredictor(model_name=model_name)
    return predictor.predict(X)

