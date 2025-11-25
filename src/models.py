"""
ML Models for Cyberattack Early-Warning System
Classification models with anomaly detection for risk level forecasting.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import pickle
import warnings
warnings.filterwarnings('ignore')

# scikit-learn imports
from sklearn.ensemble import (
    RandomForestClassifier, 
    GradientBoostingClassifier,
    IsolationForest,
    VotingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import (
    train_test_split, 
    cross_val_score,
    StratifiedKFold,
    TimeSeriesSplit
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
import joblib


class RiskClassifier:
    """Multi-class risk level classifier with ensemble methods."""
    
    def __init__(self, model_type: str = "ensemble", config: Dict = None):
        self.model_type = model_type
        self.config = config or {}
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.is_fitted = False
        
    def _create_model(self):
        """Create the classification model based on type."""
        if self.model_type == "random_forest":
            return RandomForestClassifier(
                n_estimators=self.config.get('n_estimators', 100),
                max_depth=self.config.get('max_depth', 10),
                min_samples_split=self.config.get('min_samples_split', 5),
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            )
        
        elif self.model_type == "gradient_boosting":
            return GradientBoostingClassifier(
                n_estimators=self.config.get('n_estimators', 100),
                learning_rate=self.config.get('learning_rate', 0.1),
                max_depth=self.config.get('max_depth', 5),
                random_state=42
            )
        
        elif self.model_type == "logistic":
            return LogisticRegression(
                C=self.config.get('C', 1.0),
                class_weight='balanced',
                max_iter=1000,
                random_state=42
            )
        
        elif self.model_type == "mlp":
            return MLPClassifier(
                hidden_layer_sizes=self.config.get('hidden_layers', (100, 50)),
                learning_rate='adaptive',
                max_iter=500,
                random_state=42
            )
        
        elif self.model_type == "ensemble":
            # Voting ensemble of multiple classifiers
            rf = RandomForestClassifier(
                n_estimators=100, max_depth=10, 
                class_weight='balanced', random_state=42, n_jobs=-1
            )
            gb = GradientBoostingClassifier(
                n_estimators=100, learning_rate=0.1, 
                max_depth=5, random_state=42
            )
            lr = LogisticRegression(
                class_weight='balanced', max_iter=1000, random_state=42
            )
            
            return VotingClassifier(
                estimators=[('rf', rf), ('gb', gb), ('lr', lr)],
                voting='soft'
            )
        
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def fit(self, X: pd.DataFrame, y: pd.Series, 
            feature_names: List[str] = None) -> 'RiskClassifier':
        """Fit the classifier."""
        self.feature_names = feature_names or list(X.columns)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Create and fit model
        self.model = self._create_model()
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict risk levels."""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict risk level probabilities."""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance scores."""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
        
        importance = None
        
        if self.model_type == "random_forest":
            importance = self.model.feature_importances_
        elif self.model_type == "gradient_boosting":
            importance = self.model.feature_importances_
        elif self.model_type == "logistic":
            # Use absolute coefficient values as importance
            importance = np.abs(self.model.coef_).mean(axis=0)
        elif self.model_type == "ensemble":
            # Average importance from RF and GB
            rf_importance = self.model.named_estimators_['rf'].feature_importances_
            gb_importance = self.model.named_estimators_['gb'].feature_importances_
            importance = (rf_importance + gb_importance) / 2
        
        if importance is not None:
            return pd.DataFrame({
                'feature': self.feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)
        
        return pd.DataFrame()
    
    def save(self, path: str):
        """Save model to disk."""
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'model_type': self.model_type,
            'config': self.config
        }, path)
    
    @classmethod
    def load(cls, path: str) -> 'RiskClassifier':
        """Load model from disk."""
        data = joblib.load(path)
        classifier = cls(model_type=data['model_type'], config=data['config'])
        classifier.model = data['model']
        classifier.scaler = data['scaler']
        classifier.feature_names = data['feature_names']
        classifier.is_fitted = True
        return classifier


class AnomalySignalDetector:
    """Isolation Forest-based anomaly detector for exploit signal bursts."""
    
    def __init__(self, contamination: float = 0.1):
        self.contamination = contamination
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def fit(self, X: pd.DataFrame) -> 'AnomalySignalDetector':
        """Fit the anomaly detector."""
        X_scaled = self.scaler.fit_transform(X)
        
        self.model = IsolationForest(
            n_estimators=100,
            contamination=self.contamination,
            random_state=42,
            n_jobs=-1
        )
        self.model.fit(X_scaled)
        self.is_fitted = True
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict anomalies (-1 for anomaly, 1 for normal)."""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def score_samples(self, X: pd.DataFrame) -> np.ndarray:
        """Get anomaly scores (lower = more anomalous)."""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
        
        X_scaled = self.scaler.transform(X)
        return self.model.score_samples(X_scaled)
    
    def get_anomaly_probability(self, X: pd.DataFrame) -> np.ndarray:
        """Convert anomaly scores to pseudo-probabilities."""
        scores = self.score_samples(X)
        # Normalize to 0-1 range (inverted so higher = more anomalous)
        min_score, max_score = scores.min(), scores.max()
        if max_score - min_score > 0:
            probs = 1 - (scores - min_score) / (max_score - min_score)
        else:
            probs = np.zeros_like(scores)
        return probs


class HybridRiskPredictor:
    """
    Hybrid model combining classification and anomaly detection.
    Uses anomaly scores as additional features for risk prediction.
    """
    
    def __init__(self, 
                 classifier_type: str = "ensemble",
                 anomaly_contamination: float = 0.1):
        self.classifier = RiskClassifier(model_type=classifier_type)
        self.anomaly_detector = AnomalySignalDetector(contamination=anomaly_contamination)
        self.anomaly_features = None
        self.is_fitted = False
        
    def fit(self, X: pd.DataFrame, y: pd.Series,
            anomaly_feature_cols: List[str] = None) -> 'HybridRiskPredictor':
        """
        Fit both the anomaly detector and classifier.
        
        Args:
            X: Feature matrix
            y: Target labels
            anomaly_feature_cols: Columns to use for anomaly detection
        """
        # Determine anomaly features
        if anomaly_feature_cols is None:
            # Use volume/count features for anomaly detection
            self.anomaly_features = [c for c in X.columns 
                                    if any(kw in c.lower() for kw in 
                                          ['count', 'total', 'attacks', 'burst'])]
        else:
            self.anomaly_features = anomaly_feature_cols
        
        # Fit anomaly detector on subset of features
        if self.anomaly_features:
            anomaly_X = X[self.anomaly_features]
            self.anomaly_detector.fit(anomaly_X)
            
            # Get anomaly scores and add as features
            anomaly_scores = self.anomaly_detector.get_anomaly_probability(anomaly_X)
            X_enhanced = X.copy()
            X_enhanced['anomaly_probability'] = anomaly_scores
        else:
            X_enhanced = X.copy()
        
        # Fit classifier on enhanced features
        self.classifier.fit(X_enhanced, y, list(X_enhanced.columns))
        self.is_fitted = True
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict risk levels."""
        X_enhanced = self._enhance_features(X)
        return self.classifier.predict(X_enhanced)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict risk probabilities."""
        X_enhanced = self._enhance_features(X)
        return self.classifier.predict_proba(X_enhanced)
    
    def _enhance_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Add anomaly scores to features."""
        X_enhanced = X.copy()
        
        if self.anomaly_features:
            anomaly_X = X[self.anomaly_features]
            anomaly_scores = self.anomaly_detector.get_anomaly_probability(anomaly_X)
            X_enhanced['anomaly_probability'] = anomaly_scores
        
        return X_enhanced
    
    def get_prediction_explanation(self, X: pd.DataFrame) -> pd.DataFrame:
        """Get detailed prediction with explanation."""
        X_enhanced = self._enhance_features(X)
        
        predictions = self.classifier.predict(X_enhanced)
        probabilities = self.classifier.predict_proba(X_enhanced)
        
        # Get anomaly assessment
        if self.anomaly_features:
            anomaly_scores = self.anomaly_detector.get_anomaly_probability(X[self.anomaly_features])
        else:
            anomaly_scores = np.zeros(len(X))
        
        risk_labels = ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']
        
        results = []
        for i in range(len(X)):
            result = {
                'predicted_risk_level': risk_labels[int(predictions[i])],
                'confidence': float(probabilities[i].max()),
                'anomaly_score': float(anomaly_scores[i]),
                'risk_probabilities': {
                    risk_labels[j]: float(probabilities[i][j]) 
                    for j in range(len(risk_labels)) if j < len(probabilities[i])
                }
            }
            results.append(result)
        
        return pd.DataFrame(results)
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance from the classifier."""
        return self.classifier.get_feature_importance()
    
    def save(self, path: str):
        """Save the hybrid model."""
        joblib.dump({
            'classifier': self.classifier,
            'anomaly_detector': self.anomaly_detector,
            'anomaly_features': self.anomaly_features
        }, path)
    
    @classmethod
    def load(cls, path: str) -> 'HybridRiskPredictor':
        """Load the hybrid model."""
        data = joblib.load(path)
        predictor = cls()
        predictor.classifier = data['classifier']
        predictor.anomaly_detector = data['anomaly_detector']
        predictor.anomaly_features = data['anomaly_features']
        predictor.is_fitted = True
        return predictor


class TemporalCrossValidator:
    """Time-series aware cross-validation for the risk models."""
    
    def __init__(self, n_splits: int = 5):
        self.n_splits = n_splits
        
    def validate(self, model: HybridRiskPredictor, 
                X: pd.DataFrame, y: pd.Series) -> Dict[str, List[float]]:
        """
        Perform temporal cross-validation.
        Uses expanding window to respect time ordering.
        """
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        
        metrics = {
            'accuracy': [],
            'precision_macro': [],
            'recall_macro': [],
            'f1_macro': []
        }
        
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Clone model for this fold
            fold_model = HybridRiskPredictor(
                classifier_type=model.classifier.model_type
            )
            fold_model.fit(X_train, y_train, model.anomaly_features)
            
            y_pred = fold_model.predict(X_test)
            
            metrics['accuracy'].append(accuracy_score(y_test, y_pred))
            metrics['precision_macro'].append(
                precision_score(y_test, y_pred, average='macro', zero_division=0)
            )
            metrics['recall_macro'].append(
                recall_score(y_test, y_pred, average='macro', zero_division=0)
            )
            metrics['f1_macro'].append(
                f1_score(y_test, y_pred, average='macro', zero_division=0)
            )
        
        return metrics


class ModelSelector:
    """Compare and select the best model for risk prediction."""
    
    def __init__(self):
        self.model_types = ['random_forest', 'gradient_boosting', 'logistic', 'ensemble']
        self.results = {}
        
    def compare_models(self, X: pd.DataFrame, y: pd.Series, 
                      cv_splits: int = 5) -> pd.DataFrame:
        """Compare different model types using cross-validation."""
        validator = TemporalCrossValidator(n_splits=cv_splits)
        
        for model_type in self.model_types:
            print(f"Evaluating {model_type}...")
            model = HybridRiskPredictor(classifier_type=model_type)
            metrics = validator.validate(model, X, y)
            
            self.results[model_type] = {
                metric: np.mean(values) for metric, values in metrics.items()
            }
            self.results[model_type]['std_f1'] = np.std(metrics['f1_macro'])
        
        return pd.DataFrame(self.results).T
    
    def get_best_model(self, metric: str = 'f1_macro') -> str:
        """Return the best model type based on specified metric."""
        if not self.results:
            raise ValueError("Run compare_models first")
        
        return max(self.results.keys(), key=lambda k: self.results[k][metric])


if __name__ == "__main__":
    # Test the models
    import sys
    sys.path.append('..')
    from data_loaders import DataAggregator
    from feature_engineering import FeatureEngineeringPipeline
    from config import DATA_DIR
    
    # Load and prepare data
    print("Loading data...")
    aggregator = DataAggregator(DATA_DIR)
    raw_data = aggregator.load_all()
    weekly_df = aggregator.create_weekly_aggregation(raw_data)
    
    print("Engineering features...")
    pipeline = FeatureEngineeringPipeline()
    features_df = pipeline.fit_transform(weekly_df, raw_data)
    
    # Prepare X and y
    target_col = 'next_week_risk_level'
    exclude_cols = ['week_start', 'next_week_risk_score', 'next_week_risk_level', 
                    'next_week_elevated_risk', 'current_risk_score']
    
    # Drop rows with NaN target
    features_df = features_df.dropna(subset=[target_col])
    
    feature_cols = [c for c in features_df.columns if c not in exclude_cols]
    X = features_df[feature_cols]
    y = features_df[target_col].astype(int)
    
    print(f"\nDataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Class distribution:\n{y.value_counts()}")
    
    # Compare models
    print("\n" + "="*50)
    print("Model Comparison")
    print("="*50)
    
    selector = ModelSelector()
    comparison = selector.compare_models(X, y, cv_splits=3)
    print(comparison)
    
    best_model_type = selector.get_best_model()
    print(f"\nBest model: {best_model_type}")
    
    # Train final model
    print("\n" + "="*50)
    print("Training Final Model")
    print("="*50)
    
    final_model = HybridRiskPredictor(classifier_type=best_model_type)
    final_model.fit(X, y)
    
    # Feature importance
    importance = final_model.get_feature_importance()
    print("\nTop 15 Feature Importances:")
    print(importance.head(15))
