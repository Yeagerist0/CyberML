"""
Main Pipeline for Cyberattack Early-Warning System
End-to-end orchestration for risk forecasting.
"""
import sys
import os
import argparse
from pathlib import Path
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from config import (
    DATA_DIR, MODELS_DIR, OUTPUTS_DIR,
    MODEL_CONFIG, EVALUATION_CONFIG, TARGET_SECTORS
)
from src.data_loaders import DataAggregator
from src.feature_engineering import FeatureEngineeringPipeline
from src.models import HybridRiskPredictor, ModelSelector
from src.evaluation import ComprehensiveEvaluationReport


class CyberAttackEarlyWarningPipeline:
    """
    End-to-end pipeline for cyberattack early warning system.
    
    Forecasts next-week risk level of attacks on public-facing services
    using OSINT signals and adversary activity indicators.
    """
    
    def __init__(self, data_dir: Path = None, output_dir: Path = None):
        self.data_dir = data_dir or DATA_DIR
        self.output_dir = output_dir or OUTPUTS_DIR
        self.models_dir = MODELS_DIR
        
        # Pipeline components
        self.aggregator = None
        self.feature_pipeline = None
        self.model = None
        self.reporter = None
        
        # Data
        self.raw_data = None
        self.weekly_data = None
        self.features_df = None
        
        # Ensure directories exist
        self.output_dir.mkdir(exist_ok=True)
        self.models_dir.mkdir(exist_ok=True)
        
    def load_data(self) -> 'CyberAttackEarlyWarningPipeline':
        """Load data from all OSINT sources."""
        print("\n" + "="*60)
        print("STEP 1: DATA LOADING")
        print("="*60)
        
        self.aggregator = DataAggregator(self.data_dir)
        self.raw_data = self.aggregator.load_all()
        
        print("\nData sources loaded:")
        for name, df in self.raw_data.items():
            print(f"  {name}: {len(df)} records")
        
        return self
    
    def aggregate_weekly(self) -> 'CyberAttackEarlyWarningPipeline':
        """Aggregate data into weekly feature vectors."""
        print("\n" + "="*60)
        print("STEP 2: WEEKLY AGGREGATION")
        print("="*60)
        
        self.weekly_data = self.aggregator.create_weekly_aggregation(self.raw_data)
        print(f"Created {len(self.weekly_data)} weekly records with {len(self.weekly_data.columns)} base features")
        
        return self
    
    def engineer_features(self) -> 'CyberAttackEarlyWarningPipeline':
        """Apply feature engineering pipeline."""
        print("\n" + "="*60)
        print("STEP 3: FEATURE ENGINEERING")
        print("="*60)
        
        self.feature_pipeline = FeatureEngineeringPipeline()
        self.features_df = self.feature_pipeline.fit_transform(self.weekly_data, self.raw_data)
        
        print(f"Final feature set: {self.features_df.shape[0]} samples, {self.features_df.shape[1]} features")
        
        # Show feature groups
        feature_groups = self.feature_pipeline.get_feature_importance_groups()
        print("\nFeature Groups:")
        for group_name, features in feature_groups.items():
            available = [f for f in features if f in self.features_df.columns]
            print(f"  {group_name}: {len(available)} features")
        
        return self
    
    def train_model(self, model_type: str = 'ensemble', 
                   compare_models: bool = False) -> 'CyberAttackEarlyWarningPipeline':
        """Train the risk prediction model."""
        print("\n" + "="*60)
        print("STEP 4: MODEL TRAINING")
        print("="*60)
        
        # Prepare data
        target_col = 'next_week_risk_level'
        exclude_cols = ['week_start', 'next_week_risk_score', 'next_week_risk_level', 
                        'next_week_elevated_risk', 'current_risk_score']
        
        # Drop rows with NaN target
        train_df = self.features_df.dropna(subset=[target_col])
        feature_cols = [c for c in train_df.columns if c not in exclude_cols]
        
        X = train_df[feature_cols]
        y = train_df[target_col].astype(int)
        
        print(f"Training data: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"Class distribution:\n{y.value_counts().sort_index()}")
        
        # Compare models if requested
        if compare_models:
            print("\nComparing model types...")
            selector = ModelSelector()
            comparison = selector.compare_models(X, y, cv_splits=3)
            print(comparison)
            model_type = selector.get_best_model()
            print(f"Best model: {model_type}")
        
        # Train final model
        print(f"\nTraining {model_type} model...")
        self.model = HybridRiskPredictor(classifier_type=model_type)
        self.model.fit(X, y)
        
        # Show feature importance
        importance = self.model.get_feature_importance()
        print("\nTop 10 Feature Importances:")
        print(importance.head(10).to_string())
        
        # Save model
        model_path = self.models_dir / 'risk_predictor.joblib'
        self.model.save(str(model_path))
        print(f"\nModel saved to: {model_path}")
        
        return self
    
    def evaluate(self, test_split: float = 0.2) -> dict:
        """Evaluate the model with comprehensive metrics."""
        print("\n" + "="*60)
        print("STEP 5: EVALUATION")
        print("="*60)
        
        # Prepare data
        target_col = 'next_week_risk_level'
        exclude_cols = ['week_start', 'next_week_risk_score', 'next_week_risk_level', 
                        'next_week_elevated_risk', 'current_risk_score']
        
        eval_df = self.features_df.dropna(subset=[target_col])
        feature_cols = [c for c in eval_df.columns if c not in exclude_cols]
        
        X = eval_df[feature_cols]
        y = eval_df[target_col].astype(int)
        
        # Temporal train-test split
        split_idx = int(len(X) * (1 - test_split))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
        
        # Retrain model on training set for fair evaluation
        eval_model = HybridRiskPredictor(classifier_type=self.model.classifier.model_type)
        eval_model.fit(X_train, y_train)
        
        # Predict
        y_pred = eval_model.predict(X_test)
        y_proba = eval_model.predict_proba(X_test)
        
        # Create predictions dataframe for lead-time analysis
        predictions_df = eval_df.iloc[split_idx:][['week_start']].copy()
        predictions_df['actual_risk'] = y_test.values
        predictions_df['predicted_risk'] = y_pred
        
        # Generate comprehensive report
        self.reporter = ComprehensiveEvaluationReport(output_dir=str(self.output_dir))
        report = self.reporter.generate_report(
            y_test.values, y_pred, y_proba,
            predictions_df=predictions_df,
            feature_importance=eval_model.get_feature_importance()
        )
        
        self.reporter.print_full_report(report)
        
        # Save report
        report_path = self.output_dir / 'evaluation_report.json'
        # Convert numpy arrays to lists for JSON serialization
        json_report = self._prepare_report_for_json(report)
        with open(report_path, 'w') as f:
            json.dump(json_report, f, indent=2, default=str)
        print(f"\nReport saved to: {report_path}")
        
        return report
    
    def _prepare_report_for_json(self, report: dict) -> dict:
        """Convert report to JSON-serializable format."""
        import numpy as np
        
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(v) for v in obj]
            return obj
        
        return convert(report)
    
    def predict_next_week(self) -> dict:
        """Generate prediction for next week's risk level."""
        print("\n" + "="*60)
        print("NEXT WEEK RISK FORECAST")
        print("="*60)
        
        if self.model is None:
            raise ValueError("Model not trained. Run train_model() first.")
        
        # Use the most recent week's features
        target_col = 'next_week_risk_level'
        exclude_cols = ['week_start', 'next_week_risk_score', 'next_week_risk_level', 
                        'next_week_elevated_risk', 'current_risk_score']
        
        feature_cols = [c for c in self.features_df.columns if c not in exclude_cols]
        latest_features = self.features_df[feature_cols].iloc[[-1]]
        latest_week = self.features_df['week_start'].iloc[-1]
        
        # Get prediction with explanation
        explanation = self.model.get_prediction_explanation(latest_features)
        
        # Get anomaly indicators
        anomaly_cols = [c for c in self.features_df.columns if 'anomaly' in c.lower() or 'burst' in c.lower()]
        if anomaly_cols:
            latest_anomalies = self.features_df[anomaly_cols].iloc[-1]
            active_anomalies = latest_anomalies[latest_anomalies > 0].to_dict()
        else:
            active_anomalies = {}
        
        forecast = {
            'forecast_date': str(datetime.now().date()),
            'target_week': str(latest_week + pd.Timedelta(days=7)),
            'predicted_risk_level': explanation['predicted_risk_level'].iloc[0],
            'confidence': float(explanation['confidence'].iloc[0]),
            'anomaly_score': float(explanation['anomaly_score'].iloc[0]),
            'risk_probabilities': explanation['risk_probabilities'].iloc[0],
            'active_anomaly_signals': active_anomalies
        }
        
        print(f"\nForecast Date: {forecast['forecast_date']}")
        print(f"Target Week: {forecast['target_week']}")
        print(f"\nPredicted Risk Level: {forecast['predicted_risk_level']}")
        print(f"Confidence: {forecast['confidence']:.2%}")
        print(f"Anomaly Score: {forecast['anomaly_score']:.3f}")
        
        print("\nRisk Probabilities:")
        for level, prob in forecast['risk_probabilities'].items():
            bar = '█' * int(prob * 20)
            print(f"  {level:10s}: {bar} {prob:.2%}")
        
        if active_anomalies:
            print("\nActive Anomaly Signals:")
            for signal, value in active_anomalies.items():
                print(f"  - {signal}: {value:.3f}")
        
        # Save forecast
        forecast_path = self.output_dir / 'latest_forecast.json'
        with open(forecast_path, 'w') as f:
            json.dump(forecast, f, indent=2, default=str)
        print(f"\nForecast saved to: {forecast_path}")
        
        return forecast
    
    def run_full_pipeline(self, compare_models: bool = False) -> dict:
        """Run the complete pipeline."""
        print("\n" + "="*60)
        print("CYBERATTACK EARLY-WARNING SYSTEM")
        print("="*60)
        print(f"Started at: {datetime.now()}")
        
        # Execute all steps
        self.load_data()
        self.aggregate_weekly()
        self.engineer_features()
        self.train_model(compare_models=compare_models)
        report = self.evaluate()
        forecast = self.predict_next_week()
        
        print("\n" + "="*60)
        print("PIPELINE COMPLETE")
        print("="*60)
        print(f"Finished at: {datetime.now()}")
        print(f"\nOutputs saved to: {self.output_dir}")
        print(f"Model saved to: {self.models_dir}")
        
        return {
            'evaluation_report': report,
            'forecast': forecast
        }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Cyberattack Early-Warning System: Forecast next-week risk levels'
    )
    parser.add_argument(
        '--data-dir', type=str, default=None,
        help='Path to data directory'
    )
    parser.add_argument(
        '--output-dir', type=str, default=None,
        help='Path to output directory'
    )
    parser.add_argument(
        '--compare-models', action='store_true',
        help='Compare different model types before training'
    )
    parser.add_argument(
        '--predict-only', action='store_true',
        help='Only generate prediction (requires trained model)'
    )
    
    args = parser.parse_args()
    
    # Initialize pipeline
    data_dir = Path(args.data_dir) if args.data_dir else None
    output_dir = Path(args.output_dir) if args.output_dir else None
    
    pipeline = CyberAttackEarlyWarningPipeline(data_dir=data_dir, output_dir=output_dir)
    
    if args.predict_only:
        # Load existing model and make prediction
        model_path = pipeline.models_dir / 'risk_predictor.joblib'
        if not model_path.exists():
            print("Error: No trained model found. Run full pipeline first.")
            return
        
        pipeline.model = HybridRiskPredictor.load(str(model_path))
        pipeline.load_data()
        pipeline.aggregate_weekly()
        pipeline.engineer_features()
        pipeline.predict_next_week()
    else:
        # Run full pipeline
        pipeline.run_full_pipeline(compare_models=args.compare_models)


# Import pandas for predict_next_week
import pandas as pd

if __name__ == "__main__":
    main()
