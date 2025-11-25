"""
Evaluation Framework for Cyberattack Early-Warning System
Implements precision-recall analysis and lead-time benefit metrics.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score,
    precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report,
    roc_auc_score, roc_curve
)
from sklearn.calibration import calibration_curve
import warnings
warnings.filterwarnings('ignore')


class RiskPredictionEvaluator:
    """Comprehensive evaluation for risk prediction models."""
    
    def __init__(self, risk_labels: List[str] = None):
        self.risk_labels = risk_labels or ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']
        self.results = {}
        
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray, 
                y_proba: np.ndarray = None) -> Dict:
        """
        Evaluate predictions with multiple metrics.
        
        Args:
            y_true: True risk levels (0-3)
            y_pred: Predicted risk levels
            y_proba: Prediction probabilities (optional)
        """
        results = {}
        
        # Basic metrics
        results['accuracy'] = accuracy_score(y_true, y_pred)
        results['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
        results['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
        results['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
        
        # Weighted metrics (account for class imbalance)
        results['precision_weighted'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        results['recall_weighted'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        results['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # Per-class metrics
        for i, label in enumerate(self.risk_labels):
            y_true_binary = (y_true == i).astype(int)
            y_pred_binary = (y_pred == i).astype(int)
            
            results[f'precision_{label}'] = precision_score(y_true_binary, y_pred_binary, zero_division=0)
            results[f'recall_{label}'] = recall_score(y_true_binary, y_pred_binary, zero_division=0)
            results[f'f1_{label}'] = f1_score(y_true_binary, y_pred_binary, zero_division=0)
        
        # Elevated risk detection (HIGH or CRITICAL)
        y_true_elevated = (y_true >= 2).astype(int)
        y_pred_elevated = (y_pred >= 2).astype(int)
        
        results['elevated_precision'] = precision_score(y_true_elevated, y_pred_elevated, zero_division=0)
        results['elevated_recall'] = recall_score(y_true_elevated, y_pred_elevated, zero_division=0)
        results['elevated_f1'] = f1_score(y_true_elevated, y_pred_elevated, zero_division=0)
        
        # Confusion matrix
        results['confusion_matrix'] = confusion_matrix(y_true, y_pred)
        
        # Classification report
        results['classification_report'] = classification_report(
            y_true, y_pred, target_names=self.risk_labels, zero_division=0
        )
        
        # Probability-based metrics if available
        if y_proba is not None:
            results.update(self._evaluate_probabilities(y_true, y_proba))
        
        self.results = results
        return results
    
    def _evaluate_probabilities(self, y_true: np.ndarray, y_proba: np.ndarray) -> Dict:
        """Evaluate probability calibration and AUC metrics."""
        results = {}
        
        # One-vs-rest AUC for each class
        for i, label in enumerate(self.risk_labels):
            if i < y_proba.shape[1]:
                y_true_binary = (y_true == i).astype(int)
                try:
                    results[f'auc_{label}'] = roc_auc_score(y_true_binary, y_proba[:, i])
                except:
                    results[f'auc_{label}'] = 0.0
        
        # Elevated risk AUC
        y_true_elevated = (y_true >= 2).astype(int)
        if y_proba.shape[1] >= 4:
            elevated_proba = y_proba[:, 2:].sum(axis=1)  # Sum of HIGH and CRITICAL probs
        else:
            elevated_proba = y_proba[:, -1]
        
        try:
            results['elevated_auc'] = roc_auc_score(y_true_elevated, elevated_proba)
        except:
            results['elevated_auc'] = 0.0
        
        return results
    
    def get_precision_recall_analysis(self, y_true: np.ndarray, 
                                      y_proba: np.ndarray) -> Dict:
        """
        Detailed precision-recall analysis for elevated risk detection.
        """
        # Focus on elevated risk (HIGH or CRITICAL)
        y_true_elevated = (y_true >= 2).astype(int)
        
        if y_proba.shape[1] >= 4:
            elevated_proba = y_proba[:, 2:].sum(axis=1)
        else:
            elevated_proba = y_proba[:, -1]
        
        # Precision-recall curve
        precision, recall, thresholds = precision_recall_curve(y_true_elevated, elevated_proba)
        
        # Average precision
        avg_precision = average_precision_score(y_true_elevated, elevated_proba)
        
        # Find optimal threshold (F1 maximizing)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
        
        # Precision at different recall levels
        recall_levels = [0.5, 0.7, 0.8, 0.9, 0.95]
        precision_at_recall = {}
        for target_recall in recall_levels:
            idx = np.argmin(np.abs(recall - target_recall))
            precision_at_recall[f'precision_at_recall_{int(target_recall*100)}'] = precision[idx]
        
        return {
            'precision': precision,
            'recall': recall,
            'thresholds': thresholds,
            'average_precision': avg_precision,
            'optimal_threshold': optimal_threshold,
            'optimal_f1': f1_scores[optimal_idx],
            **precision_at_recall
        }
    
    def print_summary(self):
        """Print evaluation summary."""
        if not self.results:
            print("No results to display. Run evaluate() first.")
            return
        
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        
        print("\nOverall Metrics:")
        print(f"  Accuracy:          {self.results['accuracy']:.4f}")
        print(f"  Precision (macro): {self.results['precision_macro']:.4f}")
        print(f"  Recall (macro):    {self.results['recall_macro']:.4f}")
        print(f"  F1 (macro):        {self.results['f1_macro']:.4f}")
        
        print("\nElevated Risk Detection (HIGH/CRITICAL):")
        print(f"  Precision:         {self.results['elevated_precision']:.4f}")
        print(f"  Recall:            {self.results['elevated_recall']:.4f}")
        print(f"  F1:                {self.results['elevated_f1']:.4f}")
        if 'elevated_auc' in self.results:
            print(f"  AUC:               {self.results['elevated_auc']:.4f}")
        
        print("\nPer-Class Metrics:")
        for label in self.risk_labels:
            print(f"  {label}:")
            print(f"    Precision: {self.results.get(f'precision_{label}', 0):.4f}")
            print(f"    Recall:    {self.results.get(f'recall_{label}', 0):.4f}")
            print(f"    F1:        {self.results.get(f'f1_{label}', 0):.4f}")
        
        print("\nConfusion Matrix:")
        print(self.results['confusion_matrix'])
        
        print("\n" + "="*60)


class LeadTimeAnalyzer:
    """Analyze lead-time benefit of early warning predictions."""
    
    def __init__(self, forecast_horizon_days: int = 7):
        self.forecast_horizon_days = forecast_horizon_days
        
    def analyze_lead_time(self, predictions_df: pd.DataFrame,
                         date_col: str = 'week_start',
                         pred_col: str = 'predicted_risk',
                         true_col: str = 'actual_risk') -> Dict:
        """
        Analyze lead-time benefit of predictions.
        
        Measures how many days in advance elevated risks were predicted.
        """
        df = predictions_df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(date_col).reset_index(drop=True)
        
        results = {
            'lead_time_days': [],
            'detection_type': [],
            'true_positives': 0,
            'false_positives': 0,
            'false_negatives': 0,
            'early_warnings': 0
        }
        
        # Identify actual elevated risk periods
        actual_elevated = df[df[true_col] >= 2].copy()
        predicted_elevated = df[df[pred_col] >= 2].copy()
        
        for idx, row in actual_elevated.iterrows():
            event_date = row[date_col]
            
            # Check if this event was predicted
            was_predicted = any(predicted_elevated[date_col] == event_date)
            
            if was_predicted:
                results['true_positives'] += 1
                results['lead_time_days'].append(self.forecast_horizon_days)
                results['detection_type'].append('on_time')
            else:
                # Check for early warning in previous week
                prev_week = event_date - timedelta(days=7)
                early_warning = any(
                    (predicted_elevated[date_col] >= prev_week) & 
                    (predicted_elevated[date_col] < event_date)
                )
                
                if early_warning:
                    results['early_warnings'] += 1
                    results['lead_time_days'].append(self.forecast_horizon_days + 7)
                    results['detection_type'].append('early')
                else:
                    results['false_negatives'] += 1
        
        # Count false positives
        for idx, row in predicted_elevated.iterrows():
            pred_date = row[date_col]
            # Check next week's actual risk
            next_week = pred_date + timedelta(days=7)
            future_elevated = any(
                (actual_elevated[date_col] >= pred_date) & 
                (actual_elevated[date_col] <= next_week + timedelta(days=7))
            )
            if not future_elevated:
                results['false_positives'] += 1
        
        # Compute summary statistics
        if results['lead_time_days']:
            results['avg_lead_time'] = np.mean(results['lead_time_days'])
            results['max_lead_time'] = max(results['lead_time_days'])
        else:
            results['avg_lead_time'] = 0
            results['max_lead_time'] = 0
        
        # Detection rate
        total_events = len(actual_elevated)
        if total_events > 0:
            results['detection_rate'] = (results['true_positives'] + results['early_warnings']) / total_events
        else:
            results['detection_rate'] = 0
        
        return results
    
    def compute_response_time_benefit(self, lead_time_results: Dict,
                                      baseline_response_hours: float = 72,
                                      improved_response_hours: float = 24) -> Dict:
        """
        Compute potential benefit in terms of response time improvement.
        
        Args:
            lead_time_results: Results from analyze_lead_time()
            baseline_response_hours: Typical response time without early warning
            improved_response_hours: Response time with early warning
        """
        total_warnings = lead_time_results['true_positives'] + lead_time_results['early_warnings']
        
        # Time saved per correct warning
        time_saved_per_warning = baseline_response_hours - improved_response_hours
        
        # Total potential time saved
        total_time_saved_hours = total_warnings * time_saved_per_warning
        
        # Average lead time in hours
        avg_lead_time_hours = lead_time_results['avg_lead_time'] * 24
        
        return {
            'total_correct_warnings': total_warnings,
            'time_saved_per_warning_hours': time_saved_per_warning,
            'total_time_saved_hours': total_time_saved_hours,
            'avg_lead_time_hours': avg_lead_time_hours,
            'detection_rate': lead_time_results['detection_rate'],
            'false_alarm_rate': lead_time_results['false_positives'] / max(1, total_warnings + lead_time_results['false_positives'])
        }
    
    def print_lead_time_report(self, lead_time_results: Dict, benefit_results: Dict = None):
        """Print lead-time analysis report."""
        print("\n" + "="*60)
        print("LEAD-TIME BENEFIT ANALYSIS")
        print("="*60)
        
        print("\nDetection Summary:")
        print(f"  True Positives (on-time): {lead_time_results['true_positives']}")
        print(f"  Early Warnings:           {lead_time_results['early_warnings']}")
        print(f"  False Negatives (missed): {lead_time_results['false_negatives']}")
        print(f"  False Positives:          {lead_time_results['false_positives']}")
        
        print(f"\nLead-Time Metrics:")
        print(f"  Average Lead Time: {lead_time_results['avg_lead_time']:.1f} days")
        print(f"  Maximum Lead Time: {lead_time_results['max_lead_time']:.1f} days")
        print(f"  Detection Rate:    {lead_time_results['detection_rate']:.2%}")
        
        if benefit_results:
            print(f"\nPotential Response Time Benefits:")
            print(f"  Total Correct Warnings:   {benefit_results['total_correct_warnings']}")
            print(f"  Time Saved per Warning:   {benefit_results['time_saved_per_warning_hours']:.1f} hours")
            print(f"  Total Time Saved:         {benefit_results['total_time_saved_hours']:.1f} hours")
            print(f"  False Alarm Rate:         {benefit_results['false_alarm_rate']:.2%}")
        
        print("\n" + "="*60)


class EvaluationVisualizer:
    """Generate visualizations for model evaluation."""
    
    def __init__(self, output_dir: str = None):
        self.output_dir = output_dir
        
    def plot_confusion_matrix(self, cm: np.ndarray, 
                             labels: List[str] = None,
                             title: str = 'Confusion Matrix',
                             save_path: str = None):
        """Plot confusion matrix heatmap."""
        labels = labels or ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=labels, yticklabels=labels)
        plt.title(title)
        plt.ylabel('Actual Risk Level')
        plt.xlabel('Predicted Risk Level')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return plt.gcf()
    
    def plot_precision_recall_curve(self, precision: np.ndarray,
                                   recall: np.ndarray,
                                   avg_precision: float,
                                   title: str = 'Precision-Recall Curve (Elevated Risk)',
                                   save_path: str = None):
        """Plot precision-recall curve."""
        plt.figure(figsize=(10, 6))
        plt.plot(recall, precision, 'b-', linewidth=2,
                label=f'AP = {avg_precision:.3f}')
        plt.fill_between(recall, precision, alpha=0.2)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(title)
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return plt.gcf()
    
    def plot_feature_importance(self, importance_df: pd.DataFrame,
                               top_n: int = 20,
                               title: str = 'Top Feature Importances',
                               save_path: str = None):
        """Plot feature importance bar chart."""
        top_features = importance_df.head(top_n)
        
        plt.figure(figsize=(12, 8))
        bars = plt.barh(range(len(top_features)), top_features['importance'].values)
        plt.yticks(range(len(top_features)), top_features['feature'].values)
        plt.xlabel('Importance')
        plt.title(title)
        plt.gca().invert_yaxis()
        
        # Color bars by importance
        colors = plt.cm.RdYlGn(top_features['importance'].values / top_features['importance'].max())
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return plt.gcf()
    
    def plot_risk_timeline(self, df: pd.DataFrame,
                          date_col: str = 'week_start',
                          actual_col: str = 'actual_risk',
                          predicted_col: str = 'predicted_risk',
                          title: str = 'Risk Level Timeline',
                          save_path: str = None):
        """Plot timeline of actual vs predicted risk levels."""
        fig, ax = plt.subplots(figsize=(14, 6))
        
        dates = pd.to_datetime(df[date_col])
        
        # Plot actual and predicted
        ax.plot(dates, df[actual_col], 'b-', label='Actual Risk', linewidth=2, marker='o', markersize=4)
        ax.plot(dates, df[predicted_col], 'r--', label='Predicted Risk', linewidth=2, marker='x', markersize=4)
        
        # Add threshold line for elevated risk
        ax.axhline(y=2, color='orange', linestyle=':', linewidth=1.5, label='Elevated Risk Threshold')
        
        # Shade elevated risk regions
        elevated_mask = df[actual_col] >= 2
        for i in range(len(df)):
            if elevated_mask.iloc[i]:
                ax.axvspan(dates.iloc[i] - timedelta(days=3.5), 
                          dates.iloc[i] + timedelta(days=3.5),
                          alpha=0.2, color='red')
        
        ax.set_xlabel('Date')
        ax.set_ylabel('Risk Level')
        ax.set_title(title)
        ax.legend(loc='best')
        ax.set_yticks([0, 1, 2, 3])
        ax.set_yticklabels(['LOW', 'MEDIUM', 'HIGH', 'CRITICAL'])
        ax.grid(True, alpha=0.3)
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return fig


class ComprehensiveEvaluationReport:
    """Generate comprehensive evaluation report."""
    
    def __init__(self, output_dir: str = None):
        self.output_dir = output_dir
        self.evaluator = RiskPredictionEvaluator()
        self.lead_time_analyzer = LeadTimeAnalyzer()
        self.visualizer = EvaluationVisualizer(output_dir)
        
    def generate_report(self, y_true: np.ndarray, y_pred: np.ndarray,
                       y_proba: np.ndarray = None,
                       predictions_df: pd.DataFrame = None,
                       feature_importance: pd.DataFrame = None) -> Dict:
        """
        Generate comprehensive evaluation report.
        
        Args:
            y_true: True risk levels
            y_pred: Predicted risk levels  
            y_proba: Prediction probabilities
            predictions_df: DataFrame with dates for lead-time analysis
            feature_importance: Feature importance DataFrame
        """
        report = {}
        
        # 1. Basic evaluation metrics
        print("Computing evaluation metrics...")
        report['metrics'] = self.evaluator.evaluate(y_true, y_pred, y_proba)
        
        # 2. Precision-recall analysis
        if y_proba is not None:
            print("Computing precision-recall analysis...")
            report['pr_analysis'] = self.evaluator.get_precision_recall_analysis(y_true, y_proba)
        
        # 3. Lead-time analysis
        if predictions_df is not None:
            print("Computing lead-time benefit analysis...")
            lead_time_results = self.lead_time_analyzer.analyze_lead_time(predictions_df)
            report['lead_time'] = lead_time_results
            report['response_benefit'] = self.lead_time_analyzer.compute_response_time_benefit(lead_time_results)
        
        # 4. Generate visualizations
        if self.output_dir:
            print("Generating visualizations...")
            self._save_visualizations(report, feature_importance)
        
        return report
    
    def _save_visualizations(self, report: Dict, feature_importance: pd.DataFrame = None):
        """Save all visualizations."""
        import os
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Confusion matrix
        if 'confusion_matrix' in report.get('metrics', {}):
            self.visualizer.plot_confusion_matrix(
                report['metrics']['confusion_matrix'],
                save_path=f"{self.output_dir}/confusion_matrix.png"
            )
        
        # Precision-recall curve
        if 'pr_analysis' in report:
            self.visualizer.plot_precision_recall_curve(
                report['pr_analysis']['precision'],
                report['pr_analysis']['recall'],
                report['pr_analysis']['average_precision'],
                save_path=f"{self.output_dir}/precision_recall_curve.png"
            )
        
        # Feature importance
        if feature_importance is not None:
            self.visualizer.plot_feature_importance(
                feature_importance,
                save_path=f"{self.output_dir}/feature_importance.png"
            )
    
    def print_full_report(self, report: Dict):
        """Print complete evaluation report."""
        self.evaluator.results = report.get('metrics', {})
        self.evaluator.print_summary()
        
        if 'lead_time' in report:
            self.lead_time_analyzer.print_lead_time_report(
                report['lead_time'],
                report.get('response_benefit')
            )
        
        if 'pr_analysis' in report:
            print("\n" + "="*60)
            print("PRECISION-RECALL ANALYSIS")
            print("="*60)
            pr = report['pr_analysis']
            print(f"  Average Precision:    {pr['average_precision']:.4f}")
            print(f"  Optimal Threshold:    {pr['optimal_threshold']:.4f}")
            print(f"  Optimal F1:           {pr['optimal_f1']:.4f}")
            for key, value in pr.items():
                if key.startswith('precision_at_recall'):
                    recall_pct = key.split('_')[-1]
                    print(f"  Precision @ {recall_pct}% Recall: {value:.4f}")


if __name__ == "__main__":
    # Test evaluation framework
    import sys
    sys.path.append('..')
    from data_loaders import DataAggregator
    from feature_engineering import FeatureEngineeringPipeline
    from models import HybridRiskPredictor
    from config import DATA_DIR, OUTPUTS_DIR
    
    # Load and prepare data
    print("Loading and preparing data...")
    aggregator = DataAggregator(DATA_DIR)
    raw_data = aggregator.load_all()
    weekly_df = aggregator.create_weekly_aggregation(raw_data)
    
    pipeline = FeatureEngineeringPipeline()
    features_df = pipeline.fit_transform(weekly_df, raw_data)
    
    # Prepare X and y
    target_col = 'next_week_risk_level'
    exclude_cols = ['week_start', 'next_week_risk_score', 'next_week_risk_level', 
                    'next_week_elevated_risk', 'current_risk_score']
    
    features_df = features_df.dropna(subset=[target_col])
    feature_cols = [c for c in features_df.columns if c not in exclude_cols]
    
    X = features_df[feature_cols]
    y = features_df[target_col].astype(int)
    
    # Train-test split (temporal)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    # Train model
    print("\nTraining model...")
    model = HybridRiskPredictor(classifier_type='ensemble')
    model.fit(X_train, y_train)
    
    # Predict
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    
    # Create predictions dataframe for lead-time analysis
    predictions_df = features_df.iloc[split_idx:][['week_start']].copy()
    predictions_df['actual_risk'] = y_test.values
    predictions_df['predicted_risk'] = y_pred
    
    # Generate comprehensive report
    print("\nGenerating evaluation report...")
    reporter = ComprehensiveEvaluationReport(output_dir=str(OUTPUTS_DIR))
    report = reporter.generate_report(
        y_test.values, y_pred, y_proba,
        predictions_df=predictions_df,
        feature_importance=model.get_feature_importance()
    )
    
    reporter.print_full_report(report)
