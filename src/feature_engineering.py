"""
Feature Engineering for Cyberattack Early-Warning System
Implements lag features, anomaly detection on signal bursts, and text embeddings.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from collections import defaultdict


class LagFeatureGenerator:
    """Generate lag features from time series data."""
    
    def __init__(self, lag_weeks: List[int] = [1, 2, 3, 4]):
        self.lag_weeks = lag_weeks
        
    def generate_lag_features(self, df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
        """Generate lagged versions of specified features."""
        df = df.copy()
        df = df.sort_values('week_start').reset_index(drop=True)
        
        new_features = {}
        
        for col in feature_cols:
            if col not in df.columns or col == 'week_start':
                continue
                
            for lag in self.lag_weeks:
                # Lagged value
                new_features[f"{col}_lag{lag}"] = df[col].shift(lag)
                
            # Rolling statistics
            new_features[f"{col}_rolling_mean_4w"] = df[col].rolling(window=4, min_periods=1).mean()
            new_features[f"{col}_rolling_std_4w"] = df[col].rolling(window=4, min_periods=1).std()
            new_features[f"{col}_rolling_max_4w"] = df[col].rolling(window=4, min_periods=1).max()
            
            # Week-over-week change
            new_features[f"{col}_wow_change"] = df[col].diff()
            new_features[f"{col}_wow_pct_change"] = df[col].pct_change().replace([np.inf, -np.inf], np.nan)
            
            # Exponential weighted moving average
            new_features[f"{col}_ewma"] = df[col].ewm(span=4, adjust=False).mean()
        
        # Add all new features to dataframe
        for feat_name, feat_values in new_features.items():
            df[feat_name] = feat_values
            
        return df
    
    def generate_cross_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate cross-source interaction features."""
        df = df.copy()
        
        # Attack intensity vs exposure (honeypot attacks * vulnerable services)
        if 'honeypot_total_attacks' in df.columns and 'shodan_vulnerable_services' in df.columns:
            df['attack_exposure_interaction'] = df['honeypot_total_attacks'] * df['shodan_vulnerable_services']
        
        # CVE urgency score (critical CVEs without patches)
        if 'cve_critical_count' in df.columns and 'cve_no_patch' in df.columns:
            df['cve_urgency_score'] = df['cve_critical_count'] * (df['cve_no_patch'] + 1)
        
        # PoC threat level (weaponized PoCs with high stars)
        if 'poc_weaponized_count' in df.columns and 'poc_total_stars' in df.columns:
            df['poc_threat_level'] = df['poc_weaponized_count'] * np.log1p(df['poc_total_stars'])
        
        # Intel-weighted threat score
        if 'intel_critical_count' in df.columns and 'intel_avg_confidence' in df.columns:
            df['intel_threat_score'] = df['intel_critical_count'] * df['intel_avg_confidence']
        
        # Sector-specific risk scores
        sectors = ['education', 'government', 'healthcare', 'finance', 'critical_infrastructure']
        for sector in sectors:
            attack_col = f'honeypot_{sector}_attacks'
            exposed_col = f'shodan_{sector}_exposed'
            intel_col = f'intel_{sector}_reports'
            
            risk_components = []
            if attack_col in df.columns:
                risk_components.append(df[attack_col])
            if exposed_col in df.columns:
                risk_components.append(df[exposed_col])
            if intel_col in df.columns:
                risk_components.append(df[intel_col] * 10)  # Weight intel higher
                
            if risk_components:
                df[f'{sector}_composite_risk'] = sum(risk_components)
        
        return df


class AnomalyDetector:
    """Detect anomalous signal bursts in OSINT data."""
    
    def __init__(self, lookback_days: int = 30, zscore_threshold: float = 2.5):
        self.lookback_days = lookback_days
        self.zscore_threshold = zscore_threshold
        
    def detect_signal_bursts(self, df: pd.DataFrame, signal_cols: List[str]) -> pd.DataFrame:
        """Detect anomalous bursts in signal columns using multiple methods."""
        df = df.copy()
        
        for col in signal_cols:
            if col not in df.columns:
                continue
            
            # Z-score based anomaly
            df[f'{col}_zscore'] = self._calculate_zscore(df[col])
            df[f'{col}_is_burst'] = (df[f'{col}_zscore'].abs() > self.zscore_threshold).astype(int)
            
            # IQR-based anomaly
            df[f'{col}_iqr_anomaly'] = self._iqr_anomaly(df[col])
            
            # Rate of change anomaly
            df[f'{col}_roc_anomaly'] = self._rate_of_change_anomaly(df[col])
            
            # Combined anomaly score
            df[f'{col}_anomaly_score'] = (
                df[f'{col}_is_burst'] + 
                df[f'{col}_iqr_anomaly'] + 
                df[f'{col}_roc_anomaly']
            ) / 3
        
        # Aggregate anomaly indicators
        burst_cols = [c for c in df.columns if c.endswith('_is_burst')]
        if burst_cols:
            df['total_burst_signals'] = df[burst_cols].sum(axis=1)
            df['any_burst_detected'] = (df['total_burst_signals'] > 0).astype(int)
        
        anomaly_score_cols = [c for c in df.columns if c.endswith('_anomaly_score')]
        if anomaly_score_cols:
            df['composite_anomaly_score'] = df[anomaly_score_cols].mean(axis=1)
        
        return df
    
    def _calculate_zscore(self, series: pd.Series) -> pd.Series:
        """Calculate rolling z-score."""
        rolling_mean = series.rolling(window=4, min_periods=1).mean()
        rolling_std = series.rolling(window=4, min_periods=1).std().replace(0, 1)
        return (series - rolling_mean) / rolling_std
    
    def _iqr_anomaly(self, series: pd.Series) -> pd.Series:
        """Detect anomalies using IQR method."""
        Q1 = series.rolling(window=8, min_periods=1).quantile(0.25)
        Q3 = series.rolling(window=8, min_periods=1).quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return ((series < lower_bound) | (series > upper_bound)).astype(int)
    
    def _rate_of_change_anomaly(self, series: pd.Series, threshold: float = 2.0) -> pd.Series:
        """Detect anomalies based on sudden rate of change."""
        pct_change = series.pct_change().abs().replace([np.inf, -np.inf], np.nan).fillna(0)
        mean_change = pct_change.rolling(window=4, min_periods=1).mean()
        std_change = pct_change.rolling(window=4, min_periods=1).std().replace(0, 1)
        roc_zscore = (pct_change - mean_change) / std_change
        return (roc_zscore.abs() > threshold).astype(int)


class TextEmbeddingGenerator:
    """Generate text embeddings for PoC descriptions and threat intel summaries."""
    
    def __init__(self, use_pretrained: bool = True):
        self.use_pretrained = use_pretrained
        self.embedder = None
        self._init_embedder()
        
    def _init_embedder(self):
        """Initialize the sentence transformer model."""
        try:
            from sentence_transformers import SentenceTransformer
            self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
            print("Loaded sentence-transformers model")
        except ImportError:
            print("sentence-transformers not available, using TF-IDF fallback")
            self.embedder = None
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of texts."""
        if self.embedder is not None:
            return self.embedder.encode(texts, show_progress_bar=False)
        else:
            return self._tfidf_fallback(texts)
    
    def _tfidf_fallback(self, texts: List[str]) -> np.ndarray:
        """Fallback to TF-IDF if sentence transformers not available."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD
        
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(texts)
        
        # Reduce to 384 dimensions to match sentence transformer output
        n_components = min(384, tfidf_matrix.shape[1] - 1, tfidf_matrix.shape[0] - 1)
        if n_components > 0:
            svd = TruncatedSVD(n_components=n_components)
            embeddings = svd.fit_transform(tfidf_matrix)
            # Pad to 384 dimensions if needed
            if embeddings.shape[1] < 384:
                padding = np.zeros((embeddings.shape[0], 384 - embeddings.shape[1]))
                embeddings = np.hstack([embeddings, padding])
            return embeddings
        return np.zeros((len(texts), 384))
    
    def compute_similarity_features(self, poc_texts: List[str], intel_texts: List[str]) -> Dict[str, float]:
        """Compute similarity features between PoC descriptions and threat intel."""
        if not poc_texts or not intel_texts:
            return {
                'poc_intel_max_similarity': 0.0,
                'poc_intel_avg_similarity': 0.0,
                'poc_cluster_density': 0.0
            }
        
        poc_embeddings = self.generate_embeddings(poc_texts)
        intel_embeddings = self.generate_embeddings(intel_texts)
        
        # Compute cosine similarities
        from sklearn.metrics.pairwise import cosine_similarity
        
        similarities = cosine_similarity(poc_embeddings, intel_embeddings)
        
        features = {
            'poc_intel_max_similarity': float(similarities.max()),
            'poc_intel_avg_similarity': float(similarities.mean()),
            'poc_intel_std_similarity': float(similarities.std()),
        }
        
        # PoC clustering density (how similar PoCs are to each other)
        if len(poc_texts) > 1:
            poc_self_sim = cosine_similarity(poc_embeddings)
            np.fill_diagonal(poc_self_sim, 0)
            features['poc_cluster_density'] = float(poc_self_sim.mean())
        else:
            features['poc_cluster_density'] = 0.0
        
        return features


class TargetLabelGenerator:
    """Generate target labels for next-week risk prediction."""
    
    def __init__(self, risk_thresholds: Dict[str, Tuple[float, float, float]] = None):
        # Thresholds for LOW, MEDIUM, HIGH, CRITICAL
        self.risk_thresholds = risk_thresholds or {
            'default': (25, 50, 75)  # Percentile thresholds
        }
    
    def generate_risk_labels(self, df: pd.DataFrame, 
                            risk_indicators: List[str] = None) -> pd.DataFrame:
        """Generate next-week risk level labels."""
        df = df.copy()
        
        # Default risk indicators
        if risk_indicators is None:
            risk_indicators = [
                'honeypot_total_attacks',
                'honeypot_success_rate',
                'cve_critical_count',
                'poc_weaponized_count',
                'intel_critical_count'
            ]
        
        # Filter to available columns
        risk_indicators = [c for c in risk_indicators if c in df.columns]
        
        # Compute composite risk score for current week
        df['current_risk_score'] = self._compute_risk_score(df, risk_indicators)
        
        # Target: next week's risk level (shifted by -1)
        df['next_week_risk_score'] = df['current_risk_score'].shift(-1)
        
        # Categorize into risk levels using percentiles
        thresholds = np.percentile(
            df['next_week_risk_score'].dropna(), 
            self.risk_thresholds['default']
        )
        
        df['next_week_risk_level'] = pd.cut(
            df['next_week_risk_score'],
            bins=[-np.inf, thresholds[0], thresholds[1], thresholds[2], np.inf],
            labels=[0, 1, 2, 3]  # LOW, MEDIUM, HIGH, CRITICAL
        ).astype(float)
        
        # Binary classification: elevated risk (HIGH or CRITICAL)
        df['next_week_elevated_risk'] = (df['next_week_risk_level'] >= 2).astype(float)
        
        return df
    
    def _compute_risk_score(self, df: pd.DataFrame, indicators: List[str]) -> pd.Series:
        """Compute composite risk score from multiple indicators."""
        # Normalize each indicator
        scaler = MinMaxScaler()
        
        normalized = pd.DataFrame()
        for col in indicators:
            values = df[col].values.reshape(-1, 1)
            normalized[col] = scaler.fit_transform(values).flatten()
        
        # Weighted average (can be customized)
        weights = {
            'honeypot_total_attacks': 0.20,
            'honeypot_success_rate': 0.25,  # High weight - actual compromise indicator
            'cve_critical_count': 0.15,
            'poc_weaponized_count': 0.25,  # High weight - active exploitation potential
            'intel_critical_count': 0.15
        }
        
        # Apply weights
        weighted_sum = pd.Series(0.0, index=df.index)
        total_weight = 0
        
        for col in indicators:
            weight = weights.get(col, 0.1)
            weighted_sum += normalized[col] * weight
            total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else weighted_sum


class FeatureEngineeringPipeline:
    """Complete feature engineering pipeline."""
    
    def __init__(self, lag_weeks: List[int] = [1, 2, 3, 4]):
        self.lag_generator = LagFeatureGenerator(lag_weeks)
        self.anomaly_detector = AnomalyDetector()
        self.embedding_generator = TextEmbeddingGenerator()
        self.label_generator = TargetLabelGenerator()
        self.scaler = StandardScaler()
        
    def fit_transform(self, weekly_df: pd.DataFrame, 
                      raw_data: Dict[str, pd.DataFrame] = None) -> pd.DataFrame:
        """Apply full feature engineering pipeline."""
        df = weekly_df.copy()
        
        # Get feature columns (exclude week_start)
        feature_cols = [c for c in df.columns if c != 'week_start']
        
        # 1. Generate lag features
        print("Generating lag features...")
        df = self.lag_generator.generate_lag_features(df, feature_cols)
        
        # 2. Generate cross-source interaction features
        print("Generating cross features...")
        df = self.lag_generator.generate_cross_features(df)
        
        # 3. Detect anomalous signal bursts
        print("Detecting signal bursts...")
        signal_cols = [
            'honeypot_total_attacks', 'cve_critical_count', 
            'poc_new_count', 'intel_critical_count',
            'shodan_vulnerable_services'
        ]
        signal_cols = [c for c in signal_cols if c in df.columns]
        df = self.anomaly_detector.detect_signal_bursts(df, signal_cols)
        
        # 4. Add text embedding features if raw data available
        if raw_data is not None:
            print("Generating text embedding features...")
            df = self._add_embedding_features(df, raw_data)
        
        # 5. Generate target labels
        print("Generating target labels...")
        df = self.label_generator.generate_risk_labels(df)
        
        # 6. Handle missing values
        print("Handling missing values...")
        df = self._handle_missing_values(df)
        
        return df
    
    def _add_embedding_features(self, df: pd.DataFrame, 
                                raw_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Add weekly text embedding similarity features."""
        df = df.copy()
        
        poc_df = raw_data.get('github_poc', pd.DataFrame())
        intel_df = raw_data.get('threat_intel', pd.DataFrame())
        
        embedding_features = []
        
        for idx, row in df.iterrows():
            week_start = row['week_start']
            week_end = week_start + timedelta(days=7)
            
            # Get texts for this week
            poc_texts = []
            intel_texts = []
            
            if 'discovered_date' in poc_df.columns and 'description' in poc_df.columns:
                mask = (pd.to_datetime(poc_df['discovered_date']) >= week_start) & \
                       (pd.to_datetime(poc_df['discovered_date']) < week_end)
                poc_texts = poc_df.loc[mask, 'description'].tolist()
            
            if 'published_date' in intel_df.columns and 'summary' in intel_df.columns:
                mask = (pd.to_datetime(intel_df['published_date']) >= week_start) & \
                       (pd.to_datetime(intel_df['published_date']) < week_end)
                intel_texts = intel_df.loc[mask, 'summary'].tolist()
            
            # Compute similarity features
            features = self.embedding_generator.compute_similarity_features(poc_texts, intel_texts)
            embedding_features.append(features)
        
        # Add to dataframe
        embedding_df = pd.DataFrame(embedding_features, index=df.index)
        df = pd.concat([df, embedding_df], axis=1)
        
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the feature set."""
        df = df.copy()
        
        # Fill NaN with 0 for most features
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(0)
        
        return df
    
    def get_feature_importance_groups(self) -> Dict[str, List[str]]:
        """Return feature groupings for interpretability."""
        return {
            'attack_volume': [
                'honeypot_total_attacks', 'honeypot_unique_ips', 
                'honeypot_total_attacks_lag1', 'honeypot_total_attacks_rolling_mean_4w'
            ],
            'vulnerability_exposure': [
                'cve_critical_count', 'cve_no_patch', 'cve_cisa_kev',
                'shodan_vulnerable_services', 'shodan_no_auth_count'
            ],
            'exploit_emergence': [
                'poc_new_count', 'poc_weaponized_count', 'poc_threat_level',
                'poc_intel_max_similarity'
            ],
            'threat_intelligence': [
                'intel_critical_count', 'intel_threat_score', 
                'intel_apt28_activity', 'intel_lazarus_activity'
            ],
            'anomaly_signals': [
                'total_burst_signals', 'composite_anomaly_score',
                'honeypot_total_attacks_is_burst'
            ],
            'sector_risk': [
                'education_composite_risk', 'government_composite_risk',
                'healthcare_composite_risk'
            ]
        }


if __name__ == "__main__":
    # Test feature engineering
    import sys
    sys.path.append('..')
    from data_loaders import DataAggregator
    from config import DATA_DIR
    
    # Load data
    aggregator = DataAggregator(DATA_DIR)
    raw_data = aggregator.load_all()
    weekly_df = aggregator.create_weekly_aggregation(raw_data)
    
    # Apply feature engineering
    pipeline = FeatureEngineeringPipeline()
    features_df = pipeline.fit_transform(weekly_df, raw_data)
    
    print(f"\nFinal feature set: {features_df.shape}")
    print(f"Features: {list(features_df.columns)[:20]}...")
    print(f"\nTarget distribution:")
    print(features_df['next_week_risk_level'].value_counts())
