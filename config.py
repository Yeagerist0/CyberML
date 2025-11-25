"""
Configuration for Cyberattack Early-Warning System
"""
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# Create directories if they don't exist
MODELS_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)

# Data sources configuration
DATA_SOURCES = {
    "honeypot": DATA_DIR / "honeypot",
    "shodan": DATA_DIR / "shodan",
    "cve": DATA_DIR / "cve",
    "github_poc": DATA_DIR / "github_poc",
    "threat_intel": DATA_DIR / "threat_intel",
}

# Target sectors for risk assessment
TARGET_SECTORS = ["education", "government", "healthcare", "finance", "critical_infrastructure"]

# Time configuration
FORECAST_HORIZON_DAYS = 7  # Forecast next week
LAG_FEATURES_WEEKS = [1, 2, 3, 4]  # Use past 4 weeks as features
ANOMALY_LOOKBACK_DAYS = 30

# Risk levels
RISK_LEVELS = {
    0: "LOW",
    1: "MEDIUM",
    2: "HIGH",
    3: "CRITICAL"
}

# Model hyperparameters
MODEL_CONFIG = {
    "random_forest": {
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_split": 5,
        "class_weight": "balanced"
    },
    "gradient_boosting": {
        "n_estimators": 100,
        "learning_rate": 0.1,
        "max_depth": 5
    },
    "isolation_forest": {
        "n_estimators": 100,
        "contamination": 0.1
    }
}

# Text embedding config
EMBEDDING_CONFIG = {
    "model_name": "all-MiniLM-L6-v2",  # Sentence transformers model
    "max_seq_length": 256,
    "embedding_dim": 384
}

# Evaluation settings
EVALUATION_CONFIG = {
    "test_split": 0.2,
    "cv_folds": 5,
    "lead_time_thresholds": [1, 3, 5, 7],  # Days of lead time to evaluate
}
