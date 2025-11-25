# Cyberattack Early-Warning System

**Forecast next-week risk level of attacks on public-facing services using OSINT signals and adversary activity indicators.**

## Overview

This ML system provides proactive defense capabilities by analyzing multiple OSINT (Open Source Intelligence) data sources to predict elevated attack risk before incidents occur. It targets sectors like education, government, healthcare, and critical infrastructure.

## Features

### Data Sources (OSINT)
- **Honeypot Telemetry**: Attack patterns, volumes, and success rates
- **Shodan Snapshots**: Exposed services and vulnerability assessments
- **CVE Feeds**: New vulnerabilities, CVSS scores, exploit availability
- **GitHub PoC Emergence**: Proof-of-concept exploit tracking
- **Threat Intelligence**: APT activity, TTPs, and sector targeting

### ML Approach

1. **Classification with Lag Features**
   - Weekly aggregated features with 1-4 week lag values
   - Rolling statistics (mean, std, max) over time windows
   - Week-over-week change detection

2. **Anomaly Detection on Signal Bursts**
   - Isolation Forest for exploit signal burst detection
   - Z-score and IQR-based anomaly flagging
   - Rate-of-change anomaly detection

3. **Text Embeddings for PoC Similarity**
   - Sentence transformer embeddings for exploit descriptions
   - PoC-to-threat-intel similarity scoring
   - Exploit clustering density metrics

### Risk Levels
- **LOW (0)**: Normal activity levels
- **MEDIUM (1)**: Elevated signals, standard monitoring
- **HIGH (2)**: Significant threat indicators, enhanced monitoring
- **CRITICAL (3)**: Imminent threat, immediate action required

## Project Structure

```
CyberML/
├── config.py              # Configuration settings
├── main.py                # Main pipeline orchestration
├── requirements.txt       # Python dependencies
├── README.md              # This file
├── data/                  # Data directory
│   ├── honeypot          # Honeypot telemetry data
│   ├── shodan            # Shodan exposure data
│   ├── cve               # CVE feed data
│   ├── github_poc/       # GitHub PoC data
│   └── threat_intel/     # Threat intelligence data
├── models/               # Saved models
├── outputs/              # Evaluation reports and forecasts
└── src/
    ├── __init__.py
    ├── data_loaders.py    # Data ingestion from all sources
    ├── feature_engineering.py  # Feature creation pipeline
    ├── models.py          # ML models (classification + anomaly)
    └── evaluation.py      # Metrics and lead-time analysis
```

## Installation

```bash
# Clone/navigate to project directory
cd CyberML

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Run Full Pipeline

```bash
# Basic run with sample data
python main.py

# Compare different models before training
python main.py --compare-models

# Specify custom data/output directories
python main.py --data-dir /path/to/data --output-dir /path/to/output
```

### Generate Prediction Only

```bash
# Use pre-trained model to forecast next week's risk
python main.py --predict-only
```

### Programmatic Usage

```python
from main import CyberAttackEarlyWarningPipeline

# Initialize pipeline
pipeline = CyberAttackEarlyWarningPipeline()

# Run full pipeline
results = pipeline.run_full_pipeline(compare_models=True)

# Access results
forecast = results['forecast']
report = results['evaluation_report']

print(f"Predicted Risk: {forecast['predicted_risk_level']}")
print(f"Confidence: {forecast['confidence']:.2%}")
```

### Individual Components

```python
from src.data_loaders import DataAggregator
from src.feature_engineering import FeatureEngineeringPipeline
from src.models import HybridRiskPredictor
from src.evaluation import ComprehensiveEvaluationReport

# Load data
aggregator = DataAggregator(data_dir)
raw_data = aggregator.load_all()
weekly_data = aggregator.create_weekly_aggregation(raw_data)

# Engineer features
pipeline = FeatureEngineeringPipeline()
features = pipeline.fit_transform(weekly_data, raw_data)

# Train model
model = HybridRiskPredictor(classifier_type='ensemble')
model.fit(X_train, y_train)

# Evaluate
evaluator = ComprehensiveEvaluationReport()
report = evaluator.generate_report(y_true, y_pred, y_proba)
```

## Evaluation Metrics

### Classification Performance
- **Precision/Recall/F1** (macro, weighted, per-class)
- **Elevated Risk Detection** (HIGH + CRITICAL detection rate)
- **AUC-ROC** for probability calibration

### Lead-Time Benefit Analysis
- **Average Lead Time**: Days of advance warning
- **Detection Rate**: Percentage of elevated-risk weeks predicted
- **Response Time Benefit**: Estimated hours saved per warning
- **False Alarm Rate**: Percentage of false positive predictions

## Sample Output

```
============================================================
NEXT WEEK RISK FORECAST
============================================================

Forecast Date: 2025-11-26
Target Week: 2025-12-01

Predicted Risk Level: HIGH
Confidence: 78.45%
Anomaly Score: 0.623

Risk Probabilities:
  LOW       : ██                   8.23%
  MEDIUM    : ████                 13.32%
  HIGH      : ████████████████     78.45%
  CRITICAL  : █                    0.00%

Active Anomaly Signals:
  - honeypot_total_attacks_is_burst: 1.000
  - poc_new_count_anomaly_score: 0.734
```

## Data Format

The system can work with sample/synthetic data or real OSINT feeds. For real data:

### Honeypot Data (JSON Lines or CSV)
```json
{"timestamp": "2025-11-20T10:30:00", "source_ip": "192.168.1.1", "dest_port": 22, "attack_type": "ssh_bruteforce", "target_sector": "education", "severity_score": 7.5}
```

### CVE Data
```json
{"published_date": "2025-11-20", "cve_id": "CVE-2025-1234", "cvss_score": 9.8, "exploit_available": true, "attack_vector": "network"}
```

### Shodan Data
```json
{"scan_date": "2025-11-20", "ip": "10.0.0.1", "port": 443, "service": "nginx", "vulnerability": "CVE-2024-0001", "sector": "government"}
```

## Model Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    HybridRiskPredictor                      │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐    ┌─────────────────────────────────┐ │
│  │ Anomaly Detector│    │      Ensemble Classifier        │ │
│  │ (IsolationForest│───▶│  ┌───────────┬─────────────────┐│ │
│  │  on burst cols) │    │  │RandomForest│GradientBoosting││ │
│  └─────────────────┘    │  └───────────┴─────────────────┘│ │
│         │               │         │                       │ │
│         │               │         ▼                       │ │
│         │               │  ┌─────────────────────────────┐│ │
│         └──────────────▶│  │   Soft Voting Ensemble     ││ │
│        (anomaly_prob)   │  └─────────────────────────────┘│ │
│                         └─────────────────────────────────┘ │
│                                    │                        │
│                                    ▼                        │
│                         ┌─────────────────────────────────┐ │
│                         │  Risk Level + Probabilities     │ │
│                         │  [LOW, MEDIUM, HIGH, CRITICAL]  │ │
│                         └─────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## Configuration

Edit `config.py` to customize:

```python
# Forecast settings
FORECAST_HORIZON_DAYS = 7      # Predict 1 week ahead
LAG_FEATURES_WEEKS = [1,2,3,4] # Historical lookback

# Model hyperparameters
MODEL_CONFIG = {
    "random_forest": {"n_estimators": 100, "max_depth": 10},
    "gradient_boosting": {"n_estimators": 100, "learning_rate": 0.1}
}

# Target sectors
TARGET_SECTORS = ["education", "government", "healthcare", "finance"]
```

## Extending the System

### Adding New Data Sources

1. Create a new loader class in `data_loaders.py`:
```python
class NewSourceLoader:
    def __init__(self, data_path):
        self.data_path = data_path
    
    def load(self) -> pd.DataFrame:
        # Load and return data
        pass
```

2. Add aggregation method in `DataAggregator`:
```python
def _aggregate_new_source(self, df, start, end) -> Dict:
    # Return weekly features
    pass
```

### Custom Features

Add new features in `feature_engineering.py`:
```python
def generate_custom_features(self, df):
    df['my_feature'] = ...
    return df
```

## License

MIT License - See LICENSE file for details.

## Contributing

Contributions welcome! Please submit pull requests for:
- New data source integrations
- Improved feature engineering
- Model architecture enhancements
- Documentation improvements

## Disclaimer

This system is for research and defensive purposes. Always follow responsible disclosure practices and applicable laws when collecting OSINT data.
