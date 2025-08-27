# AMEX Click Prediction Pipeline

A comprehensive machine learning pipeline for predicting click-through rates (CTR) on AMEX offers using advanced feature engineering and ensemble modeling techniques.

## Overview

This pipeline implements a state-of-the-art solution for click prediction on financial offers, combining multiple data sources and sophisticated feature engineering to achieve high performance in recommendation systems.

### Key Features

- **Advanced Feature Engineering**: Temporal, interaction, and behavioral features
- **Ensemble Modeling**: LightGBM and XGBoost with optimized hyperparameters  
- **Memory Optimization**: Efficient processing of large-scale datasets
- **SHAP Analysis**: Comprehensive feature importance and interpretability
- **Sampling Techniques**: Class imbalance handling with SMOTE, ADASYN, and others
- **Production Ready**: Modular design with comprehensive error handling

## Architecture

### Data Sources
- `train_data.parquet`: Training dataset with user-offer interactions
- `test_data.parquet`: Test dataset for predictions
- `add_event.parquet`: Event logs for CTR calculations
- `add_trans.parquet`: Transaction history for user profiling
- `offer_metadata.parquet`: Offer characteristics and metadata

### Feature Categories

1. **Temporal Features**
   - Cyclical encoding (hour, day, month)
   - Session-based metrics
   - Time decay functions
   - Holiday and business calendar features

2. **User Behavior Features**
   - Historical CTR by user and offer
   - Transaction patterns and spending habits
   - Engagement intensity metrics
   - Industry affinity scores

3. **Offer Characteristics**
   - Discount rates and categories
   - Brand and industry mappings
   - Popularity and recency metrics
   - Cross-offer interaction features

4. **Advanced Features**
   - Target encoding with Bayesian smoothing
   - Ranking and position bias correction
   - Neural embedding similarities
   - Multi-armed bandit exploration scores

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/amex-click-prediction.git
cd amex-click-prediction

# Install required packages
pip install -r requirements.txt
```

### Dependencies

```
pandas>=1.5.0
numpy>=1.21.0
scikit-learn>=1.1.0
lightgbm>=3.3.0
xgboost>=1.6.0
optuna>=3.0.0
shap>=0.41.0
imbalanced-learn>=0.9.0
psutil>=5.9.0
humanize>=4.0.0
holidays>=0.16.0
openpyxl>=3.0.0
pyarrow>=9.0.0
tqdm>=4.64.0
joblib>=1.2.0
```

## Usage

### Basic Pipeline Execution

```python
from amex_pipeline import AMEXClickPredictionPipeline

# Initialize pipeline
pipeline = AMEXClickPredictionPipeline()

# Run complete pipeline
pipeline.run_pipeline()
```

### Advanced Usage with Sampling

```python
# Run with SMOTE oversampling for imbalanced data
pipeline.run_pipeline(sampling_method='smote')

# Available sampling methods:
# - 'none': No sampling (default)
# - 'smote': SMOTE oversampling
# - 'adasyn': ADASYN oversampling  
# - 'undersample': Random undersampling
# - 'nearmiss': NearMiss undersampling
# - 'smoteenn': SMOTE + ENN
# - 'smotetomek': SMOTE + Tomek links
```


## Pipeline Stages

### 1. Data Loading and Optimization
- Memory-efficient data type casting
- Chunk-based processing for large datasets
- Automatic memory monitoring and cleanup

### 2. Feature Engineering
- **Time Features**: Cyclical encoding, session detection, decay functions
- **CTR Features**: Historical click-through rates by various dimensions
- **Transaction Features**: User spending patterns and preferences
- **Interaction Features**: Cross-feature combinations and ratios
- **Ranking Features**: Position bias correction and listwise optimization

### 3. Feature Selection
- Variance threshold filtering
- Random Forest importance-based selection
- SHAP value analysis for interpretability

### 4. Model Training
- **LightGBM Ensemble**: Multiple models with different seeds
- **XGBoost Classifier**: Gradient boosting with optimal parameters
- **Hyperparameter Optimization**: Optuna-based parameter tuning

### 5. Prediction and Submission
- Ensemble averaging of model predictions
- Proper formatting for competition submission
- Comprehensive validation and error checking

## Model Performance

The pipeline achieves competitive performance through:

- **Advanced Feature Engineering**: 200+ engineered features
- **Ensemble Methods**: Multiple model averaging reduces variance
- **Hyperparameter Optimization**: Automated tuning for optimal performance
- **Class Imbalance Handling**: Multiple sampling strategies available

## File Structure

```
amex-click-prediction/
├── README.md
├── requirements.txt
├── amex_pipeline.py          # Main pipeline class
├── data/                     # Data directory
│   ├── train_data.parquet
│   ├── test_data.parquet
│   ├── add_event.parquet
│   ├── add_trans.parquet
│   └── offer_metadata.parquet
├── models/                   # Saved models
├── submissions/              # Generated submissions
└── reports/                  # Feature importance reports
    └── variable_importance_report.xlsx
```

## Feature Importance Analysis

The pipeline generates comprehensive feature importance reports including:

- SHAP value analysis
- Conditional importance by user segments
- Feature category breakdowns
- Interaction effects visualization

Reports are automatically saved to `variable_importance_report.xlsx`.

## Memory Management

The pipeline includes sophisticated memory optimization:

- Automatic data type downcasting
- Chunk-based processing for large datasets
- Memory monitoring and cleanup
- Efficient feature storage and retrieval


## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citations

This pipeline incorporates techniques from recent academic research:

- Hierarchical Bayesian Smoothing (KDD 2022)
- Temporal Cross-validation (ICML 2023)  
- Listwise Ranking Optimization (SIGIR 2023)
- Neural Scoring Integration (CIKM 2022)
- Multi-armed Bandit Exploration (WSDM 2023)

## Authors

- Your Name - [paras1301sharma@gmail.com]

## Acknowledgments

- AMEX for providing the dataset and competition framework
- The open-source machine learning community for tools and techniques
- Research communities for advancing the state-of-the-art in recommendation systems
