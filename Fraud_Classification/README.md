# Credit Card Fraud Detection System

## Overview
A machine learning system for real-time credit card fraud detection that addresses extreme class imbalance by using advanced techniques like SMOTE oversampling, XGBoost modeling, and precision-recall optimization.

## Architecture
- **Data Processing**: Feature engineering, outlier handling, categorical encoding
- **Imbalance Handling**: SMOTE synthetic oversampling to balance fraud cases (1:3 ratio)
- **Model Training**: XGBoost with scale_pos_weight vs Logistic Regression baseline
- **Threshold Optimization**: Precision-Recall curve analysis for business-optimal threshold
- **Evaluation**: Focus on fraud recall while maintaining acceptable precision

## Tech Stack
- **Python** - Core language and data processing
- **pandas/numpy** - Data manipulation and analysis
- **scikit-learn** - Machine learning models and evaluation metrics
- **XGBoost** - Gradient boosting for imbalanced classification
- **imbalanced-learn** - SMOTE oversampling technique
- **matplotlib/seaborn** - Data visualization and model diagnostics

## How to Run

1. **Install dependencies**:
   ```bash
   pip install pandas numpy scikit-learn xgboost imbalanced-learn matplotlib seaborn
   ```

2. **Run the analysis**:
   ```bash
   python main.py
   ```

3. **Or open in Jupyter**:
   ```bash
   jupyter notebook main.ipynb
   ```

## Key Decisions
- **SMOTE Oversampling**: Conservative 1:3 ratio instead of 1:1 to avoid overfitting to synthetic data
- **XGBoost Selection**: Superior performance on imbalanced data with built-in regularization
- **Threshold Tuning**: Optimized for business balance between fraud detection and false positives
- **Feature Engineering**: Domain-specific flags (high amounts, unusual hours, extreme ages)
- **Evaluation Focus**: Prioritized fraud recall over overall accuracy given business constraints

## Usage
The system processes transaction data with features like amount, time, location, and merchant category to predict fraud probability. Use the optimal threshold (0.246) for real-time scoring and manual review flagging.

## Results
- **Dataset**: 500 transactions (27 fraud cases, 5.4% imbalance)
- **XGBoost Performance**: 50% fraud recall, 7.55% precision, F1-score 0.131
- **Key Predictors**: Transaction amount, unusual hours, high-value flags
- **Business Impact**: Detects half of all fraud cases with manageable false positive rate
