# Forest & Land Fire (Karhutla) Prediction Model using Spatiotemporal NASA FIRMS Data

This repository contains the source code, data preprocessing scripts, and model training pipelines for evaluating machine learning algorithms in predicting localized forest fire susceptibility across Indonesia. 

## Overview
Traditional reactive firefighting is costly and inefficient. This research adopts the **CRISP-ML framework** to build a proactive early warning system. By processing spatiotemporal thermal anomaly data from the **NASA MODIS satellite (FIRMS)**, the pipeline engineers temporal features, cyclical date representations, and rolling 7-day lookback windows to train and evaluate three distinct predictive architectures:
1. **Random Forest (RF):** Baseline ensemble model.
2. **XGBoost:** Gradient boosting model optimized for low false-alarm rates and high precision.
3. **Long Short-Term Memory (LSTM):** Recurrent neural network designed to capture sequential dependencies and maximize positive-class recall.

## Key Results
* **LSTM:** Achieved the highest overall discriminative ability (ROC-AUC: 0.721) and positive recall (36.3%), making it optimal for maximizing early detection and avoiding missed events.
* **XGBoost:** Yielded the highest Average Precision (0.193), serving as the ideal model when minimizing false alarms and preserving operational resources are prioritized.

