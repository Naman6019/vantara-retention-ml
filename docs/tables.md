## Selected model (from model_card.json)

- Churn model: **Logistic Regression** (threshold 0.6709)
- Value model: **LightGBM Tweedie**
- Validation ROC-AUC: 0.7967 | Test ROC-AUC: 0.7559 | Test recall at the operating threshold: 0.7134
- CLV validation MAE: £283.76 | top-decile value capture: 0.6391


## Churn model comparison (validation-ranked)

| Model               |   Val ROC-AUC |   Val Recall |   Val Precision |   Val F1 |   Val Accuracy |   CV ROC-AUC (fwd-chain) | use_scaled   |
|:--------------------|--------------:|-------------:|----------------:|---------:|---------------:|-------------------------:|:-------------|
| Logistic Regression |        0.7967 |       0.7552 |          0.8099 |   0.7816 |         0.7325 |                   0.7815 | True         |
| Random Forest       |        0.7897 |       0.8143 |          0.794  |   0.804  |         0.7484 |                   0.8347 | False        |
| Decision Tree       |        0.7873 |       0.8041 |          0.8003 |   0.8022 |         0.7487 |                   0.774  | False        |
| XGBoost             |        0.7617 |       0.8077 |          0.7698 |   0.7883 |         0.7251 |                   0.8869 | False        |
| LightGBM            |        0.7602 |       0.7807 |          0.7805 |   0.7806 |         0.7218 |                   0.8613 | False        |
| KNN                 |        0.7562 |       0.8158 |          0.7697 |   0.7921 |         0.7286 |                   0.7976 | True         |


## Value model comparison

| Model                     |   Val R² Score |   Val MAE (£) |   Val RMSE (£) |   Top-Decile Value Capture |   Training Time (s) |
|:--------------------------|---------------:|--------------:|---------------:|---------------------------:|--------------------:|
| LightGBM Tweedie          |         0.7543 |        283.76 |        1065.09 |                     0.6391 |               0.356 |
| Random Forest Regressor   |         0.7986 |        314.04 |         964.24 |                     0.6386 |               2.958 |
| Spend Baseline (last 90d) |        -0.03   |        372.06 |        2180.61 |                     0.1253 |               0     |
| Ridge Regression (log1p)  |        -2.6231 |        442.34 |        4089.81 |                     0.584  |               0.016 |


## 5-fold stratified CV (PRD Section 11)

| Model               |   CV ROC-AUC (5-fold) |   CV std |   Folds |
|:--------------------|----------------------:|---------:|--------:|
| XGBoost             |                0.9097 |   0.0027 |       5 |
| LightGBM            |                0.8824 |   0.0016 |       5 |
| Random Forest       |                0.8288 |   0.0022 |       5 |
| KNN                 |                0.7829 |   0.0048 |       5 |
| Logistic Regression |                0.7523 |   0.0061 |       5 |
| Decision Tree       |                0.7469 |   0.0053 |       5 |


*Forward-chaining CV remains the selection criterion because the same customer appears at several stacked snapshots; the 5-fold column is the PRD-required diagnostic on the training partition.*



## Test confusion matrix (threshold 0.6709)

|                 |   Predicted retained |   Predicted churn |
|:----------------|---------------------:|------------------:|
| Actual retained |                 1438 |               704 |
| Actual churned  |                  625 |              1556 |


Accuracy 69.3%, precision 68.8%, recall 71.3%, ROC-AUC 0.7559.
