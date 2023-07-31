# ETA-prediction-autonomous-shuttles

Official implementation of Estimated Time of Arrival in Autonomous Shuttle Services: Insights from 5 Real-World Pilots

## Contents

* `src/baselines` : Training scripts for the the Mean regressor, Lag Model, Linear regression, Random forest and XGboost


## Hyperparameter settings 

### Dwell time 

XGBoost: 
| City | learning rate    | max depth    |  n estimators| reg lambda | subsample  |
| :---:   | :---:           | :---: |        :---: |       :---: |      :---: |
| Linköping |   0.1   | 6   | 100  | 2  |0.9 |
