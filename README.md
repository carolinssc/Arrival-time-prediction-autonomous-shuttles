# ETA-prediction-autonomous-shuttles

Official implementation of Estimated Time of Arrival in Autonomous Shuttle Services: Insights from 5 Real-World Pilots

## Contents

* `src/baselines` : Training scripts for the the Mean regressor, Lag Model, Linear regression, Random forest and XGboost


## Hyperparameter settings 

### Dwell time 

XGBoost: 
| City | learning rate    | max depth    |  n estimators| reg lambda | subsample  | gamma |
| :---:   | :---:           | :---: |        :---: |       :---: |      :---: |  :---: |
| Linköping |   0.1   | 6   | 100  | 2  |0.9 | 0 |
| Tampere  |   0.03   | 6   | 80  | 5  |0.7 |0 |
| Rouen |   0.1   | 6   | 60  | 2  |0.9 |  0.25 |
| Madrid |   0.03   | 4   | 40  | 5  |0.9 |  0 |
| Graz |   0.1   | 20  | 30  | 2  |0.8 |0.1 |
