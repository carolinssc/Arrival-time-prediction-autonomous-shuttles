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


Random Forest: 
| City | n_estimators    | min_samples_split    |  min_samples_leaf| max_depth | bootstrap  |
| :---:   | :---:           | :---: |        :---: |       :---: |      :---: | 
| Linköping |   287  | 5   | 2  | 20  |True |
| Tampere  |   377   | 10  |  4  | 10 |True 
| Rouen |   491  | 5   | 4  | 10  |True |
| Madrid |   214   |  5   | 4  | 60  |True |
| Graz |   197   | 2  | 4  | 400  |True|

