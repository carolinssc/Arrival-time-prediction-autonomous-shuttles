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
| City | n estimators    | min samples split    |  min samples leaf| max depth | bootstrap  |
| :---:   | :---:           | :---: |        :---: |       :---: |      :---: | 
| Linköping |   287  | 5   | 2  | 20  |True |
| Tampere  |   377   | 10  |  4  | 10 |True 
| Rouen |   491  | 5   | 4  | 10  |True |
| Madrid |   214   |  5   | 4  | 60  |True |
| Graz |   197   | 2  | 4  | 40  |True|

### Run time 

XGBoost: 
| City | learning rate    | max depth    |  n estimators| reg lambda | subsample  | gamma |
| :---:   | :---:           | :---: |        :---: |       :---: |      :---: |  :---: |
| Linköping |   0.1   | 6   | 60  | 1, |0.9 | 0.5 |
| Tampere  |   0.1   | 6   | 50  | 2  |0.9 |0 |
| Rouen |   0.1   | 4   | 50  | 5  |0.8 |  0.0 |
| Madrid |   0.1   | 4   | 50  | 1  |0.8 |  0 |
| Graz |   0.3   | 1  | 20  | 2  |0.8 |0.0 |


Random Forest: 
| City | n estimators    | min samples_split    |  min samples_leaf| max depth | bootstrap  |
| :---:   | :---:           | :---: |        :---: |       :---: |      :---: | 
| Linköping |   491  | 5   | 4  | 10  |True |
| Tampere  |   463   | 10  | 2  | 30 |True 
| Rouen |   123  | 10   | 4  | 10  |True |
| Madrid |   108   |  10   |  1  | 10  |True |
| Graz |   197   | 2  | 4  | 40  |True|

