
import argparse
import numpy as np 
import pandas as pd 
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import  GridSearchCV
import xgboost as xgb
import os 
import json 


def simple_model_evaluation(y_test, y_pred_test):
    # Simple model evaluation that computes and prints MSE, RMSE and MAPE for the training and testing set

    test_error_mse = np.square(y_test - y_pred_test).sum() / y_test.shape[0]


    test_error_mae = (1 / y_test.shape[0]) * (np.absolute(y_test - y_pred_test)).sum()
    print("-----------MSE----------")
    print("Testing error: {}".format(test_error_mse))
    print("-----------RMSE----------")
    print("Testing error: {}".format(np.sqrt(test_error_mse)))
    print("-----------MAE----------")
    print("Testing error: {}".format(test_error_mae))
    return np.sqrt(test_error_mse), test_error_mae


def MeanRegressor(data_train, data_test):
    # Compute the mean based on both 'segment_id' and 'vehicle_id'
    mean = data_train.groupby(['segment_id', 'vehicle_id'])['travel_time'].mean()

    # Map the computed mean to the test set based on both columns
    data_test['mean'] = data_test.set_index(['segment_id', 'vehicle_id']).index.map(mean)

    return data_test['mean']

def predict_lag(data_test):
    y_pred_test = data_test['lags1'].to_numpy()
    return y_pred_test

def Linear_regression(X_train, X_test, y_train): 
    lr_w = LinearRegression()
    lr_w.fit(X_train, y_train)

    y_pred_lr_test = lr_w.predict(X_test)
    return y_pred_lr_test

def random_forest(X_train, X_test, y_train): 
    rf_base = RandomForestRegressor()
    rf_base.fit(X_train, y_train) 

    y_pred_rf_base = rf_base.predict(X_test)

    #load params from json 
    with open(f"/data/processed/{city}_baselines/best_params_rf_travel_time.json", "r") as read_file:
        params = json.load(read_file)

    best_random = RandomForestRegressor(**params)
    # Fit the random search model
    best_random.fit(X_train, y_train)

    y_pred_rf_best = best_random.predict(X_test)

    return y_pred_rf_base, y_pred_rf_best

def hyperparameter_random_forest(): 
    n_estimators = [int(x) for x in np.linspace(start = 50, stop = 500, num = 50)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf,
                'bootstrap': bootstrap}

    rf = RandomForestRegressor()
    # Random search of parameters, using 3 fold cross validation, 
    # search across 100 different combinations, and use all available cores
    rf_random = GridSearchCV(estimator = rf, param_grid = random_grid, cv = 3, verbose=0, random_state=42, n_jobs = 15)
    # Fit the random search model
    rf_random.fit(X_train, y_train)

    best_params = rf_random.best_params_    
    #save tp json
    with open(f"/data/processed/{city}_baselines/best_params_rf_travel_time.json", "w") as outfile:
        json.dump(best_params, outfile)

def hyperparemter_xgboost(X_train, y_train):

    param_grid = { 
    # Learning rate shrinks the weights to make the boosting process more conservative
    "learning_rate": [0.03, 0.1, 0.3] ,
    # Maximum depth of the tree, increasing it increases the model complexity.
    "max_depth": [1, 4, 6, 10, 15, 20, 25, 30], 
    # Gamma specifies the minimum loss reduction required to make a split.
    'reg_lambda' : [1, 2, 5], 
    "n_estimators" :  [20, 30, 40, 50,  60,  70,  80,  90, 100],
    "subsample" : [i/10.0 for i in range(7,10)], 
     "gamma":  [0, 0.1, 0.25, 0.5]}
    
    regressor=xgb.XGBRegressor()

    random_search = GridSearchCV(estimator=regressor, 
                            param_grid=param_grid, 
                            n_jobs=15, 
                            verbose=0, cv = 3, random_state=42)

    random_search.fit(X_train, y_train)

    best_params = random_search.best_params_    
    #save tp json
    with open(f"/data/processed/{city}_baselines/best_params_xg_travel_time.json", "w") as outfile:
        json.dump(best_params, outfile)


def xgboost(X_train, X_test, y_train):
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    evallist = [(dtrain, 'train')]

    param = {'objective': 'reg:squarederror'}
    #param['nthread'] = 4
    param['eval_metric'] = 'rmse'
    param['eta'] = 0.1
    param['reg_lambda'] = 0.5
    param['max_depth'] = 4
    num_round =100

    with open(f"/data/processed/{city}_baselines/best_params_xg_travel_time.json", "r") as read_file:
        params = json.load(read_file)
        param.update(params)
    num_round = param['n_estimators']

    bst_dwell = xgb.train(param, dtrain, num_round, evallist)
    y_pred_test = bst_dwell.predict(dtest)
    
    return y_pred_test


if __name__ == "__main__":
    #add parameter via argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", type=str, default="linkoping")
    parser.add_argument("--num_vehicles", type=int, default=3)
    args = parser.parse_args()
    city = args.city
    num_vehicles = args.num_vehicles

    X_train = np.load(f"/data/processed/{city}_baselines/X_train_travel_time.npy", allow_pickle=True)
    X_test = np.load(f"/data/processed/{city}_baselines/X_test_travel_time.npy", allow_pickle=True)
    y_train = np.load(f"/data/processed/{city}_baselines/y_train_travel_time.npy", allow_pickle=True)
    y_test = np.load(f"/data/processed/{city}_baselines/y_test_travel_time.npy", allow_pickle=True)

    data_train = pd.read_csv(f"/data/processed/{city}_baselines/data_train_travel_time.csv")
    data_test = pd.read_csv(f"/data/processed/{city}_baselines/data_test_travel_time.csv")

    #mean regressor 
    print('Fit Mean regressor')
    y_pred_mean = MeanRegressor(data_train, data_test)
    print('Fit Lag regressor')
    y_pred_lag = predict_lag(data_test)
    print('Fit Linear regressor')
    y_pred_lr = Linear_regression(X_train, X_test, y_train)
    print('Fit Random Forest regressor')
    if not os.path.isfile(f"/data/processed/{city}_baselines/best_params_rf_travel_time.json"):
        print("No best params found...creating params")
        hyperparameter_random_forest()
    y_pred_rf_base, y_pred_rf_best = random_forest(X_train, X_test, y_train)
    print('Fit XGBoost regressor')
    if not os.path.isfile(f"/data/processed/{city}_baselines/best_params_xg_travel_time.json"):
        print("No best params found...creating params")
        hyperparemter_xgboost(X_train, y_train)
    y_pred_xgb = xgboost(X_train, X_test, y_train)

    #simple_model_evaluation
    print("-----------Mean----------")
    rmse, mae = simple_model_evaluation(y_test, y_pred_mean)
    print(f'{round(rmse,2)} & {round(mae,2)}')
    print("-----------Lag----------")
    rmse, mae = simple_model_evaluation(y_test, y_pred_lag)
    print(f'{round(rmse,2)} & {round(mae,2)}')
    print("-----------Linear Regression----------")
    rmse, mae = simple_model_evaluation(y_test, y_pred_lr)
    print(f'{round(rmse,2)} & {round(mae,2)}')
    print("-----------Random Forest Base----------")
    rmse, mae = simple_model_evaluation(y_test, y_pred_rf_base)
    print(f'{round(rmse,2)} & {round(mae,2)}')
    print("-----------Random Forest Best----------")
    rmse, mae = simple_model_evaluation(y_test, y_pred_rf_best)
    print(f'{round(rmse,2)} & {round(mae,2)}')
    print("-----------XGBoost----------")
    rmse, mae = simple_model_evaluation(y_test, y_pred_xgb)
    print(f'{round(rmse,2)} & {round(mae,2)}')

    
    data_pred = pd.DataFrame()
    data_pred['mean'] = y_pred_mean
    data_pred['lag'] = y_pred_lag
    data_pred['lr'] = y_pred_lr
    data_pred['rf'] = y_pred_rf_best
    data_pred['xgb'] = y_pred_xgb
    data_pred['y_test'] = y_test

    data_pred.to_csv(f"/data/processed/{city}_baselines/data_pred_travel_time.csv", index=False) 

