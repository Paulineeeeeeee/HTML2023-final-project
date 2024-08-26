import pandas as pd
import calendar
from datetime import datetime
from scipy import stats
from scipy.stats import norm
import numpy as np
import pickle

 # 模型 
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso, ElasticNet
from sklearn.kernel_ridge import KernelRidge 
from sklearn.svm import SVR 
from sklearn.neighbors import KNeighborsRegressor 
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor 
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor 
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor 
from sklearn.ensemble import StackingRegressor 
import xgboost as xgb 
import lightgbm as lgb 
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer, mean_squared_error



def weighted_absolute_error(y_true, y_pred, s_i):
    

        # 计算绝对误差
    abs_error = abs(y_true - y_pred) / s_i

        # 计算权重
    weights = (np.abs(y_true / s_i - 1/3) + np.abs(y_true / s_i - 2/3))

        # 计算加权误差
    weighted_error = weights * abs_error * 3

        # 累加损失的平均值或总和
    avg_loss = np.mean(weighted_error)

    return avg_loss

def make_scorer_with_si(si_train):
    def weighted_absolute_error_with_si(y_true, y_pred):
        abs_error = np.abs(y_true - y_pred) / si_train
        weights = (np.abs(y_true / si_train - 1/3) + np.abs(y_true / si_train - 2/3))
        weighted_error = weights * abs_error * 3
        return np.mean(weighted_error)

    return make_scorer(weighted_absolute_error_with_si, greater_is_better=False)

def best_model(model, param_dist):

        scorer = make_scorer_with_si(si_train)
        random_search = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=100, cv=5, scoring=scorer, n_jobs=-1, verbose=2, random_state=42)
        random_search.fit(X_train, y_train)  # 假设 X_train 和 y_train 已被定义
        print("最佳参数:", random_search.best_params_)
        print("最佳模型评分:", random_search.best_score_)
        scorer_with_si_test = make_scorer_with_si(si_test)
        test_loss = scorer_with_si_test(best_model, X_test, y_test)
        print(f"{model} Loss: {test_loss}")
        
        return random_search.best_estimator_
    

if __name__ == '__main__':

    data = pd.read_csv("data.csv")
    X = (data.drop(['sbi'],axis = 1))
    y = (data['sbi'])

    # prediction
    print("prediction")
    with open('private_id.pkl', 'rb') as file:
        private_id = pickle.load(file)
    with open('private.pkl', 'rb') as file:
        private = pickle.load(file)

    private = private[['tot','station','is_weekend','month','day','weekday','hour','minute','high','low','weather','lng','lat']]
    

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    si_train = X_train['tot']
    si_test = X_test['tot']

    print('min max scaler')
    scaler = MinMaxScaler(feature_range=(0, 1)).fit(X)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    private = scaler.transform(private)




    svr = SVR()
    param_grid = {'C': [1, 10, 100], 'gamma': [0.001, 0.0001]}
    model = best_model(svr, param_grid)
    
    private_pred = model.predict(private)
    results_df = pd.DataFrame({
            'id': private_id,
            'sbi': private_pred
    })

    results_df.to_csv(f'outputs/svr.csv', index=False)

    rf_model = RandomForestRegressor()
    param_dist = {
                'max_depth': range(3, 11),
                'n_estimators': range(100, 201, 50),
        }
    model = best_model(rf_model, param_dist)
    
    private_pred = model.predict(private)
    results_df = pd.DataFrame({
            'id': private_id,
            'sbi': private_pred
    })

    results_df.to_csv(f'outputs/rf.csv', index=False)
    
    param_dist = {
                'eta': np.linspace(0.01, 0.3, 10),
                'max_depth': range(3, 11),
                'subsample': np.linspace(0.5, 1.0, 6),
                'colsample_bytree': np.linspace(0.3, 1.0, 8),
                'n_estimators': range(100, 1001, 100),
                'lambda': [0, 1, 2, 3, 4, 5],
                'alpha': [0, 1, 2, 3, 4, 5],
                'min_child_weight': range(1, 11)
        }
    
    xgb_reg = xgb.XGBRegressor()
    model = best_model(xgb_reg, param_dist)
    private_pred = model.predict(private)

    results_df = pd.DataFrame({
            'id': private_id,
            'sbi': private_pred
    })

    results_df.to_csv(f'outputs/xgboost.csv', index=False) 
    
    estimators = [
        ('lin', LinearRegression()),
        ('svr', SVR())
    ]
    stacking_reg = StackingRegressor(estimators=estimators, final_estimator=RandomForestRegressor())
    stacking_reg.fit(X_train, y_train)
    predictions = stacking_reg.predict(X_test)
    private_pred = stacking_reg.predict(private)
    print(f"the loss of stacking_reg: {weighted_absolute_error( predictions, y_test , si_test) }")

    results_df = pd.DataFrame({
            'id': private_id,
            'sbi': private_pred
    })

    results_df.to_csv(f'outputs/stacking_reg.csv', index=False)


    '''lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_train)
    predictions = lin_reg.predict(X_test)
    private_pred = lin_reg.predict(private)
    print(f"the loss of linear regreassion: {weighted_absolute_error( predictions, y_test , si_test) }")

    results_df = pd.DataFrame({
            'id': private_id,
            'sbi': private_pred
    })

    results_df.to_csv(f'outputs/lin.csv', index=False)

    # 运用GridSearchCV调参 
    params = { 'alpha': 1 / np.array([0.1, 1, 2, 3, 4, 10, 30, 100, 200, 300, 400, 800, 900, 1000])}
    find_best_params（）
    ridge = Ridge()
    ridge.fit(X_train, y_train)
    predictions = ridge.predict(X_test)
    private_pred = ridge.predict(private)
    print(f"the loss of ridge regreassion: {weighted_absolute_error( predictions, y_test , si_test) }")

    results_df = pd.DataFrame({
            'id': private_id,
            'sbi': private_pred
    })

    results_df.to_csv(f'outputs/ridge.csv', index=False)

    lasso = Lasso()
    lasso.fit(X_train, y_train)
    predictions = lasso.predict(X_test)
    private_pred = lasso.predict(private)

    print(f"the loss of lasso regreassion: {weighted_absolute_error( predictions, y_test , si_test) }")

    results_df = pd.DataFrame({
            'id': private_id,
            'sbi': private_pred
    })

    results_df.to_csv(f'outputs/lasso.csv', index=False)

    elastic_net = ElasticNet()
    elastic_net.fit(X_train, y_train)
    predictions = elastic_net.predict(X_test)
    private_pred = elastic_net.predict(private)

    print(f"the loss of elastic regreassion: {weighted_absolute_error( predictions, y_test , si_test) }")

    results_df = pd.DataFrame({
            'id': private_id,
            'sbi': private_pred
    })

    results_df.to_csv(f'outputs/elastic.csv', index=False)

    kernel_ridge = KernelRidge()
    kernel_ridge.fit(X_train, y_train)
    predictions = kernel_ridge.predict(X_test)
    private_pred = kernel_ridge.predict(private)

    print(f"the loss of kernel_ridge: {weighted_absolute_error( predictions, y_test , si_test) }")

    results_df = pd.DataFrame({
            'id': private_id,
            'sbi': private_pred
    })

    results_df.to_csv(f'outputs/kernel_ridge.csv', index=False)

    knn = KNeighborsRegressor()
    knn.fit(X_train, y_train)
    predictions = knn.predict(X_test)
    private_pred = knn.predict(private)

    print(f"the loss of knn: {weighted_absolute_error( predictions, y_test , si_test) }")


    results_df = pd.DataFrame({
            'id': private_id,
            'sbi': private_pred
    })

    results_df.to_csv(f'outputs/knn.csv', index=False)

    tree_reg = DecisionTreeRegressor()
    tree_reg.fit(X_train, y_train)
    predictions = tree_reg.predict(X_test)
    private_pred = tree_reg.predict(private)

    print(f"the loss of tree_reg: {weighted_absolute_error( predictions, y_test , si_test) }")


    results_df = pd.DataFrame({
            'id': private_id,
            'sbi': private_pred
    })

    results_df.to_csv(f'outputs/tree_reg.csv', index=False)

    extra_tree_reg = ExtraTreeRegressor()
    extra_tree_reg.fit(X_train, y_train)
    predictions = extra_tree_reg.predict(X_test)
    private_pred = extra_tree_reg.predict(private)

    print(f"the loss of extra_tree_reg: {weighted_absolute_error( predictions, y_test , si_test) }")


    results_df = pd.DataFrame({
            'id': private_id,
            'sbi': private_pred
    })

    results_df.to_csv(f'outputs/extra_tree_reg.csv', index=False)

    random_forest = RandomForestRegressor()
    random_forest.fit(X_train, y_train)
    predictions = random_forest.predict(X_test)
    private_pred = random_forest.predict(private)
    print(f"the loss of random_forest: {weighted_absolute_error( predictions, y_test , si_test) }")

    results_df = pd.DataFrame({
            'id': private_id,
            'sbi': private_pred
    })

    results_df.to_csv(f'outputs/random_forest.csv', index=False)

    ada_boost = AdaBoostRegressor()
    ada_boost.fit(X_train, y_train)
    predictions = ada_boost.predict(X_test)
    private_pred = ada_boost.predict(private)
    print(f"the loss of ada_boost: {weighted_absolute_error( predictions, y_test , si_test) }")

    results_df = pd.DataFrame({
            'id': private_id,
            'sbi': private_pred
    })

    results_df.to_csv(f'outputs/ada_boost.csv', index=False)

    gradient_boosting = GradientBoostingRegressor()
    gradient_boosting.fit(X_train, y_train)
    predictions = gradient_boosting.predict(X_test)
    private_pred = gradient_boosting.predict(private)
    print(f"the loss of gradient_boosting: {weighted_absolute_error( predictions, y_test , si_test) }")

    results_df = pd.DataFrame({
            'id': private_id,
            'sbi': private_pred
    })

    results_df.to_csv(f'outputs/gradient_boosting.csv', index=False)

    xgb_model = xgb.XGBRegressor()
    xgb_model.fit(X_train, y_train)
    predictions = xgb_model.predict(X_test)

    private_pred = xgb_model.predict(private)
    print(f"the loss of xgb_model: {weighted_absolute_error( predictions, y_test , si_test )}")

    results_df = pd.DataFrame({
            'id': private_id,
            'sbi': private_pred
    })

    results_df.to_csv(f'outputs/xgb_model.csv', index=False)

    lgb_model = lgb.LGBMRegressor()
    lgb_model.fit(X_train, y_train)
    predictions = lgb_model.predict(X_test)

    private_pred = lgb_model.predict(private)
    print(f"the loss of lgb_model: {weighted_absolute_error( predictions, y_test , si_test) }")

    results_df = pd.DataFrame({
            'id': private_id,
            'sbi': private_pred
    })

    results_df.to_csv(f'outputs/lgb_model.csv', index=False)'''


    estimators = [
        ('ridge', Ridge()),
        ('svr', SVR())
    ]
    stacking_reg = StackingRegressor(estimators=estimators, final_estimator=RandomForestRegressor())
    stacking_reg.fit(X_train, y_train)
    predictions = stacking_reg.predict(X_test)
    private_pred = stacking_reg.predict(private)
    print(f"the loss of stacking_reg: {weighted_absolute_error( predictions, y_test , si_test) }")

    results_df = pd.DataFrame({
            'id': private_id,
            'sbi': private_pred
    })

    results_df.to_csv(f'outputs/stacking_reg.csv', index=False)