__author__ = "Ivar Vargas Belizario and Liz Maribel Huancapaza Hilasaca"
__copyright__ = "Copyright 2024, registration in process"
__credits__ = ["Ivar Vargas Belizario", "Liz Maribel Huancapaza Hilasaca"]
__license__ = "MIT"
__version__ = "1.0"
__maintainer__ = "Ivar Vargas Belizario"
__email__ = "ivargasbelizario@gmail.com"
__status__ = "development"


import warnings
warnings.simplefilter("ignore")

from sklearn.base import clone
import pandas as pd
import numpy as np
from sktime.datasets import load_airline
from sktime.forecasting.model_selection import temporal_train_test_split
# from sktime.utils.plotting.forecasting import plot_ys
from sktime.utils.plotting import plot_series

from datetime import datetime
# from sklearn.metrics import mean_absolute_percentage_error
from sktime.performance_metrics.forecasting import mean_absolute_percentage_error

from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor

from sktime.forecasting.model_selection import ExpandingWindowSplitter
from sktime.forecasting.model_selection import ForecastingGridSearchCV
from sktime.forecasting.compose import TransformedTargetForecaster
from sktime.forecasting.theta import ThetaForecaster
from sktime.forecasting.compose import make_reduction


from sklearn.ensemble import GradientBoostingRegressor
from sktime.split import SingleWindowSplitter
from sktime.forecasting.model_selection import ForecastingRandomizedSearchCV

from sktime.transformations.series.detrend import Deseasonalizer, Detrender
from sktime.forecasting.trend import PolynomialTrendForecaster

from sktime.utils import mlflow_sktime  
import os

pd.set_option('display.float_format', lambda x: '%.2f' % x)

def readdata():
    dateparse = lambda x: datetime.strptime(x, '%Y-%m')
    y = pd.read_csv('dataset/data.csv', header=0, index_col=0, parse_dates=['month'], date_parser=dateparse)
    y.index = pd.PeriodIndex(y.index, freq="M")
    return y

def savefigure(filename, y_test, y_pred):
    print(filename)
    fig, ax = plot_series(y_test, y_pred, labels=["real", "predicted"]);
    fig.savefig(filename, dpi=300, bbox_inches='tight')
    mape = mean_absolute_percentage_error(y_test, y_pred, symmetric=False)
    print('SMAPE %:', mape)
    return mape, y_test["sunat_income"].tolist(), y_pred["sunat_income"].tolist()

def transform2TimeSeriesRegressor(regressor):
    forecaster = TransformedTargetForecaster(
        [
            ("deseasonalize", Deseasonalizer(model="multiplicative", sp=24)),
            ("detrend", Detrender(forecaster=PolynomialTrendForecaster(degree=1))),
            ("forecast", make_reduction(regressor, window_length=24, strategy="recursive"),
            ),
        ]
    )
    return forecaster
    
    
# read the dataset in time series format
y_time = readdata()

# split in train a test subsets
y_train, y_test = temporal_train_test_split(y_time, test_size=12)

# define the size of test windows to make predictions
fh = np.arange(len(y_test)) + 1

# define the size of test windows to make predictions
fh2024 = np.arange(len(fh)+12) + 1

# window for cross-validation
validation_size = 12
cv = SingleWindowSplitter(window_length=len(y_train)-validation_size, fh=validation_size)


# define the models and the ranges of their parameters
models = {
    "XGBRegressor_RS":{
        "regressor": XGBRegressor(objective='reg:squarederror', random_state=42),
        "param_grid":{
            'deseasonalize__model': ['multiplicative', 'additive'],
            'detrend__forecaster__degree': [1, 2, 3],
            'forecast__estimator__max_depth': [3, 5, 6, 10, 15, 20],
            'forecast__estimator__learning_rate': [0.01, 0.1, 0.2, 0.3],
            'forecast__estimator__subsample': np.arange(0.5, 1.0, 0.1),
            'forecast__estimator__colsample_bytree': np.arange(0.4, 1.0, 0.1),
            'forecast__estimator__colsample_bylevel': np.arange(0.4, 1.0, 0.1),
            'forecast__estimator__n_estimators': [100, 200, 500, 1000]
        }
    },

    "RandomForestRegressor_RS":{
        "regressor": RandomForestRegressor(random_state=42),
        "param_grid":{
            'deseasonalize__model': ['multiplicative', 'additive'],
            # 'detrend__forecaster__degree': [1, 2, 3],
            "forecast__estimator__n_estimators": [50,100,200,300,500],
            'forecast__estimator__max_depth': [50, 80, 90, 100], # Maximum number of levels in each decision tree
            'forecast__estimator__max_features': [1, 2, 3, 4], # Maximum number of features considered for splitting a node
            'forecast__estimator__min_samples_leaf': [1, 3, 4, 5], # Minimum number of data points allowed in a leaf node
            
        }
    },

    "MLPRegressor_RS":{
        "regressor": MLPRegressor(random_state=42, verbose=False),
        "param_grid":{
            'deseasonalize__model': ['multiplicative', 'additive'],
            # 'detrend__forecaster__degree': [2, 3],

            # 'estimator__hidden_layer_sizes': [(150,100,50), (120,80,40), (100,50,30),
            # (50,50,50), (32,64,32), (50,100,50), (100,1)],
            # # 'estimator__max_iter': [50, 100],
            # 'estimator__activation': ['relu','tanh','logistic'],
            # 'estimator__alpha': [0.0001, 0.001, 0.01, 0.05],
            # 'estimator__learning_rate': ['constant','adaptive'],
            # 'estimator__solver': ['adam'],
            
            # 'forecast__estimator__hidden_layer_sizes': [(150,100,50), (120,80,40), (100,50,30),
            # (50,50,50), (32,64,32), (50,100,50), (100,1)],
            'forecast__estimator__hidden_layer_sizes': [
            (1,),(2,),(3,),(4,),(5,),(6,),(7,),(8,),(9,),(10,),(11,), (12,),(13,),(14,),(15,),(16,),(17,),(18,),(19,),(20,),(21,)
            ],
            'forecast__estimator__activation' : ['identity', 'logistic', 'tanh', 'relu'],
            'forecast__estimator__solver' : ['sgd', 'adam'],
            'forecast__estimator__alpha': [0.0001, 0.001, 0.01, 0.05],
            'forecast__estimator__learning_rate': ['constant', 'invscaling', 'adaptive'],
            # 'forecast__estimator__learning_rate_init': [ 0.01, 0.05],
            'forecast__estimator__max_iter':[300],
            # 'forecast__estimator__random_state':[42],
        }
    },

    "KNeighborsRegressor_RS":{
        "regressor": KNeighborsRegressor(),
        "param_grid":{
            'deseasonalize__model': ['multiplicative', 'additive'],
            # 'detrend__forecaster__degree': [2, 3],
            "forecast__estimator__n_neighbors": np.arange(1, 10),
        }
    },  

}

# regressor = XGBRegressor(objective='reg:squarederror', random_state=42)
# regressor = RandomForestRegressor(random_state=42)
# regressor = MLPRegressor(random_state=42)
# regressor = KNeighborsRegressor()

# to each model configuration make:
results2023 = []
results2024 = []
for name, value in models.items():
    regressor = value["regressor"]
    param_grid = value["param_grid"]

    # *********** 2023 *************
    # transfor from clasical regressor to time seriers regressor    
    forecaster = transform2TimeSeriesRegressor(clone(regressor))

    # training model and compute the hyperparameter tunig with randomised search    
    gscv = ForecastingRandomizedSearchCV(
        forecaster,cv=cv, param_distributions=param_grid, n_iter=400, random_state=42,)
    gscv.fit(y_train)
    bp = gscv.best_params_
    
    # make the predicitons on test subset (year 2023)
    y_pred = gscv.predict(fh)
    # compute plots and MAPE valeus
    mape, yre, ypr = savefigure("p_"+name+"_2023.png", y_test, y_pred)
    # save results
    results2023.append((name, mape, bp, yre, ypr))

    # save forecarter on file .h5
    model_path = os.path.join("models",name+'.joblib')
    
    mlflow_sktime.save_model(sktime_model=gscv,path=model_path)
    loaded_model = mlflow_sktime.load_model(model_uri=model_path)
    print("loaded_model", loaded_model)
    
#     loaded_model = mlflow_sktime.load_model(model_uri=model_path)  
#     loaded_model.predict(fh=[1, 2, 3])  

    # *********** 2024 *************    
    
#     y_pred = gscv.predict(fh2024)
#     # compute plots and MAPE valeus
#     mape, yre, ypr = savefigure("p_"+name+"_2024.png", y_test, y_pred)
#     # save results
#     results2024.append((name, mape, bp, yre, ypr))
    
    
    # transfor from clasical regressor to time seriers regressor    
    forecaster = transform2TimeSeriesRegressor(clone(regressor))

    # training model and compute the hyperparameter tunig with randomised search    
    gscv = ForecastingRandomizedSearchCV(
        forecaster,cv=cv, param_distributions=param_grid, n_iter=400, random_state=42,)
    gscv.fit(y_time)
    bp = gscv.best_params_
    
    # make the predicitons on test subset (year 2023)
    y_pred = gscv.predict(fh)
    # compute plots and MAPE valeus
    mape, yre, ypr = savefigure("p_"+name+"_2024.png", y_test, y_pred)
    # save results
    results2024.append((name, mape, bp, yre, ypr))
    


    
#--------------------------------------------
# other model
from sktime.forecasting.arima import AutoARIMA
arima_forecasting = AutoARIMA(sp=12, random_state=42)
arima_forecasting.fit(y_train)
y_arima = arima_forecasting.predict(fh)
mape, yre, ypr = savefigure("p_AutoARIMA_2023.png", y_test, y_arima)
# print results
print(mape, yre, ypr) 


arima_forecasting = AutoARIMA(sp=12, random_state=42)
arima_forecasting.fit(y_time)
y_arima = arima_forecasting.predict(fh)
mape, yre, ypr = savefigure("p_AutoARIMA_2024.png", y_test, y_arima)
# print results
print(mape, yre, ypr) 
#--------------------------------------------




# print results 
for na, mp, bp, yre, ypr in results2023:
    print (na, "MAPE",mp)
    print (bp)
    print (yre)
    print (ypr)
    print ()

# print results 
for na, mp, bp, yre, ypr in results2024:
#     print (na, "MAPE",mp)
    print (bp)
    print (yre)
    print (ypr)
    print ()

