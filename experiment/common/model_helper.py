import pandas as pd
import numpy as np

from common.outputs_helper import regression_results

from sklearn.metrics import make_scorer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV

def _rmse(actual: list[float], predict: list[float]) -> float:
    """Calculate the Root Mean Square Error between actuals and the predictions.

    Args:
        actual (list[float]): An array of actual values.
        predict (list[float]): An array of the corresponding predicted values.

    Returns:
        float: Root Mean Square Error
    """
    predict = np.array(predict)
    actual = np.array(actual)
    distance = predict - actual
    square_distance = distance ** 2
    mean_square_distance = square_distance.mean()
    score = np.sqrt(mean_square_distance)
    return score

def search_grid(model, X_train: pd.DataFrame, y_train: pd.DataFrame, X_test: pd.DataFrame, y_test: pd.DataFrame, param_search, nsplits: int = 10, greater_is_better: bool = False, title: str = 'Results') -> tuple:
    """Hyper Parameter optimization routine to search for the optimum estimator.

    Args:
        model (object): Estimator to search against
        X_train (pd.DataFrame): Model X Train Dataset
        y_train (pd.DataFrame): Model y Train Dataset
        X_test (pd.DataFrame): Model X Test Dataset
        y_test (pd.DataFrame): Model y Test Dataset
        param_search (_type_): Hyperparameter Configuration
        nsplits (int, optional): TimeSeriesSplit number of splits. Defaults to 10.
        greater_is_better (bool, optional): If function is a scorer (positive) or loss (negative). Defaults to False.
        title (str, optional): Title of Model. Defaults to 'Results'.

    Returns:
        tuple: (Best Score, Best Model, Y True Values, Y Prediction Values)
    """
    
    rmse_score = make_scorer(_rmse, greater_is_better = greater_is_better)
    tscv = TimeSeriesSplit(n_splits=nsplits)
    gsearch = GridSearchCV(estimator=model, cv=tscv,
    param_grid=param_search, scoring=rmse_score)

    gsearch.fit(X_train, y_train)
    best_score = gsearch.best_score_
    best_model = gsearch.best_estimator_

    y_true = y_test.values
    y_pred = best_model.predict(X_test)

    regression_results(title, y_true, y_pred)
    
    return (best_score, best_model, y_true, y_pred)
