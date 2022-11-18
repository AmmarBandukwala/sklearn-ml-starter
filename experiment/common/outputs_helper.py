
import os
import datetime as dt
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import sklearn.metrics as metrics

def regression_results(title: str, y_true: list[float], y_pred: list[float]) -> None:
    """Provide a summary view of the model performance.

    Args:
        title (str): Name of model.
        y_true (list[float]): y true values.
        y_pred (list[float]): y predicted values.
    """
    explained_variance=metrics.explained_variance_score(y_true, y_pred)
    mean_absolute_error=metrics.mean_absolute_error(y_true, y_pred) 
    mse=metrics.mean_squared_error(y_true, y_pred) 
    mean_squared_log_error=metrics.mean_squared_log_error(y_true, y_pred)
    median_absolute_error=metrics.median_absolute_error(y_true, y_pred)
    r2=metrics.r2_score(y_true, y_pred)
    print(f'---- Start {title} Results----')
    print('explained_variance: ', round(explained_variance,4))    
    print('mean_squared_log_error: ', round(mean_squared_log_error,4))
    print('median_absolute_error: ', median_absolute_error)
    print('r2: ', round(r2,4))
    print('MAE: ', round(mean_absolute_error,4))
    print('MSE: ', round(mse,4))
    print('RMSE: ', round(np.sqrt(mse),4))
    print(f'---- End {title} Results----')

def _save_figure(fileName: str) -> None:
    """Save matlibplot figured in buffer to file on disk to outputs/plots folder relative to root of solution.

    Args:
        fileName (str): Name of file on disk.
    """
    os.makedirs(os.path.join('outputs', 'plots'), exist_ok=True)
    path = os.path.join('outputs', 'plots', fileName)
    plt.savefig(path, bbox_inches = 'tight')
    plt.clf()

def generate_stock_time_series_plot(data: dict[str, pd.DataFrame], fileName: str) -> None:
    """Take in a dictionary with title of company as key, and the value a pandas dictionary.
    Args:
        data (dict[str, pd.DataFrame]): Company Name, Pandas Data Frame w/Columns(Date, High, Low)
        fileName (str): Name of file output.
    """
    
    plt.title(','.join(data.keys()))
    plt.figure(figsize=(15,10))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=60))
    
    for title in data:
        df = data[title]
        x_dates = [dt.datetime.strptime(d, '%m/%d/%Y').date() for d in df.Date.values]
        plt.plot(x_dates, df['High'], label=f'{title}_High')
        plt.plot(x_dates, df['Low'], label=f'{title}_Low')
        
    plt.xlabel('Time Scale')
    plt.ylabel('Scaled USD')
    plt.legend()
    plt.gcf().autofmt_xdate()
    
    _save_figure(fileName)

def generate_box_plot(results: dict[str, object], fileName: str) -> None:
    """Create a Box Plot Graph

    Args:
        results (dict[str, object]): Dictionary of cv_results with model names as keys.
        fileName (str): Target file name output.
    """
    plt.boxplot(results.values(), labels=results.keys())
    plt.title('Algorithm Comparison')
    
    _save_figure(fileName)

def variable_importance_plot(imp: list[float], features: pd.Index, indices: list[int], fileName: str) -> None:
    """Create a Variable Important Bar Graph

    Args:
        imp (list[float]): Array of coefficient weights for each feature.
        features (pd.Index): Pandas index of features.
        indices (list[int]): Corresponding order of pandas index in conjunection with importantance weights.
        fileName (str): Target file name output.
    """
    plt.title('Feature Importances')
    plt.barh(range(len(indices)), imp[indices], color='b', align='center')
    plt.yticks(range(len(indices)), [features[i] for i in indices])
    plt.xlabel('Relative Importance')
    
    _save_figure(fileName)

def generate_xy_plot(y_test, y_train, y_pred, y_train_pred, fileName: str):
    plt.title('X-Y Train vs Test')
    plt.plot(np.arange(len(y_pred)) + len(y_train),y_test)
    plt.plot(np.arange(len(y_pred)) + len(y_train), y_pred)
    plt.plot(y_train)
    plt.plot(y_train_pred)
    plt.legend(['y_test', 'y_test_pred', 'y_train','y_train_pred'])

    _save_figure(fileName)
    