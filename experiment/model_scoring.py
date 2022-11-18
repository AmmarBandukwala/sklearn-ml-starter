import argparse
import os
import pandas as pd
import numpy as np

import common.mlflow_helper as mlflow_helper

from common.bootstrap import setup
from data_prep import StockData

def init():
    """Call function to load model and any other dependecies.
    """
    ## DEV 
    debug_path = os.path.join("experiment", "parameters", "parameters.yml")
    parameters = setup(argparse.Namespace(parameter_file_local=debug_path, parameter_file_url=None))

    ## PROD
    # parameters = setup()
    
    global data_path
    global model
    global date_column
    global target_feature
    
    data_path = parameters["GENERAL"]["Data_Path"]
    model_name = parameters["GENERAL"]["Model_Name"]    
    date_column = parameters["PARAMETERS"]["Index"]
    target_feature = parameters["PARAMETERS"]["Target"]
    
    exp_runs = mlflow_helper.list_all_runs_for_experiment_by_name(experiment_name=model_name)
    run_id = mlflow_helper.get_run_id_lowest_rmse(exp_runs)
    model = mlflow_helper.get_model_by_run_id(run_id=run_id)
    
def predict(data: pd.DataFrame) -> np.ndarray:
    """Invoke function with data set and appropiate columnar features to provide prediction.

    Args:
        data (pd.DataFrame): Input Model Data

    Returns:
        np.ndarray: Predictions
    """
    results = model.predict(data)
    return results

if __name__ == "__main__":
    
    # Load Model
    init()
    
    # Load Data
    stock_data = StockData('Microsoft', pd.read_csv(filepath_or_buffer=os.path.join(data_path, 'Microsoft_5_Year_Historical.csv')))
    test_data = StockData.convert_currency_float(stock_data.Data)
    
    test_data.loc[:, 'Yesterday_Close'] = test_data.loc[:, target_feature].shift()
    test_data.loc[:, 'Yesterday_Diff'] = test_data.loc[:, 'Yesterday_Close'].diff()
    test_data['Yesterday_Close'] = test_data['Yesterday_Close'].fillna(0)
    test_data['Yesterday_Diff'] = test_data['Yesterday_Diff'].fillna(0)
    
    test_data = test_data.drop(date_column, axis=1)
    test_data = test_data.drop(target_feature, axis=1)
    
    print(test_data.head())

    # Run Predict
    results = predict(test_data)
    print(results)