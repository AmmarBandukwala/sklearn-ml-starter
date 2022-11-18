import argparse
import os
import sys
import mlflow
import pandas as pd
import numpy as np

from datetime import datetime
from common.bootstrap import setup
from common.model_helper import search_grid
from common.outputs_helper import generate_box_plot, generate_stock_time_series_plot, generate_xy_plot
from data_prep import load_data

from scipy.stats import pearsonr

from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# DEV
debug_path = os.path.join("experiment", "parameters", "parameters.yml")
parameters = setup(argparse.Namespace(parameter_file_local=debug_path, parameter_file_url=None))

# PROD
# parameters = setup()

# Initialize Parameters
data_path = parameters["GENERAL"]["Data_Path"]
model_name = parameters["GENERAL"]["Model_Name"]
features = parameters["PARAMETERS"]["Features"]
date_column = parameters["PARAMETERS"]["Index"]
target_feature = parameters["PARAMETERS"]["Target"]
train_year = datetime.strptime(parameters["PARAMETERS"]["Train_Year"], '%Y-%m-%d')
test_year = datetime.strptime(parameters["PARAMETERS"]["Test_Year"], '%Y-%m-%d')
hyper_parameters_catalog = parameters["PARAMETERS"]["Hyper_Parameters"]

# Start Run
mlflow.set_experiment(experiment_name=model_name)
run = mlflow.start_run()

# Load SP 500 Benchmark
sp_benchmark = load_data(data_path, 'SPX500_5_Year_Benchmark.csv')

# Load multiple test data sets from different company stocks.
catepillar = load_data(data_path, 'Caterpillar_5_Year_Historical.csv')
gme = load_data(data_path, 'GME_5_Year_Historical.csv')
tesla = load_data(data_path, 'Tesla_5_Year_Historical.csv')

# Create initial plot showing loaded data sets. (non-scaled)
generate_stock_time_series_plot(dict(zip(
    [catepillar.Title, gme.Title, tesla.Title],
    [catepillar.Data, gme.Data, tesla.Data]
)), 'compare_historic_5_year_stocks.png')

# Pearson's correlation coefficient = covariance(X, Y) / (stdv(X) * stdv(Y))
corr1, _ = pearsonr(catepillar.Data[target_feature], gme.Data[target_feature])
print(f'PEARSON_CORRELATION_CAT_GME: {corr1}')
corr2, _ = pearsonr(catepillar.Data[target_feature], tesla.Data[target_feature])
print(f'PEARSON_CORRELATION_CAT_TESLA: {corr2}')
corr3, _ = pearsonr(tesla.Data[target_feature], gme.Data[target_feature])
print(f'PEARSON_CORRELATION_TESLA_GME: {corr3}')

# Prepare Test / Train Data (Selected Tesla)
merged_df = pd.concat([tesla.Data], axis=0)
merged_df.loc[:, 'Yesterday_Close'] = merged_df.loc[:, target_feature].shift()
merged_df.loc[:, 'Yesterday_Diff'] = merged_df.loc[:, 'Yesterday_Close'].diff()
merged_df['Yesterday_Close'] = merged_df['Yesterday_Close'].fillna(0)
merged_df['Yesterday_Diff'] = merged_df['Yesterday_Diff'].fillna(0)

ml_df = merged_df.copy()
ml_df[date_column] = pd.to_datetime(ml_df[date_column])
ml_df.set_index(date_column, inplace=True)

# Pick years to train and test against.
X_train = ml_df.loc[:train_year].drop([target_feature], axis=1)
y_train = ml_df.loc[:train_year, target_feature]
X_test = ml_df.loc[str(test_year.year)].drop([target_feature], axis=1)
y_test = ml_df.loc[str(test_year.year), target_feature]

# Model Train - Evaluate multiple models and cross validate.
mlflow.autolog(disable=True)

models = {
    'LR': LinearRegression(),
    'NN': MLPRegressor(solver='lbfgs'),
    'KNN': KNeighborsRegressor(),
    'RF': RandomForestRegressor(n_estimators=10),
    'SVR': SVR(gamma='auto')
}

results = {}

for name in models:
    model = models[name]
    tscv = TimeSeriesSplit(n_splits=10)
    cv_results = cross_val_score(model, X_train, y_train, cv=tscv, scoring='r2')
    results[name] = cv_results
    print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

# KNN or NN (Your milage may vary) seems like the best model.
generate_box_plot(results, 'compare_models_box_plot.png')

mlflow.autolog(disable=False)

# Pick a valid model based on initial accuracy.
accuracy_results = {model_name:cv_result.mean() for model_name, cv_result in results.items()}
best_model_name = max(accuracy_results, key=accuracy_results.get)
model = models[best_model_name]
hyper_parameters = hyper_parameters_catalog[best_model_name]
print(f'{best_model_name} was chosen as the best model.')

print('Hyperparameter Optimization - Attempt 1 (Single Periods)')
best_score, best_model, y_true, y_pred = search_grid(
    model=model, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, param_search=hyper_parameters, title='Attempt 1')

# Let shift the values to get better results, lets use 2 period instead of one.
ml_df_2 = ml_df.copy()
ml_df_2['Yesterday_Close-1'] = ml_df_2['Yesterday_Close'].shift()
ml_df_2['Yesterday-1_Diff'] = ml_df_2['Yesterday_Close-1'].diff()
ml_df_2['Yesterday_Close-1'] = ml_df_2['Yesterday_Close-1'].fillna(0)
ml_df_2['Yesterday-1_Diff'] = ml_df_2['Yesterday-1_Diff'].fillna(0)

# Pick years to train and test against.
X_train = ml_df_2.loc[:train_year].drop([target_feature], axis=1)
y_train = ml_df_2.loc[:train_year, target_feature]
X_test = ml_df_2.loc[str(test_year.year)].drop([target_feature], axis=1)
y_test = ml_df_2.loc[str(test_year.year), target_feature]

print('Hyperparameter Optimization - Attempt 2 (Two Periods)')
best_score, best_model, y_true, y_pred = search_grid(
    model=model, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, param_search=hyper_parameters, title='Attempt 2')

# X-Y Train-Test
generate_xy_plot(y_test=y_test, y_train=y_train, y_pred=y_pred, y_train_pred=y_true, fileName='x_y_train_test.png')

# End Run
mlflow.end_run()

# Register the Model (Example if deploy tracking server with register.)
# mlflow.sklearn.log_model(
#     sk_model=best_model,
#     artifact_path="sk-ex-model",
#     registered_model_name="sk-learn-random-forest-model"
# )
