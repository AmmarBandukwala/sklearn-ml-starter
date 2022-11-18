import os
import mlflow
import mlflow.sklearn
import pandas as pd

from mlflow.entities import Experiment, Run, RunInfo, Param, Metric, RunTag, FileInfo, ViewType

# MLFlow Column Defnition
# 'run_id', 'experiment_id', 'status', 'artifact_uri', 'start_time',
# 'end_time', 'metrics.std_test_score', 'metrics.mean_fit_time',
# 'metrics.std_score_time', 'metrics.std_fit_time',
# 'metrics.mean_score_time', 'metrics.rank_test_score',
# 'metrics.mean_test_score', 'metrics.best_cv_score',
# 'metrics.training_mse', 'metrics.training_rmse', 'metrics.training_mae',
# 'metrics.training_r2_score', 'metrics.training_score',
# 'params.positive', 'params.copy_X', 'params.normalize', 'params.n_jobs',
# 'params.fit_intercept', 'params.pre_dispatch', 'params.estimator',
# 'params.verbose', 'params.best_positive', 'params.best_fit_intercept',
# 'params.param_grid', 'params.error_score', 'params.return_train_score',
# 'params.cv', 'params.scoring', 'params.refit', 'tags.estimator_class',
# 'tags.mlflow.autologging', 'tags.mlflow.user',
# 'tags.mlflow.source.name', 'tags.mlflow.runName',
# 'tags.mlflow.parentRunId', 'tags.mlflow.source.git.commit',
# 'tags.mlflow.source.type', 'tags.estimator_name',
# 'tags.mlflow.log-model.history'

def list_all_runs_for_experiment_by_name(experiment_name: str, filter_string = "", order_by: list[str] = ["attribute.start_time DESC"]) -> pd.DataFrame:
    experiment = mlflow.get_experiment_by_name(name=experiment_name)
    print(f"Artifact Location {experiment.artifact_location} for experiment {experiment.name} with id {experiment.experiment_id}.")
    df = mlflow.search_runs(experiment_ids=[experiment.experiment_id], filter_string=filter_string, order_by=order_by)
    return df

def get_run_id_lowest_rmse(df_experiments: pd.DataFrame) -> str:
    run_id = df_experiments.loc[df_experiments['metrics.training_rmse'].idxmin()]['run_id']
    return run_id

def get_model_by_run_id(run_id: str):    
    model = mlflow.sklearn.load_model("runs:/" + run_id + "/model")
    return model

def get_artifacts_by_run_id(run_id: str, download: bool = False) -> str:
    client = mlflow.tracking.MlflowClient()
    run = client.get_run(run_id=run_id)
    experiment_id = run.info.experiment_id
    experiment = client.get_experiment(experiment_id=experiment_id)
    model_name = run_id
    if experiment:
        model_name = experiment.name
    if run:
        dir_metadata: list[FileInfo] = client.list_artifacts(run_id=run_id)
        [print(info.path) for info in dir_metadata]
        dst_path = os.path.join('outputs', model_name)
        if not os.path.exists(dst_path):
            os.makedirs(dst_path, exist_ok=True)
        if download:
            path = client.download_artifacts(run_id=run_id, path='best_estimator/', dst_path=dst_path)
            return path
    return None