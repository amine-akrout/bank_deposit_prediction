import datetime
import logging
import shutil

import mlflow
import pandas as pd
from airflow.decorators import dag, task
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s: [%(levelname)s]: %(message)s"
)

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": datetime.timedelta(minutes=5),
}


@dag(
    default_args=default_args,
    schedule_interval=datetime.timedelta(days=1),
    start_date=datetime.datetime(2022, 1, 1),
    catchup=False,
    tags=["mlops"],
)
def ml_training_pipeline():
    # create a dummy task for starting the pipeline
    @task
    def start_pipeline():
        logging.info("Pipeline started")

    @task
    def read_data(file_path: str):
        try:
            data = pd.read_csv(file_path)
            logging.info(f"Reading data from {file_path}")
            return data
        except FileNotFoundError:
            logging.exception("Unable to read data file, check file path")

    @task(multiple_outputs=True)
    def split_data(data: pd.DataFrame, test_size=0.3):
        logging.info("Splitting data into train and validation sets")
        y = data["deposit"]
        X = data.drop(columns="deposit", axis=1)
        cat_features = list(set(X.columns) - set(X._get_numeric_data().columns))
        X[cat_features] = X[cat_features].astype(str)
        x_train, x_valid, y_train, y_valid = train_test_split(
            X, y, test_size=test_size, random_state=1234
        )
        return {
            "x_train": x_train,
            "x_valid": x_valid,
            "y_train": y_train,
            "y_valid": y_valid,
            "cat_features": cat_features,
        }

    @task
    def train_log_model(training_data):
        x_train = training_data["x_train"]
        x_valid = training_data["x_valid"]
        y_train = training_data["y_train"]
        y_valid = training_data["y_valid"]
        cat_features = training_data["cat_features"]

        with mlflow.start_run(run_name="run") as run:
            params = {
                "loss_function": "Logloss",
                "eval_metric": "AUC",
                "verbose": 200,
                "random_seed": 1234,
            }
            cb_model = CatBoostClassifier(**params)
            cb_model.fit(
                x_train,
                y_train,
                cat_features=cat_features,
                eval_set=(x_valid, y_valid),
                use_best_model=True,
                plot=False,
            )
            mlflow.log_params(params)
            mlflow.sklearn.log_model(cb_model, "catboost-model")

            model_path = f"mlruns/0/{run.info.run_id}/artifacts/catboost-model"
            shutil.copytree(model_path, "./catboost-model", dirs_exist_ok=True)
            logging.info(f"Model saved at: {model_path}")

    @task
    def end_pipeline():
        logging.info("Pipeline completed")

    start = start_pipeline()
    data = read_data("data/bank.csv")
    training_data = split_data(data)
    model_training = train_log_model(training_data)
    end = end_pipeline()
    start >> data >> training_data >> model_training >> end


dag_instance = ml_training_pipeline()
