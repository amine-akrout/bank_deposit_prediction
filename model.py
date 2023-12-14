"""
This module trains the model and saves it in the mlruns directory
"""
import datetime
import logging
import shutil

import mlflow
import pandas as pd
from catboost import CatBoostClassifier
from prefect import task, flow
from prefect.tasks import task_input_hash
from sklearn.model_selection import train_test_split

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s: [%(levelname)s]: %(message)s"
)


@task(
    name="read_data",
    cache_key_fn=task_input_hash,
    cache_expiration=datetime.timedelta(days=1),
)
def read_data(file_path):
    """Read data from a csv file

    Args:
        file_path (str): path to the data file

    Returns:
        pandas.DataFrame: data frame containing the data
    """
    try:
        data = pd.read_csv(file_path)
        logging.info("Reading data from %s", file_path)
    except FileNotFoundError:
        logging.exception("Unable to read data file, check file path")
    return data


@task(name="split_data", cache_key_fn=task_input_hash)
def split_data(data, test_size=0.3):
    """Split data into train and validation sets

    Args:
        data (pandas.DataFrame): data frame containing the data
        test_size (float, optional): size of validation set. Defaults to 0.3.

    Returns:
        tuple: tuple containing train and validation sets
    """
    # pylint: disable=invalid-name, protected-access
    logging.info("Splitting data into train and validation sets")
    y = data["deposit"]
    X = data.drop(columns="deposit", axis=1)
    cat_features = list(set(X.columns) - set(X._get_numeric_data().columns))
    X[cat_features] = X[cat_features].astype(str)
    x_train, x_valid, y_train, y_valid = train_test_split(
        X, y, test_size=test_size, random_state=1234
    )
    logging.info("Data split into train and validation")
    return x_train, x_valid, y_train, y_valid, cat_features


def fetch_logged_data(run_id):
    """Fetch logged data from MLflow

    Args:
        run_id (str): Unique identifier for runs

    Returns:
        params (dict): params for the model
        metrics (dict): metrics for the model
        tags (dict): tags for the model
        artifacts (list): list of artifacts
    """
    client = mlflow.tracking.MlflowClient()
    data = client.get_run(run_id).data
    tags = {k: v for k, v in data.tags.items() if not k.startswith("mlflow.")}
    artifacts = [f.path for f in client.list_artifacts(run_id, "model")]
    return data.params, data.metrics, tags, artifacts


@task(name="train_log_model")
def train_log_model(x_train, x_valid, y_train, y_valid, cat_features):
    """Train and log model

    Args:
        x_train (pandas.DataFrame): training data
        x_valid (pandas.DataFrame): validation data
        y_train (pandas.DataFrame): training labels
        y_valid (pandas.DataFrame): validation labels
        cat_features (list): list of categorical features
    """
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
        train_mertics = cb_model.get_best_score()["learn"]
        logging.info(train_mertics)
        validation_mertics = cb_model.get_best_score()["validation"]
        logging.info(validation_mertics)
        # mlflow.log_metric('train_AUC', train_mertics['AUC'])
        mlflow.log_metric("train_loglooss", train_mertics["Logloss"])
        mlflow.log_metric("val_AUC", validation_mertics["AUC"])
        mlflow.log_metric("val_logloss", validation_mertics["Logloss"])
        mlflow.log_params(params)

        # Log catboost model
        mlflow.sklearn.log_model(cb_model, "catboost-model")

        params, metrics, tags, artifacts = fetch_logged_data(run.info.run_id)
        logging.info("Params: %s", params)
        logging.info("Metrics: %s", metrics)
        logging.info("Tags: %s", tags)
        logging.info("Artifacts: %s", artifacts)
        logging.info("Run ID: %s", run.info.run_id)

        model_path = f"mlruns/0/{run.info.run_id}/artifacts/catboost-model"
        logging.info("Model saved at: %s", model_path)

        shutil.copytree(model_path, "./catboost-model", dirs_exist_ok=True)


@flow(
    name="training flow",
    description="This flow trains the model and saves it in the mlruns directory",
    retries=2,
    log_prints=True,
)
def main():
    """Main function"""
    data = read_data("data/bank.csv")
    x_train, x_valid, y_train, y_valid, cat_features = split_data(data)
    train_log_model(x_train, x_valid, y_train, y_valid, cat_features)


if __name__ == "__main__":
    main()
