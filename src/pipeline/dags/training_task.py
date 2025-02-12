""" Airflow DAG for training a CatBoost model and exporting it to S3 """

import datetime
import logging
import os
import shutil

import mlflow
import pandas as pd
from airflow.decorators import dag, task, task_group
from airflow.models import Connection
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from airflow.settings import Session
from airflow.utils.dates import days_ago
from catboost import CatBoostClassifier
from dotenv import load_dotenv
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


# create a dummy task for starting the pipeline
@task
def start_pipeline():
    """Dummy task to start the pipeline"""
    logging.info("Pipeline started")


@task
def create_s3_connection():
    """Create an S3 connection in Airflow."""
    # Load environment variables
    load_dotenv()
    aws_access_key_id = os.environ.get("S3_ACCESS_KEY")
    aws_secret_access_key = os.environ.get("S3_SECRET_KEY")
    host = os.environ.get("S3_HOST")

    s3_conn = Connection(
        conn_id="my_s3_connection",
        conn_type="aws",
        extra={
            "aws_access_key_id": aws_access_key_id,
            "aws_secret_access_key": aws_secret_access_key,
            "host": host,
        },
    )
    logging.info("S3 connection created")
    logging.info("s3_conn: %s", s3_conn)
    # Start a session with Airflow's database
    session = Session()

    # Check if this connection already exists to avoid duplicates
    if (
        not session.query(Connection)
        .filter(Connection.conn_id == s3_conn.conn_id)
        .first()
    ):
        # Add the new connection to the session
        session.add(s3_conn)
        # Commit the session to save the connection
        session.commit()
        logging.info("S3 connection added successfully.")
    else:
        logging.info("Connection already exists.")
    # Close the session
    session.close()


@task
def download_dataset(source_file, dest_file):
    """Download dataset from S3"""
    if not os.path.exists(dest_file):
        os.makedirs(dest_file)
    hook = S3Hook("my_s3_connection")
    file_name = hook.download_file(
        key=source_file, bucket_name="data-bucket", local_path=dest_file
    )
    logging.info("Downloaded file: %s", file_name)
    return file_name


@task
def read_data(file_path: str):
    """Read data from a parquet file"""
    try:
        data = pd.read_parquet(file_path)
        logging.info("Reading data from %s", file_path)
        return {"data": data}
    except FileNotFoundError:
        logging.exception("Unable to read data file, check file path")
        return {"data": None}


@task(multiple_outputs=True)
def split_data(data, test_size=0.3):
    """Split data into train and validation sets"""
    logging.info("Splitting data into train and validation sets")
    data = data["data"]
    # pylint: disable=invalid-name
    y = data["deposit"]
    X = data.drop(columns="deposit", axis=1)
    # pylint: disable=protected-access
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
    """Train a CatBoost model"""
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
        logging.info("Model saved at: %s", model_path)


@task
def export_model_to_s3():
    """
    Export the model to S3
    """
    local_model_path = "./catboost-model"
    s3_bucket = "model-bucket"
    hook = S3Hook("my_s3_connection")
    for file in os.listdir(local_model_path):
        hook.load_file(
            filename=f"{local_model_path}/{file}",
            key=f"model/{file}",
            bucket_name=s3_bucket,
        )
        logging.info("Uploaded %s to S3 bucket", file)


@task
def delete_temp_files():
    """Delete temporary files and directories"""
    logging.info("Deleting temporary files")
    for file in os.listdir("data/"):
        if file.startswith("airflow_tmp"):
            os.remove("data/" + file)
            logging.info("Deleted file: %s", file)
    logging.info("Temporary files deleted")
    logging.info("Deleting model directory")
    shutil.rmtree("./catboost-model")


@task
def end_pipeline():
    """Dummy task to end the pipeline"""
    logging.info("Pipeline completed")


@task_group
def get_data():
    """Get data from S3"""
    s3_connection = create_s3_connection()
    file = download_dataset("data/bank.parquet", "data/")
    data = read_data(file)
    s3_connection >> file >> data
    return data


@dag(
    default_args=default_args,
    schedule_interval=None,
    start_date=days_ago(1),
    catchup=False,
    tags=["mlops"],
)
def ml_training_pipeline():
    """Train a CatBoost model and export it to S3"""
    start = start_pipeline()
    data = get_data()
    training_data = split_data(data)
    model_training = train_log_model(training_data)
    model_export = export_model_to_s3()
    cleanup = delete_temp_files()
    end = end_pipeline()

    (start >> data >> training_data >> model_training >> model_export >> cleanup >> end)


dag_instance = ml_training_pipeline()
