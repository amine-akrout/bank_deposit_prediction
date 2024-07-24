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
    logging.info("Pipeline started")


@task
def create_s3_connection():
    # Load environment variables
    load_dotenv()
    aws_access_key_id = os.environ.get("S3_ACCESS_KEY")
    print(f"aws_access_key_id: {aws_access_key_id}")
    aws_secret_access_key = os.environ.get("S3_SECRET_KEY")
    print(f"aws_secret_access_key: {aws_secret_access_key}")

    s3_conn = Connection(
        conn_id="my_s3_connection",
        conn_type="aws",
        extra={
            "aws_access_key_id": aws_access_key_id,
            "aws_secret_access_key": aws_secret_access_key,
            "host": "http://minio:9000",
        },
    )
    logging.info("S3 connection created")
    logging.info(f"s3_conn: {s3_conn}")
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
    hook = S3Hook("my_s3_connection")
    file_name = hook.download_file(
        key=source_file, bucket_name="data-bucket", local_path=dest_file
    )
    logging.info(f"Downloaded file: {file_name}")
    return file_name


@task
def read_data(file_path: str):
    try:
        data = pd.read_parquet(file_path)
        logging.info(f"Reading data from {file_path}")
        return {"data": data}
    except FileNotFoundError:
        logging.exception("Unable to read data file, check file path")


@task(multiple_outputs=True)
def split_data(data, test_size=0.3):
    logging.info("Splitting data into train and validation sets")
    data = data["data"]
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


# create a task to delete the temporary files created during the pipeline (they are inside data/ folder and start with "airflow_tmp*")
@task
def delete_temp_files():
    logging.info("Deleting temporary files")
    for file in os.listdir("data/"):
        if file.startswith("airflow_tmp"):
            os.remove(f"data/{file}")
            logging.info(f"Deleted file: {file}")


@task
def end_pipeline():
    logging.info("Pipeline completed")


@task_group
def get_data():
    s3_connection = create_s3_connection()
    file = download_dataset("data/bank.parquet", "data/")
    data = read_data(file)
    s3_connection >> file >> data
    return data


@dag(
    default_args=default_args,
    schedule_interval=datetime.timedelta(days=1),
    start_date=datetime.datetime(2022, 1, 1),
    catchup=False,
    tags=["mlops"],
)
def ml_training_pipeline():

    start = start_pipeline()

    data = get_data()
    # training_data = split_data(data)
    # model_training = train_log_model(training_data)
    cleanup = delete_temp_files()
    # end = end_pipeline()
    # start >> data >> training_data >> model_training >> end

    start >> data >> cleanup


dag_instance = ml_training_pipeline()
