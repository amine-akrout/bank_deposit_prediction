import io
import logging
import os

import pandas as pd
from airflow.decorators import dag, task
from airflow.models import Connection
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from airflow.settings import Session
from airflow.utils.dates import days_ago
from dotenv import load_dotenv
from ucimlrepo import fetch_ucirepo

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s: [%(levelname)s]: %(message)s"
)

# Constants
BUCKET_NAME = "data-bucket"
OBJECT_NAME = "data/bank.parquet"


@task
def create_s3_connection():
    # Load environment variables
    load_dotenv()
    aws_access_key_id = os.environ.get("S3_ACCESS_KEY")
    aws_secret_access_key = os.environ.get("S3_SECRET_KEY")
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
def fetch_data():
    """Fetch data using the UCIML repository."""
    logging.info("Fetching data from the UCIML repository.")
    bank_marketing = fetch_ucirepo(id=222)
    features = bank_marketing.data.features
    targets = bank_marketing.data.targets
    # rename the target column to 'deposit'
    targets.columns = ["deposit"]
    return {"data": pd.concat([features, targets], axis=1)}


@task
def create_s3_bucket_if_not_exists():
    hook = S3Hook(aws_conn_id="my_s3_connection")
    # Check if the bucket already exists
    if not hook.check_for_bucket(BUCKET_NAME):
        try:
            # Create the bucket
            hook.create_bucket(BUCKET_NAME)
            logging.info(f"Bucket '{BUCKET_NAME}' created successfully .")
        except Exception as e:
            logging.error(f"Failed to create bucket '{BUCKET_NAME}'. Error: {e}")
    else:
        logging.info(f"Bucket '{BUCKET_NAME}' already exists. No action taken.")


@task
def upload_data_to_s3(data):
    """Upload the data to an S3 bucket."""
    data = data["data"]
    logging.info("Preparing data for upload to S3.")

    # Convert the DataFrame to a Parquet file in bytes
    buffer = io.BytesIO()
    data.to_parquet(buffer, index=False)
    buffer.seek(0)  # Move cursor to the beginning of the buffer before reading

    logging.info("Uploading data to S3.")
    # Create an instance of the S3Hook
    hook = S3Hook(
        aws_conn_id="my_s3_connection"
    )  # Ensure this connection ID exists in Airflow

    # Use the hook to upload data to the bucket
    hook.load_bytes(
        buffer.getvalue(),
        bucket_name=BUCKET_NAME,
        key=OBJECT_NAME,
        replace=True,  # Set to True to overwrite if the object already exists
    )
    logging.info(
        "Data uploaded successfully to bucket '{}' with key '{}'.".format(
            BUCKET_NAME, OBJECT_NAME
        )
    )


@task
def start_pipeline():
    logging.info("Pipeline started")


@task
def end_pipeline():
    logging.info("Pipeline ended")


# Define the DAG
@dag(
    schedule_interval="@daily",
    start_date=days_ago(1),
    catchup=False,
    tags=["etl"],
    default_args={"owner": "airflow", "retries": 1},
)
def etl_to_s3_dag():
    # Define Task Flow
    start = start_pipeline()
    s3_connection = create_s3_connection()
    data = fetch_data()
    create_s3_bucket = create_s3_bucket_if_not_exists()
    upload_data = upload_data_to_s3(data)
    end = end_pipeline()

    # Define Task Dependencies
    (start >> s3_connection >> data >> create_s3_bucket >> upload_data >> end)


# Assign the DAG to a variable
dag_instance = etl_to_s3_dag()
