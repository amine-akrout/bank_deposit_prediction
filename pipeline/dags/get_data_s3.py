import io
import logging
import os

import boto3
import boto3.exceptions
import pandas as pd
from airflow.decorators import dag, task
from airflow.utils.dates import days_ago
from ucimlrepo import fetch_ucirepo

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s: [%(levelname)s]: %(message)s"
)

# Constants
BUCKET_NAME = "data-bucket"
OBJECT_NAME = "data/bank.parquet"


# Define the DAG
@dag(
    schedule_interval="@daily",
    start_date=days_ago(1),
    catchup=False,
    tags=["etl"],
    default_args={"owner": "airflow", "retries": 1},
)
def etl_to_s3_dag():
    @task
    def fetch_data():
        """Fetch data using the UCIML repository."""
        logging.info("Fetching data from the UCIML repository.")
        return fetch_ucirepo(id=222)

    @task
    def prepare_data(bank_marketing):
        """Extract features and targets, and combine them into a DataFrame."""
        logging.info("Preparing the data for upload.")
        features = bank_marketing.data.features
        targets = bank_marketing.data.targets
        return pd.concat([features, targets], axis=1)

    @task
    def create_s3_session():
        """Create an S3 session using credentials from environment variables."""
        logging.info("Creating an S3 session.")
        ACCESS_KEY = os.environ.get("S3_ACCESS_KEY")
        SECRET_KEY = os.environ.get("S3_SECRET_KEY")
        return boto3.Session(
            aws_access_key_id=ACCESS_KEY,
            aws_secret_access_key=SECRET_KEY,
            region_name="us-east-1",
        )

    @task
    def get_s3_client(session):
        """Get an S3 client from the given session."""
        logging.info("Creating an S3 client.")
        endpoint_url = os.environ.get("S3_ENDPOINT_URL", "http://localhost:9000")
        logging.info(f"Using S3 endpoint URL: {endpoint_url}")
        return session.client("s3", endpoint_url=endpoint_url)

    @task
    def create_bucket_if_not_exists(s3_client):
        """Create an S3 bucket if it does not already exist, with improved error handling and feedback."""
        logging.info(f"Creating bucket '{BUCKET_NAME}' if it does not exist.")
        try:
            # Check if the bucket already exists
            if not s3_client.head_bucket(Bucket=BUCKET_NAME):
                s3_client.create_bucket(Bucket=BUCKET_NAME)
                logging.info(f"Bucket '{BUCKET_NAME}' created.")
            else:
                logging.info(f"Bucket '{BUCKET_NAME}' already exists.")
        except s3_client.exceptions.NoSuchBucket:
            s3_client.create_bucket(Bucket=BUCKET_NAME)
            logging.info(f"Bucket '{BUCKET_NAME}' created.")
        except s3_client.exceptions.ClientError as e:
            if e.response["Error"]["Code"] == "404":
                s3_client.create_bucket(Bucket=BUCKET_NAME)
                logging.info(f"Bucket '{BUCKET_NAME}' created.")
            else:
                logging.error(f"Failed to create bucket '{BUCKET_NAME}': {e}")
        except Exception as e:
            logging.error(f"An unexpected error occurred: {e}")

    @task
    def upload_data_to_s3(s3_client, data):
        """Upload data to an S3 bucket with enhanced exception handling."""
        buffer = io.BytesIO()
        try:
            data.to_parquet(buffer, index=False)
            buffer.seek(0)
            s3_client.upload_fileobj(buffer, BUCKET_NAME, OBJECT_NAME)
            print("Data saved in S3 bucket.")
        except boto3.exceptions.S3UploadFailedError as e:
            print(f"Failed to upload data to S3: {e}")
            # Handle S3-specific upload failure, e.g., permissions, connectivity
        except IOError as e:
            print(f"IOError while handling the data buffer: {e}")
            # Handle potential issues with reading or writing to the buffer
        except Exception as e:
            print(f"An unexpected error occurred during the upload: {e}")

    # Define Task Flow

    data = fetch_data()
    prepared_data = prepare_data(data)
    session = create_s3_session()
    s3_client = get_s3_client(session)
    create_bucket_if_not_exists(s3_client)
    upload_data_to_s3(s3_client, prepared_data)


# Assign the DAG to a variable
dag_instance = etl_to_s3_dag()
