"""
Module to run the web application.
"""

import logging
import os
from typing import Any, List

import mlflow
from dotenv import load_dotenv
from flask import Flask, Response, render_template, request

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load configuration from environment variables
# MODEL_PATH = os.getenv("MODEL_PATH", "./catboost-model")
MODEL_S3_PATH = os.getenv("MODEL_S3_PATH", "s3://model-bucket/model")
PORT = int(os.getenv("PORT", 5000))

try:
    loaded_model = mlflow.pyfunc.load_model(MODEL_S3_PATH)
    logger.info("Model loaded successfully from Minio S3")
except Exception as e:
    logger.error(f"Failed to load model from {MODEL_S3_PATH}: {e}")
    raise

app = Flask(__name__)


@app.route("/")
def entry_page() -> str:
    """
    Homepage: Serve the visualization page.
    """
    return render_template("index.html")


@app.route("/predict", methods=["GET", "POST"])
def render_message() -> Response:
    """
    Webpage for prediction.
    """
    try:
        # Get data from the form
        form_data = {
            "age": int(request.form["age"]),
            "job": request.form["job"],
            "marital": request.form["marital"],
            "education": request.form["education"],
            "default": request.form["default"],
            "balance": int(request.form["balance"]),
            "housing": request.form["housing"],
            "loan": request.form["loan"],
            "contact": request.form["contact"],
            "day": int(request.form["day"]),
            "month": request.form["month"],
            "duration": int(request.form["duration"]),
            "campaign": int(request.form["campaign"]),
            "pdays": int(request.form["pdays"]),
            "previous": int(request.form["previous"]),
            "poutcome": request.form["poutcome"],
        }

        # Prepare data for prediction
        data: List[List[Any]] = [list(form_data.values())]

        # Make prediction
        preds = loaded_model.predict(data)[0]
        logger.info("Prediction executed successfully")

        # Prepare the result message
        message = f"Has the client subscribed a term deposit? ==> {preds}!"
    except KeyError as e:
        message = f"Missing form field: {e}"
        logger.error(message)
    except ValueError as e:
        message = f"Invalid input value: {e}"
        logger.error(message)
    except Exception as e:
        message = f"Error encountered: {e}"
        logger.error(message)

    # Return the result to the web page
    return render_template("index.html", message=message)


if __name__ == "__main__":
    try:
        logger.info(f"Starting Flask app on port {PORT}")
        app.run(host="0.0.0.0", port=PORT)
    except Exception as e:
        logger.error(f"Failed to start Flask app: {e}")
        raise
