FROM python:3.9-slim

# Allow statements and log messages to immediately appear in the Knative logs
ENV PYTHONUNBUFFERED True

# Copy local code to the container image.
ENV APP_HOME /app
WORKDIR $APP_HOME
COPY . ./

# Install production dependencies.
RUN pip install -r requirements.txt
RUN pip install gunicorn

EXPOSE 5000

# Run the web service on container startup using gunicorn

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "src.api.app:app"]