FROM apache/airflow:2.8.2-python3.9

USER root

RUN apt-get update && \
    apt-get -y install git && \
    apt-get clean

COPY ./requirements.txt .

USER airflow
RUN pip install --no-cache-dir -r requirements.txt

USER root
COPY start.sh /start.sh
RUN chmod +x /start.sh

USER airflow
ENTRYPOINT ["/bin/bash","/start.sh"]