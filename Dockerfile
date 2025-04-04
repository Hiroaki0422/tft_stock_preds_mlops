FROM apache/airflow:2.10.5

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all code
COPY . /app

# Set env vars for Airflow
ENV _AIRFLOW_WWW_USER_USERNAME=airflow
ENV _AIRFLOW_WWW_USER_USERNAME=airflow

# Expose MLflow port and Airflow port
EXPOSE 5050 8080

# Default command runs nothing (services started via docker-compose)
CMD ["tail", "-f", "/dev/null"]
