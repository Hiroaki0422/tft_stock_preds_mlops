FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all code
COPY . /app

# Set env vars for Airflow
ENV AIRFLOW_HOME=/app/airflow
RUN airflow db init

# Expose MLflow port and Airflow port
EXPOSE 5050 8080

# Default command runs nothing (services started via docker-compose)
CMD ["tail", "-f", "/dev/null"]
