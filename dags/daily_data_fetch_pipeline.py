from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from datetime import datetime


def data_validation():
    print("TBD")
    return True


with DAG(
    dag_id="stock_data_pipeline_script_fetch",
    start_date=datetime(2025, 3, 28),
    schedule_interval="@daily",  # or None for manual runs
    catchup=False,
    tags=["stocks_sentiment"],
) as dag:

    fetch_data = BashOperator(
        task_id="fetch_data",
        bash_command="python /opt/airflow/src/data/fetch_data.py"
    )

    validate_data = PythonOperator(
        task_id="process_data",
        python_callable=data_validation
    )

    fetch_data >> validate_data
