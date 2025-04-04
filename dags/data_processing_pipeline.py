from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from datetime import datetime


def data_validation():
    print("TBD")
    return True


with DAG(
    dag_id="stock_data_pipeline_process_input",
    start_date=datetime(2025, 4, 3),
    schedule_interval="0 15 * * 1-5",
    catchup=False,
    tags=["stocks_sentiment"],
) as dag:

    fetch_data = BashOperator(
        task_id="fetch_data",
        bash_command="python /opt/airflow/src/data/fetch_data.py"
    )

    feature_engineer = BashOperator(
        task_id="feature_engineer",
        bash_command="python /opt/airflow/src/features/indicators_features.py"
    )

    create_training = BashOperator(
        task_id="create_training_dataset",
        bash_command="python /opt/airflow/src/data/preprocess.py"
    )

    validate_data = PythonOperator(
        task_id="validate_data",
        python_callable=data_validation
    )

    fetch_data >> feature_engineer >> create_training >> validate_data
