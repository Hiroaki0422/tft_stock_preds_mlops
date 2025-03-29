# Open Source MLOPs Setup for TFT Stock Predictor

## About This Project 
This project is complete **MLOPs** set up for TFT Stock Predictor model. This project utilized powerful open source tools to operationalize data collection, processing, training, evaluation and monitorning of the model.

## Data Pipelines
<img width="1043" alt="image" src="https://github.com/user-attachments/assets/1f956c83-08a9-403b-abb7-eda995681bd0" />
Pipelines are scheduled by **Apache Airflow** and the DAG greatly help manage pipeline execution by visualize the dependency graph of each task 

- data collection: Fetch Data and financial news daily
- feature engineer: Compute sentiment scores for financial news and compute other useful features
- data preprocess: **DAGs** managed by Airflow will create dependency

## Model Development & Monitoring 
<img width="400" height="400" alt="image" src="https://github.com/user-attachments/assets/182c2845-e83a-4dc8-999a-f15e42f5bd13" /> <br>
**MLflow** is used for tracking hyperparameter tuning, metrics such as validation loss, and versions of models with MLFlow Registry. MLFlow helps experiment tracking significantly easier.
- hyperparameter tuning: powerful open source hyperparameter tuner **Optuna** is used for tuning.
- model training: train model. Validation and metrics are tracked by **MLflow**. The trained model will be registered in **Model Registry** of MLflow
- model monitor: model evaluation pipeline will compute the performance of model and trigger re-training if declining

## Docker 
The application is dockernized and therefore portable. It can be deployed Kubernetes or other cloud cluster for more scalable and intense computation such as model training.
The container application can easily be launched by docker compose up command

## CICD for ML
**Github Actions**(workflow) is implemented for continuous integration of ML development. Automatic tests and data validation will be performed upon push to main branch to make sure the latest release is bug-free




  
