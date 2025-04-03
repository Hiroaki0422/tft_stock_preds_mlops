# MLOps-Enabled Full Stack ML Pipeline with Real-Time Inference and Workflow Automation
<img width="834" alt="image" src="https://github.com/user-attachments/assets/b5af2db8-4b18-4057-ac3e-69ed09a3d1f3" />


## About This Project 
This Application has fully operationalized the TFT stock Forecast model from my repo and implemented MLOps, such as experinment tracking and model evaluations, deployment and real-time prediction, automated data collection and validation, and finally containernization and CICD pipeline.

## Automated Data Collection, Processing and Validation
<img width="1043" alt="image" src="https://github.com/user-attachments/assets/1f956c83-08a9-403b-abb7-eda995681bd0" />
<br>
Data processing pipelines are scheduled and managed by **Apache-Airflow**. Apache airflow visualize the entire data pipelines and its dependencies as DAG, and the UI they provide you greatly help you with debugging your data pipeline and locate where exactly error comes from. 

Tasks from above pipeline:
- data collection: fetch data daily and financial news daily
- feature engineer: compute sentiment scores for financial news and engineer other useful feature 
- data preprocess: process data to add latest data into training dataset of the model
- data validation: validate that the data is successfully processed without large number of null values present

## Model Development & Monitoring 
<img width="400" height="400" alt="image" src="https://github.com/user-attachments/assets/182c2845-e83a-4dc8-999a-f15e42f5bd13" /> <br>
**MLflow** is used for tracking hyperparameter tuning, metrics such as validation loss, and versions of models with MLFlow Registry. MLFlow helps experiment tracking significantly easier.
- hyperparameter tuning: powerful open source hyperparameter tuner **Optuna** is used for tuning.
- model training: train model. Validation and metrics are tracked by **MLflow**. The trained model will be registered in **Model Registry** of MLflow
- model monitor: model evaluation pipeline will compute the performance of model weekly and those metrics are managed by MLflow. Low score meaning re-training

## Model Serving / Deployment
The best trained model will be deployed through FastAPI. Model will compute the prediction in real time based on user's input from the frontend. At the same, backend will create visualizations and send back to frontend to display.

## Docker & Container
The application is containerization and made to be portable. It is ready to be deployed to cloud clusters for production. Docker compose file is also provided so any person with docker can easily run the entire application with the simple command: docker-compose up 

## CICD for ML
**Github Actions**(workflow) is implemented for continuous integration of ML development. Automatic tests and data validation will be performed upon push to main branch to make sure the latest release is bug-free

## How to run
docker-compose up




  
