# MLOps-Enabled Full Stack ML Pipeline with Real-Time Inference and Workflow Automation

<img width="834" alt="architecture" src="https://github.com/user-attachments/assets/b5af2db8-4b18-4057-ac3e-69ed09a3d1f3" />

## 🚀 About This Project  
This application fully operationalizes a **Temporal Fusion Transformer (TFT)** stock forecasting model and implements key **MLOps components**, including:

- Experiment tracking and model evaluation  
- Deployment with real-time predictions  
- Automated data collection, processing, and validation  
- Containerization and CI/CD pipelines  

---

## 🔁 Automated Data Collection, Processing, and Validation

<img width="1043" alt="airflow-pipeline" src="https://github.com/user-attachments/assets/1f956c83-08a9-403b-abb7-eda995681bd0" />

Data pipelines are scheduled and managed with **Apache Airflow**, which visualizes task dependencies as DAGs and provides a UI for monitoring and debugging.

Pipeline tasks include:

- **Data collection**: Daily fetch of stock price data and financial news  
- **Feature engineering**: Computing sentiment scores from news and generating additional features  
- **Preprocessing**: Appending the latest data to the model’s training dataset  
- **Validation**: Ensuring data integrity and checking for nulls or anomalies  

---

## 🧠 Model Development & Monitoring

<img width="400" height="400" alt="mlflow" src="https://github.com/user-attachments/assets/182c2845-e83a-4dc8-999a-f15e42f5bd13" />

Model development is tracked and monitored using **MLflow**:

- **Hyperparameter tuning**: Handled by **Optuna** for efficient search  
- **Training**: Models are trained and evaluated; key metrics like validation loss are logged via MLflow  
- **Model registry**: Trained models are versioned and stored using MLflow’s model registry  
- **Monitoring**: A separate evaluation pipeline runs weekly to assess model performance; low scores can trigger re-training  

---

## ⚙️ Model Serving & Real-Time Inference

The best-performing model is deployed using **FastAPI** to handle real-time predictions. When users interact with the frontend:

- Input is sent to the backend via API  
- Predictions are made in real time  
- Visualizations are generated and sent back to be rendered in the UI  

---

## 📦 Containerization

The entire stack is containerized using **Docker** for portability and ease of deployment.

- Ready for cloud deployment (e.g., to a Kubernetes cluster)  
- A `docker-compose.yml` is included for local testing and development  
- Run everything with a single command:

```bash
docker-compose up
