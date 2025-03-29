import pandas as pd
import numpy as np
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss, MAE
from lightning.pytorch import Trainer, Callback
import argparse
import logging
import sys
import json
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch import Trainer
from pathlib import Path
import mlflow


# Set Mlflow Experiment to track model traiing 
mlflow.set_tracking_uri(uri="http://127.0.0.1:8888/")
mlflow.set_experiment("TFT Stock Predictor Training")


FEATURES = ['Date', 'Open', 'High', 'Low', 'Close',
        'Volume', 'Dividends', 'Stock Splits', 'month', 'day', 'day_of_week',
        'NASDAQ', 'SNP', 'DJI', 'RUT', 'VIX', 'XLK', 'XLE', 'XLF', 'XLV', 'RSI',
        'MA_20', 'MA_50', 'MA_200', 'log_return', 'RV_20', 'RV_50', 'symbol',
        'time_idx', 'sentiment']


def load_training_data(training_data):

    df = pd.read_csv(training_data)
    cols_to_convert = ['month', 'day', 'day_of_week']
    df[cols_to_convert] = df[cols_to_convert].astype(str)
    df = df[FEATURES]
    print("Training Dataset Loaded")
    
    return df

def get_hyper_parameter(parameter_path):
    # current_dir = Path(__file__).resolve().parent.parent.parent
    # input_dir = current_dir / 'config'
    # input_path =  input_dir / parameter_path

    # Open and read the JSON file
    with open(parameter_path, 'r') as file:
        best_params = json.load(file)
    return best_params

class MLflowLoggingCallback(Callback):
    def on_validation_end(self, trainer, pl_module):
        val_loss = trainer.callback_metrics.get("val_loss")
        if val_loss is not None:
            mlflow.log_metric("val_loss", val_loss.item(), step=trainer.current_epoch)

def train(train_dataset, params, batch_size=32):

    # Set the best hyper-parameter resulted from Optuna Tuning
    learning_rate=params['learning_rate']
    hidden_size=params['hidden_size']
    attention_head_size=params['attention_head_size']
    dropout=params['dropout']
    hidden_continuous_size=params['hidden_continuous_size']
    gradient_clip_val=params['gradient_clip_val']

    # Define the training dataset
    max_prediction_length = 1   # Predict next 1 day
    max_encoder_length = 10     # Use past 10 days for training

    # Train Validation split 
    cutoff_date = '2025-02-10'
    print(f"cutoff_date: {cutoff_date}")
    train_data = train_dataset[train_dataset["Date"] <= cutoff_date]
    print(f"trainng data: {len(train_data)}")
    val_data = train_dataset[train_dataset["Date"] > cutoff_date]
    print(f"validation data: {len(val_data)}")

    # Convert to PyTorch Forecasting Dataset
    training = TimeSeriesDataSet(
        train_data,
        time_idx="time_idx",
        target="Close",
        group_ids=["symbol"],  
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        static_categoricals=["symbol"],
        time_varying_known_categoricals=['month', 'day', 'day_of_week'],
        time_varying_known_reals=["time_idx"],
        time_varying_unknown_reals=["Close", "Open", "High", "Low", "Volume", "RSI", "sentiment", "MA_20", "MA_50", "MA_200", "log_return", "RV_20", "RV_50", 'NASDAQ', 'SNP', 'DJI', 'RUT', 'VIX', 'XLK', 'XLE', 'XLF', 'XLV', 'RSI'],
        target_normalizer=GroupNormalizer(groups=["symbol"]),
        allow_missing_timesteps=True
    )

    validation = TimeSeriesDataSet.from_dataset(training, val_data, predict=True)

    # Create DataLoaders
    train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=3)
    val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=3)

    # Define the checkpoint callback to save the model
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",  # Directory to save checkpoints
        filename="tft_model-{epoch:02d}-{val_loss:.4f}",  # Save the model with epoch and validation loss
        monitor="val_loss",  # Monitor the validation loss for saving the best model
        mode="min",  # Save the model with the lowest validation loss
        save_top_k=1  # Save only the best model (change to save more if needed)
    )

    # Create the EarlyStopping callback
    early_stopping = EarlyStopping(
        monitor='val_loss',  # Metric to monitor (can also be 'val_accuracy', etc.)
        patience=8,          # Number of epochs with no improvement before stopping
        verbose=True,        # Print messages when stopping
        mode='min',          # 'min' for loss (lower is better), 'max' for accuracy (higher is better)
    )


    trainer = Trainer(
        max_epochs=10,
        limit_train_batches=50,
        accelerator="auto",
        enable_model_summary=True,
        gradient_clip_val=gradient_clip_val,
        callbacks=[checkpoint_callback, early_stopping ,MLflowLoggingCallback()],
    )

    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=learning_rate,
        hidden_size=hidden_size,
        attention_head_size=attention_head_size,
        dropout=dropout,
        hidden_continuous_size=hidden_continuous_size,
        loss=MAE(),
        log_interval=10,  
        optimizer="adamw",
        reduce_on_plateau_patience=3,
    )

    # Start an MLflow run
    with mlflow.start_run():
        # Log the hyperparameters
        mlflow.log_params(params)

        # Start Training
        trainer.fit(
            tft,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Description of what the script does.")
    parser.add_argument('--training', type=str, help='Training Data Path')
    parser.add_argument('--params', type=str, help='Hyperparameters')
    parser.add_argument('--batch', type=int, help='batch size')
    return parser.parse_args()

def main():
    args = parse_args()
    print(args.training, args.params)
    training_path = args.training
    param_path = args.params

    # Load Training Data
    train_data = load_training_data(training_path)

    # Load Hyper Parameters 
    hyper_params = get_hyper_parameter(param_path)

    # Start Training
    train(train_data, hyper_params)

    print("Training Complete")
    return


    

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logging.warning("Script interrupted by user.")
        sys.exit(1)


    