from fastapi import FastAPI, Query
from fastapi.responses import FileResponse
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from datetime import datetime

app = FastAPI()

SAVE_DIR = f"/data/saved_plots/{datetime.today()}"
os.makedirs(SAVE_DIR, exist_ok=True)
TRAINING_DATA_PATH = ''

FEATURES = ['Date', 'Open', 'High', 'Low', 'Close',
            'Volume', 'Dividends', 'Stock Splits', 'month', 'day', 'day_of_week',
            'NASDAQ', 'SNP', 'DJI', 'RUT', 'VIX', 'XLK', 'XLE', 'XLF', 'XLV', 'RSI',
            'MA_20', 'MA_50', 'MA_200', 'log_return', 'RV_20', 'RV_50', 'symbol',
            'time_idx', 'sentiment']


def load_training_data(training_data_path=TRAINING_DATA_PATH):

    df = pd.read_csv(training_data_path)
    cols_to_convert = ['month', 'day', 'day_of_week']
    df[cols_to_convert] = df[cols_to_convert].astype(str)
    df = df[FEATURES]
    print("Training Dataset Loaded")

    return df


TRAINING_DATA = load_training_data()


def load_model(train_data):
    # Convert to PyTorch Forecasting Dataset
    # 1. Define the training dataset
    max_prediction_length = 1
    max_encoder_length = 10

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
        time_varying_unknown_reals=["Close", "Open", "High", "Low", "Volume", "RSI", "sentiment",
                                    "MA_20", "MA_50", "MA_200", "log_return", "RV_20", "RV_50",
                                    'NASDAQ', 'SNP', 'DJI', 'RUT', 'VIX',
                                    'XLK', 'XLE', 'XLF', 'XLV', 'RSI'],
        target_normalizer=GroupNormalizer(groups=["symbol"]),
        allow_missing_timesteps=True
    )

    model_path = "models/tft_model-epoch=09-val_loss=2.5493.ckpt"
    model = TemporalFusionTransformer.load_from_checkpoint(model_path)
    return model, training


MODEL, TRAINING = load_model(TRAINING_DATA)


def make_prediction(symbol, model=MODEL, training=TRAINING, input_size=2000, output_size=100):
    # Get Prediction Result
    stock_df = training[training['symbol'] == symbol]
    predict_df = stock_df.tail(input_size).reset_index()
    pred_input_df = pd.concat(
        [predict_df, predict_df.tail(1)], ignore_index=True)
    predictions = TimeSeriesDataSet.from_dataset(training, pred_input_df)
    pred_dataloader = predictions.to_dataloader(
        train=False, batch_size=32)
    predictions_q = model.predict(pred_dataloader)
    median_forecast = predictions_q[:, 1]

    # Plotting
    pred_size = output_size
    plt.plot(predict_df['Date'][-pred_size-1:].to_list() + ['next_day'], median_forecast.cpu()
             [-pred_size-2:], label="Predicted", color="blue", linewidth=2, linestyle="dashed")
    # plt.plot(low_forecast.cpu(), label="Lower Bound (5%)", color="red", linestyle="dashed")
    # plt.plot(high_forecast.cpu(), label="Upper Bound (95%)", color="green", linestyle="dashed")
    plt.plot(predict_df['Date'][-pred_size-1:], predict_df['Close']
             [-pred_size-1:], label="Actual", color="black")

    # # Labels and title
    plt.xlabel("Date")
    plt.xticks(fontsize=8)
    plt.xticks(np.arange(0, len(predict_df['Date'][-pred_size-1:]), 30))
    plt.ylabel("Predicted Value")
    plt.title(f"{symbol} Forecast")
    plt.legend()
    plt.grid(True)

    # plt.save()
    file_path = os.path.join(SAVE_DIR, f"{symbol}.png")
    plt.savefig(file_path)
    plt.close()

    return predictions_q[-1, 0]


@app.get("/predict")
def predict(ticker: str = Query(...)):
    # Mock prediction
    predicted_price = make_prediction(ticker)

    return {"ticker": ticker, "predicted_price": predicted_price, "image_url": f"/plot/{ticker}"}


@app.get("/plot/{ticker}")
def get_plot(ticker: str):
    file_path = os.path.join(SAVE_DIR, f"{ticker}.png")
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="image/png")
    return {"error": "Plot not found"}, 404
