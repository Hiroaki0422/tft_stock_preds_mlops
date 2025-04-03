from datetime import datetime
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
import numpy as np
import torch
import os
import pandas as pd
from fastapi import FastAPI, Query
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import matplotlib
matplotlib.use("Agg")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
print("✅ CORS middleware added")


SAVE_DIR = f"saved_plots/{datetime.today().date()}"
os.makedirs(SAVE_DIR, exist_ok=True)

TRAINING_DATA_PATH = f"/app/data/curated/{datetime.today().strftime('%Y-%m-%d')}_training.csv"

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

    return df.dropna()


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

    model_path = "models/model_cpu.ckpt"
    model = TemporalFusionTransformer.load_from_checkpoint(
        model_path,
        map_location=torch.device('cpu')
    ).to(torch.device('cpu'))
    return model, training


MODEL, TRAINING = load_model(TRAINING_DATA)


def get_recent_price_percent_changes(df):

    close = df['Close']
    latest = close.iloc[-1]

    change_1d = ((latest - close.iloc[-2]) /
                 close.iloc[-2]) if len(close) >= 2 else None
    change_3d = ((latest - close.iloc[-4]) /
                 close.iloc[-4]) if len(close) >= 4 else None
    change_5d = ((latest - close.iloc[-6]) /
                 close.iloc[-6]) if len(close) >= 6 else None
    print((change_1d, change_3d, change_5d))
    print(close.iloc[-10:-1])

    return (change_1d, change_3d, change_5d)


def make_prediction(symbol, model=MODEL, training=TRAINING, input_size=2000, output_size=100):
    import matplotlib.pyplot as plt
    # Get Prediction Result
    stock_df = TRAINING_DATA[TRAINING_DATA['symbol'] == symbol]
    predict_df = stock_df.tail(input_size).reset_index()
    one_diff, three_diff, five_diff = get_recent_price_percent_changes(
        predict_df)
    pred_input_df = pd.concat(
        [predict_df, predict_df.tail(1)], ignore_index=True)
    predictions = TimeSeriesDataSet.from_dataset(training, pred_input_df)
    pred_dataloader = predictions.to_dataloader(
        train=False, batch_size=32)
    predictions_q = model.predict(pred_dataloader)
    median_forecast = predictions_q[:, 0]
    print(f"**********************************")
    print(predictions_q)
    print(f"**********************************")

    predicted = predictions_q[-1, 0].item()
    latest = predict_df['Close'].iloc[-1]
    change = (predicted - latest) / latest
    print(f"predicted: {predicted}, latest: {latest}")

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

    return predicted, change*100, one_diff*100, three_diff*100, five_diff*100


@app.get("/predict")
def predict(ticker: str = Query(...)):
    # Mock prediction
    print(f"**********************************")
    print(ticker)
    print(f"**********************************")
    predicted_price, change,  one_diff, three_diff, five_diff = make_prediction(
        ticker)

    return {"ticker": ticker, "change": change, "predicted_price": predicted_price, "one_diff": one_diff, "three_diff": three_diff, "five_diff": five_diff, "image_url": f"/plot/{ticker}"}


@app.get("/plot/{ticker}")
def get_plot(ticker: str):
    file_path = os.path.join(SAVE_DIR, f"{ticker}.png")
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="image/png")
    return {"error": "Plot not found"}, 404
