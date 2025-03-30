from fastapi import FastAPI, Query
from fastapi.responses import FileResponse
import matplotlib.pyplot as plt
import random
import os

app = FastAPI()

SAVE_DIR = "saved_plots"
os.makedirs(SAVE_DIR, exist_ok=True)


@app.get("/predict")
def predict(ticker: str = Query(...)):
    # Mock prediction
    predicted_price = round(random.uniform(100, 500), 2)

    # Generate a plot
    fig, ax = plt.subplots()
    ax.bar([ticker], [predicted_price])
    ax.set_title(f"Predicted Price for {ticker}")
    ax.set_ylabel("Price ($)")
    file_path = os.path.join(SAVE_DIR, f"{ticker}.png")
    plt.savefig(file_path)
    plt.close()

    return {"ticker": ticker, "predicted_price": predicted_price, "image_url": f"/plot/{ticker}"}


@app.get("/plot/{ticker}")
def get_plot(ticker: str):
    file_path = os.path.join(SAVE_DIR, f"{ticker}.png")
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="image/png")
    return {"error": "Plot not found"}, 404
