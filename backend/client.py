import grpc
from generated import stock_pb2, stock_pb2_grpc


def save_plot(image_bytes, filename="prediction.png"):
    with open(filename, "wb") as f:
        f.write(image_bytes)
    print(f"Saved plot to {filename}")


def run():
    channel = grpc.insecure_channel('localhost:50051')
    stub = stock_pb2_grpc.StockPredictorStub(channel)

    response = stub.Predict(stock_pb2.PredictRequest(
        tickers=["AAPL", "GOOGL", "MSFT"]))
    for p in response.predictions:
        print(f"{p.ticker}: ${p.predicted_price}")

    save_plot(response.plot_image)


if __name__ == "__main__":
    run()
