import grpc
from concurrent import futures
import matplotlib.pyplot as plt
import io
import random

from generated import stock_pb2, stock_pb2_grpc


class StockPredictorService(stock_pb2_grpc.StockPredictorServicer):
    def Predict(self, request, context):
        predictions = []

        for ticker in request.tickers:
            price = round(random.uniform(100, 500), 2)  # Fake prediction
            predictions.append(stock_pb2.StockPrediction(
                ticker=ticker, predicted_price=price))

        # Plot
        fig, ax = plt.subplots()
        ax.bar([p.ticker for p in predictions], [
               p.predicted_price for p in predictions])
        ax.set_ylabel('Predicted Price')
        ax.set_title('Stock Predictions')

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        image_bytes = buf.read()
        buf.close()
        plt.close()

        return stock_pb2.PredictResponse(predictions=predictions, plot_image=image_bytes)


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    stock_pb2_grpc.add_StockPredictorServicer_to_server(
        StockPredictorService(), server)
    server.add_insecure_port('[::]:50051')
    print("gRPC server started on port 50051")
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
