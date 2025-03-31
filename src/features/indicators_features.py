
import yfinance as yf
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime


# top 50-100 publicly traded companies
SP100_Tickers = [
    'AAPL', 'ABBV', 'ABT', 'ACN', 'ADBE', 'AIG', 'AMD', 'AMGN', 'AMT', 'AMZN',
    'AVGO', 'AXP', 'BA', 'BAC', 'BK', 'BKNG', 'BLK', 'BRK.B', 'C',
    'CAT', 'CHTR', 'CL', 'CMCSA', 'COF', 'COP', 'COST', 'CRM', 'CSCO', 'CVS',
    'CVX', 'DE', 'DHR', 'DIS', 'DOW', 'DUK', 'EMR', 'F', 'FDX', 'GD', 'GE',
    'GILD', 'GM', 'GOOG', 'GOOGL', 'GS', 'HD', 'HON', 'IBM', 'INTC', 'INTU',
    'JNJ', 'JPM', 'KHC', 'KO', 'LIN', 'LLY', 'LMT', 'LOW', 'MA', 'MCD',
    'MDLZ', 'MDT', 'MET', 'META', 'MMM', 'MO', 'MRK', 'MS', 'MSFT', 'NEE',
    'NFLX', 'NKE', 'NVDA', 'ORCL', 'PEP', 'PFE', 'PG', 'PM', 'PYPL', 'QCOM',
    'RTX', 'SBUX', 'SCHW', 'SO', 'SPG', 'T', 'TGT', 'TMO', 'TMUS', 'TSLA',
    'TXN', 'UNH', 'UNP', 'UPS', 'USB', 'V', 'VZ', 'WFC', 'WMT', 'XOM'
]

# Indexes helpful to predict stock, key is API symbol for each index
INDICATORS = {"^IXIC": "NASDAQ",
              "^GSPC": "SNP",
              "^DJI": "DJI",
              "^RUT": "RUT",
              "^VIX": "VIX",
              "XLK": "XLK",
              "XLE": "XLE",
              "XLF": "XLF",
              "XLV": "XLV"
              }

# ================================
# 📌 Get Indicator Data
# ================================


def get_indicators_data(indicators=INDICATORS):
    # Create dataframe of important stock index
    indicator_df = None
    for symbol in indicators:
        ticker = yf.Ticker(symbol)
        stock_df = ticker.history(period='10y', interval='1d').reset_index()
        stock_df.head()
        stock_df = stock_df.rename(columns={
            'Open': indicators[symbol]
        })
        stock_df = stock_df[['Date', indicators[symbol]]]
        stock_df['Date'] = stock_df['Date'].dt.date

        if indicator_df is None:
            indicator_df = stock_df.copy()
            continue
        indicator_df = indicator_df.merge(stock_df, how='inner', on=['Date'])
    return indicator_df


# ================================
# 📌 Load and Process Stock Data
# ================================
def load_stock_data(ticker_symbol, indicators):
    print(f"loading {ticker_symbol}")
    ticker = yf.Ticker(ticker_symbol)
    stock_df = ticker.history(period='10y', interval='1d').reset_index()
    try:
        stock_df['month'] = stock_df['Date'].dt.month
        stock_df['day'] = stock_df['Date'].dt.day
        stock_df['day_of_week'] = stock_df['Date'].dt.dayofweek
        stock_df['Date'] = stock_df['Date'].dt.date
        stock_df = stock_df.merge(indicators, how='left', on=['Date'])
    except Exception as e:
        print(e)
        stock_df = None
    return stock_df

# ================================
# 📌 Load and Process Sentiment Data
# ================================


def load_sentiment_data(file_path):
    """Loads and processes sentiment data, aggregates sentiment scores by date."""
    sentiment_map = {'positive': 1, 'neutral': 0, 'negative': -1}

    df = pd.read_csv(file_path)
    df['sentiment_mapped'] = df['sentiment'].map(sentiment_map)
    # Convert to date (no time)
    df['Date'] = pd.to_datetime(df['Date']).dt.date

    # Aggregate sentiment scores by date
    sentiment_sum = df.groupby('Date', as_index=False)[
        'sentiment_mapped'].sum()
    sentiment_count = df.groupby('Date', as_index=False)[
        'sentiment_mapped'].count()
    sentiment_sum['sentiment'] = sentiment_sum['sentiment_mapped'] / \
        sentiment_count['sentiment_mapped']

    return sentiment_sum[['Date', 'sentiment']]

# ================================
# 📌 Compute Technical Indicators
# ================================


def calculate_technical_indicators(df):
    """Computes RSI, Moving Averages, Log Returns, and Realized Volatility."""
    def calculate_rsi(data, column='Close', period=14):
        delta = data[column].diff()
        gains = delta.where(delta > 0, 0)
        losses = -delta.where(delta < 0, 0)
        avg_gain = gains.rolling(window=period, min_periods=1).mean()
        avg_loss = losses.rolling(window=period, min_periods=1).mean()
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    df['RSI'] = calculate_rsi(df, column='Close')

    # Moving Averages
    for ma in [20, 50, 200]:
        df[f'MA_{ma}'] = df['Close'].rolling(window=ma).mean()

    # Log Returns
    df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))

    # Realized Volatility
    for rv in [20, 50]:
        df[f'RV_{rv}'] = df['log_return'].rolling(
            window=rv).std() * np.sqrt(252)

    return df  # Drop NaN values from rolling calculations

# ================================
# 📌 Create time step
# ================================


def add_time_step_from_date(df, date_column='Date', step_column='time_idx'):
    df[date_column] = pd.to_datetime(df[date_column])
    unique_dates = pd.Series(df[date_column].sort_values().unique())
    date_to_step = {date: i for i, date in enumerate(unique_dates)}
    df[step_column] = df[date_column].map(date_to_step)
    return df

# ================================
# 📌 Main Processing for Multiple Stocks
# ===============================


def feature_engineer():
    indicators_df = get_indicators_data()

    merged_dfs = []

    for stock in SP100_Tickers:
        path = "/app/data/processed/sentiment/archive/" + stock + ".csv"
        path2 = "/app/data/processed/sentiment/archive2/" + stock + ".csv"
        try:
            # Load sentiment and stock data
            sentiments_df1 = load_sentiment_data(path)
        except Exception as e:
            print(f"ERROR: {stock}, {e}")
            sentiments_df1 = None
        try:
            # Load sentiment and stock data
            sentiments_df2 = load_sentiment_data(path2)
        except Exception as e:
            print(f"ERROR: {stock}, {e}")
            sentiments_df2 = None

        if sentiments_df1 is not None and sentiments_df2 is None:
            sentiments_df = sentiments_df1
        elif sentiments_df1 is None and sentiments_df2 is not None:
            sentiments_df = sentiments_df2
        elif sentiments_df1 is None and sentiments_df2 is None:
            continue
        else:
            sentiments_df = pd.concat([sentiments_df1, sentiments_df2])
        stock_df = load_stock_data(stock, indicators_df)

        if stock_df is None:
            continue

        # Merge sentiment and stock data
        merged_df = pd.merge(stock_df, sentiments_df, on='Date', suffixes=(
            '_yt', '_sentiments'), how='left')
        merged_df = calculate_technical_indicators(merged_df)

        # Encode Stock Symbol
        merged_df['symbol'] = stock
        merged_dfs.append(merged_df)

    # Merge all stock data
    merged_df = pd.concat(merged_dfs, ignore_index=True)
    merged_df = add_time_step_from_date(merged_df)
    return merged_df


def write_output(df):
    current_dir = Path(__file__).resolve().parent.parent.parent
    output_dir = current_dir / 'data'
    output_dir.mkdir(exist_ok=True)

    # Write DataFrame to CSV
    output_path = output_dir / \
        f"/app/data/processed/{datetime.today().strftime('%Y-%m-%d')}_historical.csv"
    df.to_csv(output_path, index=False)


def main():
    df = feature_engineer()
    write_output(df)
    print(df.sample(3))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Keyboard Interrupt")
