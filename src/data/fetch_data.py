from transformers import pipeline
from collections import defaultdict
import yfinance as yf
import pandas as pd
import warnings
from datetime import datetime
import argparse
import logging
import sys
from pathlib import Path

warnings.simplefilter("ignore", RuntimeWarning)

# top 100 companies in S&P
SP100_TICKERS = [
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

# Use finbert model to create sentiment analysis pipeline for news related each stock 
sentiment_pipeline = pipeline("text-classification", model="ProsusAI/finbert")
SENTIMENT_MAP = {'positive': 1, 'neutral':0, 'negative':-1}

def get_sentiment(text):
    result = sentiment_pipeline(text)[0]  # Get first result
    
    return SENTIMENT_MAP[result['label']]  

def get_stock_and_sentiment():
    # Collect Stock Data & Related News
    stock_sent_dfs  = []
    for symbol in SP100_TICKERS[0:5]:
        print(symbol)
        dat = yf.Ticker(symbol)
        news = dat.get_news()
        if not news:
            continue
            
        # Get the result of sentiment analysis of the news 
        stock_sent_dict = defaultdict(list)
        for n in news:
            stock_sent_dict[pd.to_datetime(n['content']['pubDate']).strftime('%Y-%m-%d')].append(get_sentiment(n['content']['title']))
        data = [{'Date':pd.to_datetime(k), 'sentiment': sum(stock_sent_dict[k]) / len(stock_sent_dict[k])} for k in stock_sent_dict]
        stock_sent_df = pd.DataFrame(data)
        
        # Get recent stock prices
        ticker = yf.Ticker(symbol)
        stock_df = ticker.history(period='20d', interval='1d').reset_index()
        stock_df['Symbol'] = symbol
        stock_df['Date'] = pd.to_datetime(stock_df['Date'].dt.strftime('%Y-%m-%d'))
        stock_sent_df = pd.merge(stock_df, stock_sent_df, on='Date', suffixes=('_yt', '_sentiments'), how='left')
        stock_sent_dfs.append(stock_sent_df)

    ss_processed_df = pd.concat(stock_sent_dfs, ignore_index=True)
    len(f"retrieved_rows: {ss_processed_df}")
    return ss_processed_df

def write_output(df):
    current_dir = Path(__file__).resolve().parent.parent.parent
    output_dir = current_dir / 'data'
    output_dir.mkdir(exist_ok=True)  

    # Write DataFrame to CSV
    output_path = output_dir / f"{datetime.today().strftime('%Y-%m-%d')}_stock&sentiments.csv"
    df.to_csv(output_path, index=False)


def parse_args():
    parser = argparse.ArgumentParser(description="Description of what the script does.")
    parser.add_argument('--example', type=str, help='Example argument')
    return parser.parse_args()

def main():
    args = parse_args()

    # Fetch Data
    new_data = get_stock_and_sentiment()
    write_output(new_data)





if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logging.warning("Script interrupted by user.")
        sys.exit(1)