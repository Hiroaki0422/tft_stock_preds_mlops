from datetime import datetime
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import os
import argparse
import logging
import sys
from pathlib import Path

HISTORICAL_DATA_PATH = f"/app/data/processed/{datetime.today().strftime('%Y-%m-%d')}_historical.csv"


def process_input_data():
    path_ls = []
    # Combine newly fetched data
    for dirname, _, filenames in os.walk('/app/data/raw/'):
        for filename in filenames:
            path = os.path.join(dirname, filename)
            print(path)
            if '202' in path:
                path_ls.append(path)
    print(path_ls)

    # Sort by Date
    dates = [p[14:24] for p in path_ls]
    dates.sort()
    print(dates)

    base_path = f"/app/data/raw/{dates[0]}_stock_sentiments.csv"
    base_df = pd.read_csv(base_path)

    # Create returning dataframe
    for date in dates[1:]:
        path = "/app/data/raw/" + date + "_stock_sentiments.csv"
        new_df = pd.read_csv(path)
        entries_to_add = new_df[new_df['Date'] > base_df['Date'].max()]
        base_df = pd.concat([base_df, entries_to_add], ignore_index=True)

    base_df = base_df.rename(columns={"Symbol": "symbol"})
    base_df = base_df[['Date', 'symbol', 'sentiment']]

    print(base_df.columns)

    return base_df.dropna()


def combine_historic_and_new_data(historic_df, new_df):
    # Merge A with B on Date and symbol
    merged = pd.merge(
        historic_df,
        new_df,
        on=['Date', 'symbol'],
        how='left',
        suffixes=('_A', '_B')
    )

    # Choose sentiment: take sentiment_A unless it's missing, then use sentiment_B
    merged['sentiment'] = merged['sentiment_A'].combine_first(
        merged['sentiment_B'])
    return merged

# Train an ARIMA model to predict missing values


def impute_arima(df, column_name='sentiment'):
    # Make sure your column does not have NaNs initially
    series = df[column_name]

    # Fit ARIMA model to the non-missing data
    model = ARIMA(series, order=(5, 1, 0))  # You can experiment with the order
    model_fit = model.fit()

    # Predict the missing values using the fitted ARIMA model
    predictions = model_fit.predict(start=0, end=len(series)-1, typ='levels')

    # Fill NaN values with predicted values
    df[column_name].fillna(
        pd.Series(predictions, index=df.index), inplace=True)

    df.drop(columns=['sentiment_A', 'sentiment_B'])

    return df


def write_output(df):
    df.to_csv(
        f"/app/data/curated/{datetime.today().strftime('%Y-%m-%d')}_training.csv", index=False)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Description of what the script does.")
    parser.add_argument('--data', type=str, help='Historic Data Path')
    return parser.parse_args()


def main():
    args = parse_args()
    new_data = process_input_data()
    historic_data = pd.read_csv(HISTORICAL_DATA_PATH)
    training_dataset = combine_historic_and_new_data(historic_data, new_data)
    training_dataset = impute_arima(training_dataset)
    write_output(training_dataset)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logging.warning("Script interrupted by user.")
        sys.exit(1)
