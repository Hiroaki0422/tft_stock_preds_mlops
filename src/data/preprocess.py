from datetime import datetime
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import os
import argparse
import logging
import sys
from pathlib import Path


def process_input_data():
    path_ls = []
    # Combine newly fetched data
    for dirname, _, filenames in os.walk('/data/raw'):
        for filename in filenames:
            path = os.path.join(dirname, filename)
            print(path)
            path_ls.append(path)

    # Sort by Date
    path_ls = path_ls[1:-1]
    dates = [p[-14:-4] for p in path_ls]
    dates.sort()

    base_path = f"/input_path/{dates[0]}.csv"
    base_df = pd.read_csv(base_path)

    # Create returning dataframe
    for date in dates[1:]:
        path = "/input_path/" + date + ".csv"
        new_df = pd.read_csv(path)
        entries_to_add = new_df[new_df['Date'] > base_df['Date'].max()]
        base_df = pd.concat([base_df, entries_to_add], ignore_index=True)

    return base_df


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


def impute_arima(df, column_name):
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

    return df


def write_output(df):
    current_dir = Path(__file__).resolve().parent.parent.parent
    output_dir = current_dir / 'data'
    output_dir.mkdir(exist_ok=True)

    # Write DataFrame to CSV
    output_path = output_dir / \
        f"{datetime.today().strftime('%Y-%m-%d')}_training.csv"
    df.to_csv(output_path, index=False)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Description of what the script does.")
    parser.add_argument('--data', type=str, help='Historic Data Path')
    return parser.parse_args()


def main():
    args = parse_args()
    new_data = process_input_data()
    historic_data = pd.read_csv(args.data)
    training_dataset = combine_historic_and_new_data(historic_data, new_data)
    write_output(training_dataset)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logging.warning("Script interrupted by user.")
        sys.exit(1)
