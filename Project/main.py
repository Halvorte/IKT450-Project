# Import
import pandas as pd
from IPython.display import display
import os
import matplotlib.pyplot as plt
import time
import csv

# from pandas.table.plotting import table


# Function for sliding a window over dataframe to select window to train model on
def sliding_window(dataframe, window_size):
    # Loop to move the window by 1
    # start = 1
    # to = window_size + 1
    start = 0
    to = window_size

    for i in range(len(dataframe) - window_size):
        # Define the window
        window = dataframe.iloc[start:to]
        display(window)

        # Make prediction on window after frame
        target = dataframe.iloc[to]
        display(target)

        # Move the window 1 position down the dataframe
        start += 1
        to += 1

    return window


def f(x, y):
    if y - x > 0:
        return 1
    else:
        return 0


def calulateOpenClose(df):
    for item, value in df.items():
        result = [f(x, y) for x, y in zip(value["Open"], value["Close"])]
        value["closeMinusOpen"] = result
    return df


def createDataframes(files):
    dfs = {f'{os.path.basename(file)}_df': pd.read_csv(file, usecols=["Date", "Open", "Close"]) for file in files}
    return dfs


def addCompundScores(df):
    cs_df = pd.read_csv('compound_scores.csv')
    x = {k: v for k, v in cs_df.groupby("stock")}
    y = {}

    for item, value in df.items():
        stock_name = item.split('.')
        stock_name = stock_name[0]
        for item2, value2 in x.items():
            if item2 == stock_name:
                merged_df = value.merge(value2[['Date', 'compound score']], how='inner', on='Date')
                y[stock_name] = merged_df

    return y


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print('Fetching dataset')
    st = time.time()
    path = os.path.join(os.getcwd(), 'stock_data', 'price', 'raw')
    files = [os.path.join(path, i) for i in os.listdir(path) if os.path.isfile(os.path.join(path, i))]
    dictOfDfs = createDataframes(files)
    dictOfDfs = calulateOpenClose(dictOfDfs)

    complete_dfs = addCompundScores(dictOfDfs)


    et = time.time()
    print('Dataset ready in ' + str(et - st) + 's')

    # stock_aapl = pd.read_csv('stock_data/raw/AAPL.csv', usecols=["Date", "Open", "Close"])
    # stock_amzn = pd.read_csv('stock_data/raw/AMZN.csv')
    # stock_orcl = pd.read_csv('stock_data/ORCL.csv')
    # display(stock_aapl)
    #
    # stock_aapl["closeMinusOpen"] = calulateOpenClose(stock_aapl)
    # print(stock_aapl)
