# Import
import pandas as pd
from IPython.display import display
import matplotlib.pyplot as plt
#from pandas.table.plotting import table


# Function for sliding a window over dataframe to select window to train model on
def sliding_window(dataframe, window_size):
    # Loop to move the window by 1
    #start = 1
    #to = window_size + 1
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



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Import the data
    stock_df = pd.read_csv('stock_data/AAPL.csv')
    stock_amzn = pd.read_csv('stock_data/AMZN.csv')
    stock_orcl = pd.read_csv('stock_data/ORCL.csv')
    display(stock_df)

    yo = sliding_window(stock_df, 5)
