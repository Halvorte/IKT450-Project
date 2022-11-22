import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display
import torch
import torch.nn as nn
from torch.autograd import Variable
import time


# Funciton to read and claen the data
def data():
    df = pd.read_csv('Complete_df.csv')

    # Remove unnecessary columns
    df = df.drop(columns=['level_0', 'level_1'])

    # Split up to the different stocks
    all_stocks = df['Stock'].unique().tolist()
    print(all_stocks)

    # All stocks stored in dictionary
    stocks_df = {}
    for i in all_stocks:
        stocks_df[f'{i}'] = df[df.Stock == i]

    #print(stocks_df)

    #AAPL = df[df.Stock == 'AAPL']
    #display(AAPL)
    #display(AAPL.shape[0])


    return df, stocks_df, all_stocks


class LSTM(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length):
        super(LSTM, self).__init__()
        self.num_classes = num_classes  # number of classes
        self.num_layers = num_layers  # number of layers
        self.input_size = input_size  # input size
        self.hidden_size = hidden_size  # hidden state
        self.seq_length = seq_length  # sequence length

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)  # lstm
        self.fc_1 = nn.Linear(hidden_size, 128)  # fully connected 1
        self.fc = nn.Linear(128, num_classes)  # fully connected last layer

        self.relu = nn.ReLU()

    def forward(self, x):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))  # hidden state
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))  # internal state
        # Propagate input through LSTM
        output, (hn, cn) = self.lstm(x, (h_0, c_0))  # lstm with input, hidden, and internal state
        hn = hn.view(-1, self.hidden_size)  # reshaping the data for Dense layer next
        out = self.relu(hn)
        out = self.fc_1(out)  # first Dense
        out = self.relu(out)  # relu
        out = self.fc(out)  # Final Output
        return out


# Training loop function
def training_loop(n_epochs, lstm, optimizer, loss_fn, x_train, y_train, x_val, y_val):
    for epoch in range(n_epochs):
        outputs = lstm.forward(x_train)  # forward pass
        optimizer.zero_grad()  # caluclate the gradient, manually setting to 0

        # obtain the loss function
        loss = loss_fn(outputs, y_train)

        loss.backward()  # calculates the loss of the loss function

        optimizer.step()  # improve from loss, i.e backprop
        #if epoch % 1000 == 0:
        #    print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))



# Function for sliding a window over dataframe to select window to train model on
def sliding_window(dataframe, window_size):
    # Loop to move the window by 1
    start = 0
    to = window_size

    # for accuracy
    TP = 0      # True prediction
    FP = 0      # False prediction
    attempts = 0
    accuracies = []

    for i in range(len(dataframe) - window_size):
        attempts += 1
        # Define the window
        window = dataframe.iloc[start:to]
        #display(window)

        # Make prediction on window after frame
        target = dataframe.iloc[[to]]
        #target.to_frame()
        #display(target)

        # Do training on window, and predict on target
        # Get x and y values for the window
        y_train = window[['Close']]
        x_train = window.drop(columns=['Close', 'Stock', 'Date'])
        #print(x_train.shape, y_train.shape)

        y_val = target[['Close']]
        x_val = target.drop(columns=['Close', 'Stock', 'Date'])
        #print(x_val.shape, y_val.shape)

        #x_train_tensor = Variable(torch.tensor(x_train, requires_grad=True))
        x_train_tensor = Variable(torch.tensor(x_train.values).float())
        y_train_tensor = Variable(torch.tensor(y_train.values).float())
        x_val_tensor = Variable(torch.tensor(x_val.values).float())
        y_val_tensor = Variable(torch.tensor(y_val.values).float())

        # reshaping to rows, timestamps, features
        x_train_tensors_final = torch.reshape(x_train_tensor, (x_train_tensor.shape[0], 1, x_train_tensor.shape[1]))
        x_val_tensors_final = torch.reshape(x_val_tensor, (x_val_tensor.shape[0], 1, x_val_tensor.shape[1]))

        #print("Training Shape", x_train_tensors_final.shape, y_train_tensor.shape)
        #print("Testing Shape", x_val_tensors_final.shape, y_val_tensor.shape)

        nr_epochs = 10  # 1000 epochs
        learning_rate = 0.01  # 0.001 lr

        #input_size = 4  # number of features. nr of columns in x data
        input_size = len(x_train.columns)
        hidden_size = 2  # number of features in hidden state
        num_layers = 1  # number of stacked lstm layers

        num_classes = 1  # number of output classes. length of the target. in this case 1
        seq_length = x_train_tensors_final.shape[1]
        lstm = LSTM(num_classes, input_size, hidden_size, num_layers, seq_length)

        loss_fn = torch.nn.MSELoss()  # mean-squared error for regression
        optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate)

        # Run training
        training_loop(nr_epochs, lstm=lstm, optimizer=optimizer, loss_fn=loss_fn, x_train=x_train_tensors_final, y_train=y_train_tensor, x_val=x_val_tensors_final, y_val=y_val_tensor)

        # Prediction
        prediction = lstm.forward(x_val_tensors_final)

        # Check if the  price increased
        price_comparison = y_val.iloc[-1].Close - y_train.iloc[-1].Close
        # Check if predicted is higher or lower than previous day
        pred = prediction.item()
        pred_comparison = prediction.item() - y_train.iloc[-1].Close
        # If both incresed or decreased, true positive
        if (price_comparison > 0) and (pred_comparison > 0):
            TP += 1
        elif (price_comparison <= 0) and (pred_comparison <= 0):
            TP += 1
        else:
            FP += 1




        # Move the window 1 position down the dataframe
        start += 1
        to += 1

    # Calculate accuracy for the current window
    if attempts == 0:
        print(f'Divide by zero on attempts')
        print(f'')
        accuracy = 0
    else:
        accuracy = TP / attempts
    #accuracies.append(accuracy)
    return accuracy



if __name__=='__main__':
    # Get and clean the data
    df, stocks_df, all_stocks = data()

    # Change size of sliding window
    window_size = 5
    # If stock has fewer dates than this, dont use it.
    min_stock_size = 10

    accuracies = []

    # Starting time for training and getting accuracies
    start_time = time.time()
    # Make prediciton on one stock at the time
    for x,y in stocks_df.items():
        #display(y.head())
        if len(y) < min_stock_size:
            continue
        print('-----------------------------------------------')
        print(f'Prediction on stock: {x}')
        print('strting sliding window')
        stock_start_time = time.time()
        accuracy = sliding_window(y, window_size)
        accuracies.append(accuracy)
        stock_stop_time = time.time()
        print(accuracy)
        print(f'took: {stock_stop_time - stock_start_time} seconds')

    stop_time = time.time()
    print(f'Used a total time of {stop_time - start_time} seconds')
    print(all_stocks)
    print(accuracies)


    #display(df.head())
    #print(df.columns)
    #print(df.describe())

