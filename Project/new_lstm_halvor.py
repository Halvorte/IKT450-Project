import random

import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display
import torch
import torch.nn as nn
from torch.autograd import Variable
import time
from tqdm import tqdm  # For nice progress bar!


# Funciton to read and claen the data
def data():
    df = pd.read_csv('Complete_pp_df.csv')

    # Remove unnecessary columns
    #df = df.drop(columns=['level_0', 'level_1'])
    df = df.drop(columns=['level_1'])

    # Add a column with 1 and 0 for price increase or decrease
    close_minus_open = []
    for j in range(len(df)):
        if j == 0:
            if (df.iloc[j]['Close'] - df.iloc[j]['Open']) > 0:
                close_minus_open.append(1)
            else:
                close_minus_open.append(0)
        else:
            if (df.iloc[j]['Close'] - df.iloc[j-1]['Close']) > 0:
                close_minus_open.append(1)
            else:
                close_minus_open.append(0)

    df['Cmo'] = close_minus_open

    # Split up to the different stocks
    #all_stocks = df['Stock'].unique().tolist()
    all_stocks = df['level_0'].unique().tolist()
    print(all_stocks)

    # All stocks stored in dictionary
    stocks_df = {}
    for i in all_stocks:
        #stocks_df[f'{i}'] = df[df.Stock == i]
        stocks_df[f'{i}'] = df[df['level_0'] == i]

    return df, stocks_df, all_stocks


class LSTM(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length):
        super(LSTM, self).__init__()
        self.num_classes = num_classes  # number of classes
        self.num_layers = num_layers  # number of layers
        self.input_size = input_size  # input size
        self.hidden_size = hidden_size  # hidden state
        #self.seq_length = seq_length  # sequence length

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)  # lstm
        self.fc_1 = nn.Linear(hidden_size, 128)  # fully connected 1
        self.fc = nn.Linear(128, num_classes)  # fully connected last layer
        self.sigmoid = nn.Sigmoid()
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
        out = self.sigmoid(out)
        return out


# Function to plot loss
def plot_loss(losses):
    x = [x for x in range(len(losses))]
    plt.plot(x, losses, label='losses')
    plt.title('Losses over epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    return False


# Function to plot predicted vs real data for a stock
def plot_real_pred(dataframe, predicted_vals, window_size, predicted_vals2, true2, losses):
    df_true = dataframe
    stock = df_true['level_0'].iloc[1]
    df_true['Date'] = pd.to_datetime(df_true['Date'])
    yo = df_true.Date.iloc[::-1]    # Reversing dates to make it increasing
    yb = yo[window_size:]           # Removing first days not predicted
    ya = df_true.Date[:len(predicted_vals)] # Starts plotting when the first window ends
    #reversed_pred = reversed(predicted_vals)
    # Plot real data and predicted data
    plt.figure(figsize=(22, 10))
    plt.plot(dataframe.Date, df_true.Close, label='Actual data')    # Plot real data
    plt.plot(ya, predicted_vals, label='Predicted values')          # Plot predicted data
    plt.title(f"Closing value of stock {stock}")
    plt.xlabel("Date")
    plt.ylabel("Closing value")
    plt.legend()
    plt.show()

    # Plot error/loss
    x = [x for x in range(len(losses))]
    #plt.plot(x, losses)
    #plt.figure(figsize=(22, 10))
    plt.plot(yb, losses)
    plt.title(f'Error/loss for stock {stock}')
    plt.xlabel("Date")
    plt.ylabel("error/loss")
    plt.legend()
    plt.show()

    return False


# test function for walk forward validation
def walk_forward(dataframe, window_size, hyperparams):
    # Get data
    #x = dataframe.drop(columns=['Close', 'Date', 'level_0'])
    x = dataframe.drop(columns=['Date', 'level_0'])
    #y = dataframe[['Close']]
    y = dataframe[['Cmo']]

    # input size is the number of features. nr of columns in x data
    input_size = len(x.columns)  # Decided by the number of columns in training data
    nr_epochs = hyperparams[0]
    learning_rate = hyperparams[1]
    hidden_size = hyperparams[2]
    num_layers = hyperparams[3]
    num_classes = hyperparams[4]

    #seq_length = x_tensors_final.shape[1]   # Dont need?
    seq_length = 1
    lstm = LSTM(num_classes, input_size, hidden_size, num_layers, seq_length)

    loss_fn = torch.nn.MSELoss()  # mean-squared error for regression
    optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate)



    # Get windows and store them in a list
    y_trains = []
    y_vals = []
    windows_tensor = []
    real_vals_tensor = []
    x_val_tensors = []
    y_val_tensors = []
    to_predict = []
    price_comparisons = []
    pred_comp = []
    for i in range(window_size, len(dataframe)):
        x_train = x.iloc[i - window_size:i]
        y_train = y.iloc[i - window_size:i]
        x_val = x.iloc[i:i + 1]
        y_val = y.iloc[i:i + 1]
        x_to_pred = x.iloc[i - 1:i]

        # make data to tensors
        x_train_tensor = Variable(torch.tensor(x_train.values).float())
        y_train_tensor = Variable(torch.tensor(y_train.values).float())
        x_val_tensor = Variable(torch.tensor(x_val.values).float())
        y_val_tensor = Variable(torch.tensor(y_val.values).float())
        x_to_pred_tensor = Variable(torch.tensor(x_to_pred.values).float())

        x_train_tensors_final = torch.reshape(x_train_tensor, (x_train_tensor.shape[0], 1, x_train_tensor.shape[1]))
        x_val_tensors_final = torch.reshape(x_val_tensor, (x_val_tensor.shape[0], 1, x_val_tensor.shape[1]))
        x_to_pred_tensor_final = torch.reshape(x_to_pred_tensor, (x_to_pred_tensor.shape[0], 1, x_to_pred_tensor.shape[1]))

        windows_tensor.append(x_train_tensors_final)
        real_vals_tensor.append(y_train_tensor)
        x_val_tensors.append(x_val_tensors_final)
        y_val_tensors.append(y_val_tensor)
        to_predict.append(x_to_pred_tensor_final)
        y_trains.append(y_train)
        y_vals.append(y_val)
        # make a price comparison for each window
        price_comp = y_val.iloc[-1].Cmo
        price_comparisons.append(price_comp)

        pred_comp.append(y_train.iloc[-1].Cmo)

    epoch_losses = []
    correct_pred = 0
    wrong_pred = 0
    TP = 0
    TN = 0
    FP = 0
    FN = 0

    # Train on the data epoch amount of times
    for epoch in tqdm(range(nr_epochs)):
        # Train on all the windows
        for i in range(len(windows_tensor)):

            outputs = lstm.forward(windows_tensor[i])  # forward pass
            optimizer.zero_grad()  # caluclate the gradient, manually setting to 0

            # obtain the loss function
            loss = loss_fn(outputs, real_vals_tensor[i])
            loss.backward()  # calculates the loss of the loss function

            optimizer.step()  # improve from loss, i.e backprop
            #if epoch % 1000 == 0:
            #    print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))

        # Select a random window and get accuracy and error
        rand_windnr = random.randrange(len(windows_tensor))
        rand_window = windows_tensor[rand_windnr]
        random_windows_real_val = real_vals_tensor[rand_windnr]
        # Make prediction on window
        prediction = lstm.forward(to_predict[rand_windnr])

        # Get loss from window
        val_loss = loss_fn(prediction, random_windows_real_val)
        epoch_losses.append(val_loss.item())
        # Round up or down prediction
        prediction = round(prediction.item())  # Round the prediction up or down to 1 or 0

        # If both incresed or decreased, true positive. else wrong pred
        if (price_comparisons[rand_windnr] == 1) and (prediction == 1):
            TP += 1
            correct_pred += 1
        elif (price_comparisons[rand_windnr] == 0) and (prediction == 0):
            correct_pred += 1
            TN += 1
        elif (price_comparisons[rand_windnr] == 1) and (prediction == 0):
            FN += 1
        elif (price_comparisons[rand_windnr] == 0) and (prediction == 1):
            FP += 1
        else:
            wrong_pred += 1

    accuracy = correct_pred / nr_epochs
    plot_loss(epoch_losses)

    # Fix for division by zero faults
    print("Accuracy:", (TP + TN) / (TP + TN + FP + FN))
    print("Recall", TP / (TP + FN))
    print("Precision", TP / (TP + FP))
    print("F1", (2 * TP) / (2 * TP + FP + FN))

    # Need to fix
    # Plot predicted vs real
    #plot_real_pred(dataframe, predicted_vals, window_size, predicted_vals2, true2, losses)

    return accuracy


if __name__=='__main__':
    # Get and clean the data
    df, stocks_df, all_stocks = data()


    window_size = 5         # Change size of sliding window

    min_stock_size = 200    # If stock has fewer dates than this, dont use it.
    # Hyperparameters
    hyperparams = []
    nr_epochs = 100          # 1000 epochs
    learning_rate = 0.001    # 0.001 lr
    hidden_size = 2         # number of features in hidden state
    num_layers = 1          # number of stacked lstm layers
    num_classes = 1         # number of output classes. length of the target. in this case 1 becuase we predict 1 day ahead.
    hyperparams.extend([nr_epochs, learning_rate, hidden_size, num_layers, num_classes])

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

        accuracy = walk_forward(y, window_size, hyperparams)

        #accuracy = sliding_window(y, window_size, hyperparams)
        accuracies.append(accuracy)
        stock_stop_time = time.time()
        print(f'Accuracy: {accuracy}')
        print(f'took: {stock_stop_time - stock_start_time} seconds')

    stop_time = time.time()
    print(f'Used a total time of {stop_time - start_time} seconds')
    print(all_stocks)
    print(accuracies)
    print(f'Average accuracy: {(sum(accuracies) / len(accuracies))}')