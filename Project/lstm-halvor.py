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
    df = pd.read_csv('Complete_pp_df.csv')

    # Remove unnecessary columns
    #df = df.drop(columns=['level_0', 'level_1'])
    df = df.drop(columns=['level_1'])

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
    x = dataframe.drop(columns=['Close', 'Date', 'level_0'])
    y = dataframe[['Close']]

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

    predicted_vals = []
    predicted_vals2 = []
    true2 = []
    losses = []
    TP = 0
    FP = 0
    attempts = 0

    for i in range(window_size, len(dataframe)):
        attempts += 1
        x_train = x.iloc[i-window_size:i]
        y_train = y.iloc[i-window_size:i]
        x_val = x.iloc[i:i+1]
        y_val = y.iloc[i:i+1]
        x_test = x.iloc[i-1:i]

        # make data to tensors
        x_train_tensor = Variable(torch.tensor(x_train.values).float())
        y_train_tensor = Variable(torch.tensor(y_train.values).float())
        x_val_tensor = Variable(torch.tensor(x_val.values).float())
        y_val_tensor = Variable(torch.tensor(y_val.values).float())
        x_test_tensor = Variable(torch.tensor(x_test.values).float())

        x_train_tensors_final = torch.reshape(x_train_tensor, (x_train_tensor.shape[0], 1, x_train_tensor.shape[1]))
        x_val_tensors_final = torch.reshape(x_val_tensor, (x_val_tensor.shape[0], 1, x_val_tensor.shape[1]))
        #y_train_tensor_final = torch.reshape(y_train_tensor, (y_train_tensor[0], 1, y_train_tensor[1]))
        x_test_tensor_final = torch.reshape(x_test_tensor, (x_test_tensor.shape[0], 1, x_test_tensor.shape[1]))

        # Run training
        training_loop(nr_epochs, lstm=lstm, optimizer=optimizer, loss_fn=loss_fn, x_train=x_train_tensors_final,
                      y_train=y_train_tensor, x_val=x_val_tensors_final, y_val=y_val_tensor)
        # To predict the future, input the window of previous days
        #prediction = lstm.forward(x_val_tensors_final)  # Wrong. inputing next days values.
        #prediction = lstm.forward(x_train_tensors_final)
        #prediction = lstm.forward(y_test_tensor)
        prediction = lstm.forward(x_test_tensor_final)

        predicted_vals.append(prediction.item())


        # Loss
        val_loss = loss_fn(prediction, y_val_tensor)
        losses.append(val_loss.item())
        # Check if the  price increased
        price_comparison = y_val.iloc[-1].Close - y_train.iloc[-1].Close
        if price_comparison > 0:
            true2.append(1)
        else:
            true2.append(0)

        # Check if predicted is higher or lower than previous day
        pred_comparison = prediction.item() - y_train.iloc[-1].Close
        if pred_comparison > 0:
            predicted_vals2.append(1)
        else:
            predicted_vals2.append(0)
        # If both incresed or decreased, true positive. else wrong pred
        if (price_comparison > 0) and (pred_comparison > 0):
            TP += 1
        elif (price_comparison <= 0) and (pred_comparison <= 0):
            TP += 1
        else:
            FP += 1

    # Calculate accuracy for the current window
    if attempts == 0:
        print(f'Divide by zero on attempts')
        print(f'')
        accuracy = 0
    else:
        accuracy = TP / attempts
    # accuracies.append(accuracy)

    # Plot predicted vs real
    plot_real_pred(dataframe, predicted_vals, window_size, predicted_vals2, true2, losses)

    return accuracy


if __name__=='__main__':
    # Get and clean the data
    df, stocks_df, all_stocks = data()


    window_size = 5         # Change size of sliding window

    min_stock_size = 200    # If stock has fewer dates than this, dont use it.
    # Hyperparameters
    hyperparams = []
    nr_epochs = 10          # 1000 epochs
    learning_rate = 0.01    # 0.001 lr
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