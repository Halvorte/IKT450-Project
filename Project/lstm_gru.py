import random
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
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
from torch.utils.data import DataLoader
from torchvision import models


# Function to read and clean the data
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
    #print(all_stocks)

    # All stocks stored in dictionary
    stocks_df = {}
    for i in all_stocks:
        #stocks_df[f'{i}'] = df[df.Stock == i]
        stocks_df[f'{i}'] = df[df['level_0'] == i]

    return df, stocks_df, all_stocks


# function to split data into train val and windows
def windows(dataframe, window_size):
    #dataframe = dataframe.drop(columns=['level_0', 'Date', 'compound score'])
    dataframe = dataframe.drop(columns=['level_0', 'Date'])
    raw = dataframe.to_numpy()
    x = []
    y = []

    # Create the windows
    for i in range(len(raw) - window_size):
        row = [r for r in raw[i:i+window_size]]
        x.append(row)
        label = raw[i+window_size][-1]
        y.append(label)

    x1 = np.array(x)
    y1 = np.array(y)

    train_size = int(np.round(0.8*x1.shape[0]))
    test_size = x1.shape[0] - train_size

    x1_train, y1_train = x1[:train_size], y1[:train_size]
    x1_test, y1_test = x1[train_size:], y1[train_size:]

    # Standardise the volume column
    training_volume_mean = np.mean(x1_train[:, :, 5])
    training_volume_std = np.std(x1_train[:, :, 5])
    x1_train[:,:,5] = (x1_train[:,:,5] - training_volume_mean) / training_volume_std

    test_volume_mean = np.mean(x1_test[:, :, 5])
    test_volume_std = np.std(x1_test[:, :, 5])
    x1_test[:, :, 5] = (x1_test[:, :, 5] - test_volume_mean) / test_volume_std

    return x1_train, y1_train, x1_test, y1_test


class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(64, output_dim)
        self.fc1 = nn.Linear(hidden_dim, 64)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim, device=x.device).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim, device=x.device).requires_grad_()
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        #out, hn = self.gru(x, h0.detach())
        out = self.relu(out)
        #out = self.dropout(out)
        out = self.fc1(out)
        #out = self.relu(out)
        #out = self.sigmoid(out)
        out = self.fc(out[:, -1, :])
        out = self.sigmoid(out)
        return out


# TP TN FP FN function
def comparison(real, predicted):
    TP = 0
    TN = 0
    FP = 0
    FN = 0

    for i in range(len(real)):

        if (real[i] == 1) and (predicted[i] == 1):
            TP += 1
            #correct_pred += 1
        elif (real[i] == 0) and (predicted[i] == 0):
            #correct_pred += 1
            TN += 1
        elif (real[i] == 1) and (predicted[i] == 0):
            FN += 1
        elif (real[i] == 0) and (predicted[i] == 1):
            FP += 1

    return TP, TN, FP, FN


# Funciton to calculate train accuracy for an epoch
def train_acc(y_train_pred, y_train, t, num_epochs):
    #y_train_pred = y_train_pred.tolist()
    y_train_pred = [round(num) for num in y_train_pred]
    #y_train = y_train.tolist()
    TP, TN, FP, FN = comparison(y_train, y_train_pred)

    if t == num_epochs - 1:
        # Fix for division by zero faults
        print('\nTrain results:')
        print("Accuracy:", (TP + TN) / (TP + TN + FP + FN))
        if (TP + FN) == 0:
            print('Recall division by zero fault')
        else:
            print("Recall", TP / (TP + FN))
        if (TP + FP) == 0:
            print('Precision division by zero fault')
        else:
            print("Precision", TP / (TP + FP))
        if (2 * TP + FP + FN) == 0:
            print('F1 division by zero fault')
        else:
            print("F1", (2 * TP) / (2 * TP + FP + FN))

    accuracy = (TP + TN) / (TP + TN + FP + FN)

    return accuracy


def test_acc(y_test_pred, y_test, t, num_epochs, stock_name):
    # y_train_pred = y_train_pred.tolist()
    y_test_pred = [round(num) for num in y_test_pred]
    # y_train = y_train.tolist()
    TP, TN, FP, FN = comparison(y_test, y_test_pred)

    if t == num_epochs - 1:
        # Fix for division by zero faults
        print('\nTest results:')
        print("Accuracy:", (TP + TN) / (TP + TN + FP + FN))
        if (TP + FN) == 0:
            print('Recall division by zero fault')
        else:
            print("Recall", TP / (TP + FN))
        if (TP + FP) == 0:
            print('Precision division by zero fault')
        else:
            print("Precision", TP / (TP + FP))
        if (2 * TP + FP + FN) == 0:
            print('F1 division by zero fault')
        else:
            print("F1", (2 * TP) / (2 * TP + FP + FN))

        # Confusion matrix
        cm = confusion_matrix(y_test, y_test_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.title(f'Validation on Stock: {stock_name}')
        plt.show()

    accuracy = (TP + TN) / (TP + TN + FP + FN)

    return accuracy


# Function to plot loss and accuracy
def plot_results(train_avg_losses, test_losses, train_accuracies, test_accuracies, num_epochs, stock_name):
    x = [x for x in range(num_epochs)]
    # z = [z for z in range(len(test_losses))]
    plt.title(f'Train and Test Loss on stock {stock_name}')
    plt.plot(x, train_avg_losses, label='train losses')
    plt.plot(x, test_losses, label='test loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    plt.title(f'Train and Test Accuracy on stock {stock_name}')
    #x = [x for x in range(len(train_accuracies))]
    plt.plot(x, train_accuracies, label='Train accuracy')
    #x = [x for x in range(len(test_accuracies))]
    plt.plot(x, test_accuracies, label='Test accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
    return False


# Main function
if __name__=='__main__':
    # Check if CUDA gpu is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    # Get and clean the data
    df, stocks_df, all_stocks = data()

    window_size = 5         # Change size of sliding window

    min_stock_size = 350    # If stock has fewer days of data than this, dont use it.

    input_dim = 8
    hidden_dim = 16
    num_layers = 2
    output_dim = 1
    num_epochs = 500
    learning_rate = 0.0001
    batch_size = 1024
    num_workers = 6

    # Starting time for training and getting accuracies
    start_time = time.time()

    average_test_acc = []

    #model = LSTM(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers).to(device=device)

    # Make prediciton on one stock at the time
    for x,y in stocks_df.items():
        stock_name = x
        if len(y) < min_stock_size:
            continue
        #print('-----------------------------------------------')
        print(f'\nPrediction on stock: {x}')
        stock_start_time = time.time()

        # Get train and test
        x_train, y_train, x_test, y_test = windows(y, window_size)
        # Convert to tensors
        x_train = torch.from_numpy(x_train).type(torch.Tensor)
        y_train = torch.from_numpy(y_train).type(torch.Tensor)

        x_test = torch.from_numpy(x_test).type(torch.Tensor)
        y_test = torch.from_numpy(y_test).type(torch.Tensor)

        trainDataset = torch.utils.data.TensorDataset(x_train, y_train)
        trainDataloader = torch.utils.data.DataLoader(trainDataset, batch_size=batch_size, shuffle=True)

        testDataset = torch.utils.data.TensorDataset(x_test, y_test)
        testDataloader = torch.utils.data.DataLoader(testDataset, batch_size=batch_size, shuffle=False)

        model = LSTM(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers).to(device=device)
        #criterion = torch.nn.MSELoss(reduction='mean')
        criterion = torch.nn.BCELoss(reduction='mean')  # Binary cross entropy loss
        #criterion = torch.nn.BCEWithLogitsLoss()
        optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
        #optimiser = torch.optim.SGD(model.parameters(), lr=learning_rate)
        #optimiser = torch.optim.Adamax(model.parameters(), lr=learning_rate)

        hist = np.zeros(num_epochs)
        start_time = time.time()
        lstm = []
        train_accuracies = []
        train_avg_acc = []
        train_avg_losses = []
        test_accuracies = []
        test_losses = []
        for t in tqdm(range(num_epochs)):
            train_running_loss = 0
            test_running_loss = 0
            train_tot_loss = 0
            test_tot_loss = 0
            batch_pred_training_accuracy = []
            batch_label_training_accuracy = []
            batch_pred_test_accuracy = []
            batch_label_test_accuracy = []

            #print(f'Epoch {t}')
            model.train(True)
            for i, data in enumerate(trainDataloader):
                #print('yo, train batch')
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                inputs, labels = data
                optimiser.zero_grad()   # Zero the gradients for every batch
                inputs = inputs.to(device=device)
                labels = labels.to(device=device)

                y_train_pred = model(inputs)
                y_train_pred = y_train_pred.view(-1,)
                loss = criterion(y_train_pred, labels)

                # get batch loss
                train_running_loss += loss.item()
                train_tot_loss += loss.item()

                # predictions and real
                batch_pred_training_accuracy.extend(y_train_pred.tolist())
                batch_label_training_accuracy.extend(labels.tolist())

                loss.backward()
                optimiser.step()

            model.train(False)

            # Get train loss
            train_loss = train_running_loss / len(trainDataloader)
            train_avg_losses.append(train_loss)

            # Get train accuracy
            train_epoch_accuracy = train_acc(batch_pred_training_accuracy, batch_label_training_accuracy, t, num_epochs)
            train_accuracies.append(train_epoch_accuracy)

            model.eval()
            for j, data in enumerate(testDataloader):
                inputs, labels = data
                inputs = inputs.to(device=device)
                labels = labels.to(device=device)

                y_test_pred = model(inputs)
                y_test_pred = y_test_pred.view(-1, )
                loss = criterion(y_test_pred, labels)

                # get batch loss
                test_running_loss += loss.item()
                test_tot_loss += loss.item()

                # get batch accuracy
                batch_pred_test_accuracy.extend(y_test_pred.tolist())
                batch_label_test_accuracy.extend(labels.tolist())

            # Get test loss
            test_loss = test_running_loss / len(testDataloader)
            test_losses.append(test_loss)

            # Get test accuracy and print confusion matrix on last epoch
            test_epoch_accuracy = test_acc(batch_pred_test_accuracy, batch_label_test_accuracy, t, num_epochs, stock_name)
            test_accuracies.append(test_epoch_accuracy)

        average_test_acc.append(test_accuracies[-1])    # Add the test accuracy from last epoch to

        training_time = time.time() - start_time
        print(f"Training time: {training_time}")

        # plot loss and accuracy for train and test
        plot_results(train_avg_losses, test_losses, train_accuracies, test_accuracies, num_epochs, stock_name)


    print(f'Average test accuracy across all stocks: {sum(average_test_acc) / len(average_test_acc)}')