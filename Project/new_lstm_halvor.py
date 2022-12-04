import random
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.autograd import Variable
import time

from torch.utils.data import DataLoader
from tqdm import tqdm  # For nice progress bar!
import torch.utils.data


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
        self.fc_1 = nn.Linear(hidden_size, 7)  # fully connected 1
        #self.fc_2 = nn.Linear(7,128)
        self.fc = nn.Linear(7, num_classes)  # fully connected last layer
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device))  # hidden state
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device))  # internal state

        # Propagate input through LSTM
        output, (hn, cn) = self.lstm(x, (h_0, c_0))  # lstm with input, hidden, and internal state
        hn = hn.view(-1, self.hidden_size)  # reshaping the data for Dense layer next
        out = self.relu(hn)
        out = self.fc_1(out)  # first Dense
        out = self.dropout(out)
        out = self.relu(out)  # relu
        #out = self.fc_2(out)
        #out = self.relu(out)
        out = self.fc(out)  # Final Output layer
        out = self.sigmoid(out)     # Sigmoid to get prediction between 0 and 1
        return out


# Function to plot loss
def plot_loss(losses, stock_name):
    x = [x for x in range(len(losses))]
    plt.plot(x, losses, label='losses')
    plt.title(f'Losses over epochs for {stock_name}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    return False

def plot_acc(losses, stock_name):
    x = [x for x in range(len(losses))]
    plt.plot(x, losses, label='Accuracy')
    plt.title(f'Accuracy over epochs for {stock_name}')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
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
    stock_name = dataframe.iloc[1]['level_0']
    #x = dataframe.drop(columns=['Date', 'level_0', 'compound score'])
    x = dataframe.drop(columns=['Date', 'level_0'])
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lstm = LSTM(num_classes, input_size, hidden_size, num_layers, seq_length).to(device=device)



    #loss_fn = torch.nn.MSELoss()               # mean-squared error for regression
    #loss_fn = torch.nn.CrossEntropyLoss()      # Cross entropy loss. better for binary classification
    loss_fn = torch.nn.BCELoss()                # Binary cross entropy loss
    optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate)
    #optimizer = torch.optim.SGD(lstm.parameters(), lr=learning_rate)    # Lower accuracy than adam



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
    real_to_pred = []
    for i in range(window_size, len(dataframe)):
        x_train = x.iloc[i - window_size:i]
        y_train = y.iloc[i - window_size:i]
        x_val = x.iloc[i:i + 1]
        y_val = y.iloc[i:i + 1]
        x_to_pred = x.iloc[i - 1:i]
        real_from_pred = y.iloc[i - 1:i]

        # make data to tensors
        x_train_tensor = Variable(torch.tensor(x_train.values).float())
        y_train_tensor = Variable(torch.tensor(y_train.values).float())
        x_val_tensor = Variable(torch.tensor(x_val.values).float())
        y_val_tensor = Variable(torch.tensor(y_val.values).float())
        x_to_pred_tensor = Variable(torch.tensor(x_to_pred.values).float())
        real_from_pred_tensor = Variable(torch.tensor(real_from_pred.values).float())

        x_train_tensors_final = torch.reshape(x_train_tensor, (x_train_tensor.shape[0], 1, x_train_tensor.shape[1]))
        x_val_tensors_final = torch.reshape(x_val_tensor, (x_val_tensor.shape[0], 1, x_val_tensor.shape[1]))
        x_to_pred_tensor_final = torch.reshape(x_to_pred_tensor, (x_to_pred_tensor.shape[0], 1, x_to_pred_tensor.shape[1]))

        windows_tensor.append(x_train_tensors_final)
        real_vals_tensor.append(y_train_tensor)
        x_val_tensors.append(x_val_tensors_final)
        y_val_tensors.append(y_val_tensor)
        to_predict.append(x_to_pred_tensor_final)
        real_to_pred.append(real_from_pred_tensor)
        y_trains.append(y_train)
        y_vals.append(y_val)
        # make a price comparison for each window
        price_comp = y_val.iloc[-1].Cmo
        price_comparisons.append(price_comp)

        pred_comp.append(y_train.iloc[-1].Cmo)

    epoch_losses = []
    predictions = []
    reals = []
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
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            to_pred = windows_tensor[i]
            to_pred = to_pred.to(device=device)

            real_vals = real_vals_tensor[i]
            real_vals = real_vals.to(device=device)

            outputs = lstm.forward(to_pred)  # forward pass

            optimizer.zero_grad()  # caluclate the gradient, manually setting to 0

            # obtain the loss function
            loss = loss_fn(outputs, real_vals)
            loss.backward()  # calculates the loss of the loss function

            optimizer.step()  # improve from loss, i.e backprop
            #if epoch % 1000 == 0:
            #    print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))

        # Select a random window and get accuracy and error
        #rand_windnr = random.randrange(len(windows_tensor))
        rand_windnr = 12
        rand_window = windows_tensor[rand_windnr]
        random_windows_real_val = real_vals_tensor[rand_windnr]
        random_windows_real_val = random_windows_real_val.to(device=device)
        # Make prediction on window
        rand_to_predict = to_predict[rand_windnr]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        rand_to_predict = rand_to_predict.to(device=device)
        prediction = lstm.forward(rand_to_predict)

        # Get loss from window
        rand_real = real_to_pred[rand_windnr]
        rand_real = rand_real.to(device=device)
        val_loss = loss_fn(prediction, rand_real)
        epoch_losses.append(val_loss.item())
        # Round up or down prediction
        prediction = round(prediction.item())  # Round the prediction up or down to 1 or 0
        predictions.append(prediction)
        reals.append(price_comparisons[rand_windnr])

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
    plot_loss(epoch_losses, stock_name)

    cm = confusion_matrix(reals, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title(f'Stock: {stock_name}')
    plt.show()

    # Fix for division by zero faults
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

    # Need to fix
    # Plot predicted vs real
    #plot_real_pred(dataframe, predicted_vals, window_size, predicted_vals2, true2, losses)

    return accuracy


def get_windows(df, stock):
    tempDf = dict( (k, v) for k, v in df.items() if str(k) == str(stock) )
    newDf = pd.DataFrame()
    yDf = pd.DataFrame()
    xList = []
    yList = []
    for key, value in tempDf.items():
        a = value.drop(columns=['Date', 'Cmo', 'level_0'])
        c = value[['Cmo']]
        for i in range(len(value)):
            X = a.iloc[i - window_size:i]
            Y = c.iloc[i - window_size:i]
            X = X.values.tolist()
            Y = Y.values.tolist()
            if X and Y:
                xList.append(X)
                yList.append(Y)
            else:
                pass

    xDf = pd.Series(xList)
    yDf = pd.Series(yList)
    return xDf, yDf


def train_model(data, model, loss_function, optimizer):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_batches = len(data)
    total_loss = 0
    for X, y in data:
        to_pred = X
        to_pred = to_pred.to(device=device)

        real_vals = y
        real_vals = real_vals.to(device=device)

        outputs = model.forward(to_pred)  # forward pass

        optimizer.zero_grad()  # caluclate the gradient, manually setting to 0

        # obtain the loss function
        # Realnum, get the 5th days number
        realNum = real_vals.cpu()
        realNum = realNum.numpy()
        realNum = realNum.take([-1])
        realNum = torch.from_numpy(realNum)
        realNum = realNum.unsqueeze(1)
        predNum = outputs.cpu().detach().numpy()
        predNum = torch.tensor(predNum)

        loss = loss_function(predNum, realNum)
        loss.requires_grad = True
        loss.backward()  # calculates the loss of the loss function

        optimizer.step()  # improve from loss, i.e backprop

        total_loss += loss.item()

        # if epoch % 1000 == 0:
        #    print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))

    avg_loss = total_loss / num_batches
    print(f"Train loss: {avg_loss}")


def test_model(data, model, loss_function):
    predictions = []
    real = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_batches = len(data)
    total_loss = 0

    with torch.no_grad():
        for X, y in data:
            X = X.unsqueeze(1)
            to_pred = X
            to_pred = to_pred.to(device=device)

            real_vals = y
            real_vals = real_vals.to(device=device)

            output = model.forward(to_pred)

            total_loss += loss_function(output, real_vals)

            # Log the values
            prediction = round(output.item())
            predictions.append(prediction)
            real.append(real_vals.item())

    avg_loss = total_loss / num_batches
    print(f"Test loss: {avg_loss}")

    TP = 0
    TN = 0
    FP = 0
    FN = 0
    correct_pred = 0
    wrong_pred = 0
    for i in range(len(predictions)):
        if (predictions[i] == 1) and (real[i] == 1):
            TP += 1
            correct_pred += 1
        elif (predictions[i] == 0) and (real[i] == 0):
            correct_pred += 1
            TN += 1
        elif (predictions[i] == 1) and (real[i] == 0):
            FP += 1
        elif (predictions[i] == 0) and (real[i] == 1):
            FN += 1
        else:
            wrong_pred += 1


    # Fix for division by zero faults
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


if __name__=='__main__':
    # Check if CUDA gpu is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    # Get and clean the data
    df, stocks_df, all_stocks = data()


    window_size = 5         # Change size of sliding window

    # xDf, yDf = get_windows(stocks_df, window_size)
    stock_names = df['level_0'].unique().tolist()

    min_stock_size = 200    # If stock has fewer dates than this, dont use it.
    # Hyperparameters
    hyperparams = []
    nr_epochs = 5         # 1000 epochs
    learning_rate = 0.001    # 0.001 lr
    hidden_size = 1         # number of features in hidden state
    num_layers = 1          # number of stacked lstm layers
    num_classes = 1         # number of output classes. length of the target. in this case 1 becuase we predict 1 day ahead.
    hyperparams.extend([nr_epochs, learning_rate, hidden_size, num_layers, num_classes])

    accuracies = []

    # Starting time for training and getting accuracies
    start_time = time.time()

    # Make prediciton on one stock at the
    for stock in stock_names:
        print(stock)
        xDf, yDf = get_windows(stocks_df, stock)
        if(len(xDf) < 50):
            pass
        else:
            X_train, X_test, y_train, y_test = train_test_split(xDf, yDf, test_size=0.33, random_state=42)
        #
        # print(X_train)
        # print(X_test)
        # print(y_train)
        # print(y_test)

            new_xTest = []
            for sub_list in X_test:
                new_xTest.append(sub_list[-1])

            new_yTest = []
            for sub_list in y_test:
                new_yTest.append(sub_list[-1])

            new_xTest = torch.Tensor(new_xTest)
            new_yTest = torch.Tensor(new_yTest)

            X_train = torch.Tensor(list(X_train.values))
            y_train = torch.Tensor(list(y_train.values))
            # X_test = torch.Tensor(list(new_xTest.values))
            # y_test = torch.Tensor(list(new_yTest.values))

            trainDataset = torch.utils.data.TensorDataset(X_train, y_train)
            testDataset = torch.utils.data.TensorDataset(new_xTest, new_yTest)
            train_loader = DataLoader(trainDataset, batch_size=1, shuffle=True)
            test_loader = DataLoader(testDataset, batch_size=1, shuffle=True)
            X, y = next(iter(train_loader))
            #
            # print("Features shape:", X.shape)
            # print("Target shape:", y.shape)

            # input size is the number of features. nr of columns in x data
            input_size = X_train.shape[2]  # Decided by the number of columns in training data
            nr_epochs = hyperparams[0]
            learning_rate = hyperparams[1]
            hidden_size = hyperparams[2]
            num_layers = hyperparams[3]
            num_classes = hyperparams[4]

            # seq_length = x_tensors_final.shape[1]   # Dont need?
            seq_length = 1
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            lstm = LSTM(num_classes, input_size, hidden_size, num_layers, seq_length).to(device=device)


            # loss_fn = torch.nn.MSELoss()               # mean-squared error for regression
            # loss_fn = torch.nn.CrossEntropyLoss()      # Cross entropy loss. better for binary classification
            loss_fn = torch.nn.BCELoss()  # Binary cross entropy loss
            optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate)
            # optimizer = torch.optim.SGD(lstm.parameters(), lr=learning_rate)    # Lower accuracy than adam

            for epoch in range(nr_epochs):
                print(f"Epoch {epoch}\n---------")
                train_model(train_loader, lstm, loss_fn, optimizer=optimizer)
                test_model(test_loader, lstm, loss_fn)
                print()

    # for x,y in stocks_df.items():
    #     #display(y.head())
    #     if len(y) < min_stock_size:
    #         continue
    #     print('-----------------------------------------------')
    #     print(f'Prediction on stock: {x}')
    #     print('strting sliding window')
    #     stock_start_time = time.time()
    #
    #     accuracy = walk_forward(y, window_size, hyperparams)
    #
    #     #accuracy = sliding_window(y, window_size, hyperparams)
    #     accuracies.append(accuracy)
    #     stock_stop_time = time.time()
    #     #print(f'Accuracy: {accuracy}')
    #     print(f'took: {stock_stop_time - stock_start_time} seconds')
    #
    # stop_time = time.time()
    # print(f'Used a total time of {stop_time - start_time} seconds')
    # print(all_stocks)
    # print(accuracies)
    # print(f'Average accuracy: {(sum(accuracies) / len(accuracies))}')