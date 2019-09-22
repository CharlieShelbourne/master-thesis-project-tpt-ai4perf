import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from tqdm import tqdm as tqdm


def load_data(path, model, training=True):
    data = np.load(path)
    # data = scale(data)
    print(data.shape)
    y = data[:, :, data.shape[2] - 1]
    X = data[:, :, 0:data.shape[2] - 1]
    # print(X.shape)
    if training == True:
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)
        # X = X.swapaxes(1,2)
        # if model == 'CNN':
        # X = X.swapaxes(1,2)
        # elif model == 'LSTM':
        # X = X.swapaxes(0,1)
        # X_train = X_train.swapaxes(0,1)
        # X_val = X_val.swapaxes(0,1)
        # y_train = y_train.swapaxes(0,1)
        # y_val = y_val.swapaxes(0,1)
        # plot_data(X)

        # X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

        val_data = torch.utils.data.TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
        train_data = torch.utils.data.TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))

        return train_data, val_data
    else:
        test_data = torch.utils.data.TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
        return test_data


class CNN(nn.Module):
    def __init__(self,
                 num_features: int = 26,
                 hidden_layer = 10,
                 num_classes: int = 3):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=num_features, out_channels=hidden_layer, kernel_size=5, stride=3)
        self.fc1 = nn.Linear(hidden_layer * 5, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool1d(x, kernel_size=3)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x


class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.lstm1 = nn.LSTM(26, 15)
        self.fc1 = nn.Linear(50 * 15, 3)

    def forward(self, x):
        ln = len(x[1])
        x, _ = self.lstm1(x)
        x = F.relu(x)
        x = x.contiguous().view(x.shape[1], x.shape[0] * x.shape[2])  # ???
        x = self.fc1(x)
        return x


def predict(net, loader, model):
    lab = 0
    correct = 0
    total = 0
    error = []
    with torch.no_grad():
        for i, data in enumerate(loader):
            inputs, labels = data
            labels = labels.to(device="cpu", dtype=torch.int64)

            if model == "LSTM":
                aux = inputs.numpy()
                aux = aux.swapaxes(0, 1)
                inputs = torch.from_numpy(aux)
            if model == "CNN":
                aux = inputs.numpy()
                aux = aux.swapaxes(1, 2)
                inputs = torch.from_numpy(aux)

            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            labels = labels[:, 0]
            correct += (predicted == labels).sum().item()
            # lab += (predicted == 0).sum().item()
    # print(lab)
    error = (total - correct) / total
    # print('Accuracy of the network: %d %%' % (100 * correct / total))
    return error


if __name__ == '__main__':
    bs = 1  ## TODO: pass this as a parameter
    model = 'CNN'  ## TODO: pass this as a parameter

    train_data, val_data = load_data('./training_data.npy', model)  ## TODO: pass the path as a parameter

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=bs)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=bs)

    # test_data = load_data('./test_data.npy',model,training=False) ## TODO: pass the path as a parameter
    # test_loader = torch.utils.data.DataLoader(test_data, batch_size=bs)

    ## Call instant of model
    if model == "CNN":
        net = CNN()
    elif model == "LSTM":
        net = LSTM()
    net = net.double()

    ## Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    # criterion = nn.BCELoss()
    # optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=0)
    optimizer = optim.Adam(net.parameters(), lr=0.01)  # use adam, it is faster

    running_losses = []
    train_error = []
    val_error = []
    test_error = []

    ## TRAIN MODEL
    for epoch in range(10):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(train_loader):
            # train in batches
            inputs, labels = data

            if model == "LSTM":
                aux = inputs.numpy()
                aux = aux.swapaxes(0, 1)
                inputs = torch.from_numpy(aux)
            elif model == 'CNN':
                aux = inputs.numpy()
                aux = aux.swapaxes(1, 2)
                inputs = torch.from_numpy(aux)

            labels = labels.to(device="cpu", dtype=torch.int64)
            labels = labels[:, 0]

            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            y_pred = net(inputs)
            loss = criterion(y_pred, labels)
            # backwards pass
            loss.backward()
            # update weigths
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            if i % 10 == 9:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 10))  # INFO: you need to make sure that this loss decreases
                running_losses.append(running_loss / 10)
                running_loss = 0.0

            # net.eval() # INFO: do not forget to do the model.eval()
            # train_error.append(predict(net, train_loader, model)) # INFO: this wastes a lot of time
            # val_error.append(predict(net, val_loader, model))
            # test_error.append(predict(net, test_loader, model))
            # net.train() # INFO: do not forget to do model.train() after you finished the evaluation

    print('Finished Training')

    ## Plot learning error
    plt.plot(running_losses, label="training loss")
    # plt.plot(train_error,label = 'training')
    # plt.plot(val_error, label = 'validation')
    # plt.plot(test_error, label = 'test')
    plt.ylabel('error')
    plt.xlabel('batch')
    plt.title(model + ' batch size ' + str(bs))
    plt.legend()

    # plt.savefig('/home/francesco/Documents/Thesis_project/Results/ML/'
    # + model + '_model_perf_bs_'+str(bs)+'.pdf')

    plt.show()
