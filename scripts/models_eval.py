import pandas as pd
import sys
import csv
import random

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

from tqdm import tqdm as tqdm

import training_data_gen

from imblearn.under_sampling import RandomUnderSampler
from collections import Counter

from Transformer import AttentionIsAllYouNeed

def load_data(path, model, training=True):
    data = np.load(path)
    # data = scale(data)
    data[:,:,[data.shape[2] - 2, data.shape[2] - 1]] = data[:,:,[data.shape[2] - 1, data.shape[2] - 2]]
    y = data[:, :, data.shape[2] - 1]
    X = data[:, :, 0:data.shape[2] - 2]

    if training == True:
        data = np.delete(data, data.shape[2] - 2, 2)
        #train, val = five_folds(data)
        #train = np.split(train,[train.shape[3]-1],3)
        #X_train = train[0]
        #y_train = train[1]
        #val = np.split(val, [val.shape[2]-1], 2)
        #X_val = val[0]
        #y_val = val[1]
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

        val_data = torch.utils.data.TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))

        train_data = torch.utils.data.TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
        #train_data = []
        #for i in range(4):
        #    train_data.append(torch.utils.data.TensorDataset(torch.from_numpy(X_train[i]), torch.from_numpy(y_train[i])))

        return train_data, val_data
    else:
        data = torch.utils.data.TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
        return data


def multiclass_roc_auc_score(y_test, y_pred):
    lb = preprocessing.LabelBinarizer()

    lb.fit(y_test)

    y_test = lb.transform(y_test)
    y_pred = lb.transform(y_pred)
    try:
        return roc_auc_score(y_test, y_pred, average='micro')
    except:
        ValueError
        return 1


def df_to_array(df):
    array = np.asarray(df)#.reshape((int(df.shape[0] / 1), 1, len(list(df))))
    #array = shuffle_array(array)
    #array_resample = np.zeros((int((array.shape[0]-10000)/10), 50, 32))
    array_resample = np.zeros((int(array.shape[0]/50), 50, array.shape[1]))
    for i,j in tqdm(enumerate(range(0,(array.shape[0]-50),50))):
        array_resample[i,:,:] = array[j:j+50,:]

    return array_resample

def shuffle_array(array):
    for i in [11, 34, 56, 3]:
        np.random.seed(i)
        np.random.shuffle(array)

    return array


def min_max_norm_array(x,x2,x3,a,b):

    x3[:, :, 0] = a + ((x3[:, :, 0] - np.min(x[:, :, 0])) * (b - a) / (np.max(x[:, :, 0]) - np.min(x[:, :, 0])))
    for i in range(1, x.shape[2] - 2):
        x3[:, :, i] = a + ((x3[:, :, i] - np.min(x[:, :, i])) * (b - a) / (np.max(x[:, :, i]) - np.min(x[:, :, i])))

    x2[:, :, 0] = a + ((x2[:, :, 0] - np.min(x[:, :, 0])) * (b - a) / (np.max(x[:, :, 0]) - np.min(x[:, :, 0])))
    for i in range(1, x.shape[2] - 2):
        x2[:, :, i] = a + ((x2[:, :, i] - np.min(x[:, :, i])) * (b - a) / (np.max(x[:, :, i]) - np.min(x[:, :, i])))

    x[:,:,0] = a + ((x[:,:,0] - np.min(x[:,:,0]))*(b-a)/(np.max(x[:,:,0])-np.min(x[:,:,0])))
    for i in range(1,x.shape[2]-2):
       x[:,:,i] = a + ((x[:,:,i] - np.min(x[:,:,i]))*(b-a)/(np.max(x[:,:,i])-np.min(x[:,:,i])))

    return x,x2,x3


def data_to_tensors(data,varied):

    ##Under sample train set class 2
        # print('Original dataset shape %s' % Counter(y[:,0]))
        # ind = np.where(y[:,0] == 1)
        # rand_seq = random.sample(list(ind[0]), len(ind[0]))
        # X = np.delete(X,rand_seq[0:int(len(rand_seq)/2)],axis=0)
        # y = np.delete(y,rand_seq[0:int(len(rand_seq)/2)],axis=0)
        # print('New dataset shape %s' % Counter(y[:, 0]))

    y = data[:, :, data.shape[2] - 2].astype(int)
    X = data[:, :, 0:data.shape[2] - 2]
    if varied == True:
        for i in range(y.shape[0]):
            counts = np.bincount(y[i,:])
            y[i,:] = y[i,np.argmax(counts)]

    X = shuffle_array(X)
    y = shuffle_array(y)


    data = torch.utils.data.TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    return data

class MLP(nn.Module):
    def __init__(self,
                 num_features,
                 hidden_layer,
                 layer_num,
                 num_classes: int = 3):
        self.num_features = num_features
        self.hidden_layer = hidden_layer
        self.layers = layer_num
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(self.num_features*50, self.hidden_layer)
        self.linears = nn.ModuleList([nn.Linear(self.hidden_layer, self.hidden_layer) for i in range(self.layers-2)])
        self.fc2 = nn.Linear(int(self.hidden_layer), num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.relu(x)
        for i, layer in enumerate(self.linears):
            x = layer(x)
            x = F.relu(x)
        x = self.fc2(x)

        return x



class CNN(nn.Module):
    def __init__(self,
                 num_features,
                 hidden_layer,
                 layer_num=2,
                 num_classes: int = 3):
        self.multiples = [5, 5, 4, 4, 4, 3, 3, 3, 2, 2, 2, 1, 1, 1]
        self.num_features = num_features
        self.hidden_layer = hidden_layer
        self.layers = layer_num
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=self.num_features, out_channels=self.hidden_layer, kernel_size=5, stride=3)
        self.convs = nn.ModuleList([nn.Conv1d(in_channels=self.hidden_layer, out_channels=int(self.hidden_layer),
                                                kernel_size=2,stride=1) for i in range(self.layers - 1)])

        self.fc1 = nn.Linear(int((self.hidden_layer*self.multiples[self.layers-1])), num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        for i, layer in enumerate(self.convs):
            x = layer(x)
            x = F.relu(x)
        x = F.max_pool1d(x, kernel_size=3)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x


class LSTM(nn.Module):
    def __init__(self,num_features,batch_size,
                 hidden_layer,
                 layer_num = 3,
                 num_classes: int = 3):

        self.hidden_layer = hidden_layer
        self.layers = int(layer_num)
        self.batch_size = batch_size
        self.hidden = self.init_hidden()

        super(LSTM, self).__init__()
        self.lstm1 = nn.LSTM(num_features, self.hidden_layer,self.layers,batch_first = True,dropout=0.1)
        self.fc1 = nn.Linear(50 * self.hidden_layer,num_classes)

    def init_hidden(self):
    # This is what we'll initialise our hidden state as
        return [Variable(torch.zeros(self.layers, self.batch_size, self.hidden_layer,requires_grad=True).double().to("cpu")),\
               Variable(torch.zeros(self.layers, self.batch_size, self.hidden_layer,requires_grad=True).double().to("cpu"))]

    def forward(self,x):
        #ln = len(x[1])
        x, (self.hidden[0],self.hidden[1]) = self.lstm1(x,(self.hidden[0],self.hidden[1]))
        #x = self.lstm1(x)
        for i,hidden in enumerate(self.hidden):
            hidden.detach_()
            hidden = hidden.detach()
            self.hidden[i] = Variable(hidden, requires_grad=True)

        x = F.relu(x)
        x = x.contiguous().view(x.size(0), -1)
        #x = x.contiguous().view(x.shape[1], x.shape[0] * x.shape[2])  # ???
        x = self.fc1(x)
        return x


def predict(net, loader, model):
    device = torch.device("cpu")
    lab = 0
    correct = 0
    total = 0
    one = 0
    two = 0
    three = 0
    f1 = []
    ROC = []
    acc = []
    conf_mtrx = np.zeros((3,3))
    with torch.no_grad():
        for i, data in enumerate(loader):
            inputs, labels = data
            labels = labels.to(device=device, dtype=torch.int64)

            #if model == "LSTM":
            #    aux = inputs.numpy()
            #    aux = aux.swapaxes(0, 1)
            #    inputs = torch.from_numpy(aux)
            if model == "CNN":
                aux = inputs.numpy()
                aux = aux.swapaxes(1, 2)
                inputs = torch.from_numpy(aux)
            elif model == 'Transformer':
                aux = inputs.numpy()
                aux = aux.swapaxes(0, 1)
                inputs = torch.from_numpy(aux)
            inputs = inputs.to(device=device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            labels = labels[:, 0]
            correct += (predicted == labels).sum().item()

            conf_mtrx += confusion_matrix(labels.numpy(), predicted.numpy())

            one +=(predicted == 0).sum().item()
            two +=(predicted == 1).sum().item()
            three +=(predicted == 2).sum().item()
            # lab += (predicted == 0).sum().item()
            f1.append(f1_score(labels, predicted,average='micro'))
            ROC.append(multiclass_roc_auc_score(labels, predicted))
            acc.append(accuracy_score(labels, predicted))
    # print(lab)
    error = (total - correct) / (total + 1)
    # print('Accuracy of the network: %d %%' % (100 * correct / total))

    return error,np.mean(f1),np.mean(ROC),np.mean(acc),one,two,three,conf_mtrx



def reduce_dataset(data,size):
    for i in range(3):
        ind = np.where(data[:,0,data.shape[2]-2] == i)
        data = np.delete(data,ind[0][0:int(len(ind[0])*size)],0)
    print(Counter(data[:,0,data.shape[2]-2]))
    return data


if __name__ == '__main__':


    # set random seed for repetability
    random.seed(30)
    torch.manual_seed(30)
    bs = 800
    models = ['MLP','CNN','LSTM']
    #data_types = ['fx', 'fxL', 'ps', 'pcp1', 'pcp2', 'pcpDC', 'pcpDC2','pcpRV1','pcpRV2','pcpDCRV1','pcpDCRV2']
    data_types = ['fx', 'fxL','ps','pcpPINJ1','pcpPINJ2','pcpDC1','pcpDC2']

    df_results = pd.DataFrame(columns = ['fx ROC','fx acc','fxL ROC','fxL acc','ps ROC','ps acc','pcpPINJ1 ROC','pcpPINJ1 acc',
                                         'pcpPINJ2 ROC','pcpPINJ2 acc','pcpDC1 ROC','pcpDC1 acc','pcpDC2 ROC','pcpDC2 acc'],
                              index = ['fx','fxL','ps','pcpPINJ1',
                                         'pcpPINJ2','pcpDC1','pcpDC2'])



    varied = True


    for model in models:
        if model == 'LSTM':
            bs = 600
        for tag in data_types:
            tag_train = tag
            set_1 = 1
            tag_test = tag
            set_2 = 5

            print(tag)

            features = ['time_stamp','instructions',
            'branches', 'branch-misses', 'branch-load-misses',
            'cache-misses', 'cache-references',
            'cycles', 'context-switches','minor-faults','page-faults',
            'L1-dcache-load-misses','L1-dcache-loads', 'L1-dcache-stores', 'L1-icache-load-misses',
            'LLC-load-misses', 'LLC-store-misses', 'LLC-stores', 'LLC-loads',
            'dTLB-stores', 'dTLB-load-misses', 'dTLB-store-misses',
            'iTLB-loads', 'iTLB-load-misses', 'node-load-misses',
            'node-loads', 'node-store-misses', 'node-stores']


            #features = ['time_stamp', 'instructions']

########################################## TEST PRE-SAVED MODELS #######################################################


            varied = False

            for tag2 in data_types:
                tag_train = tag
                set_1 = 1
                tag_test = tag2
                set_2 = 5

                params = pd.read_csv('/home/francesco/Documents/Thesis_project/Models/' + model + '_' + tag_train + '.csv')
                #min_max = load_data('/home/francesco/Documents/Thesis_project/Models/'+model+'_'+tag_train+'_min_max.npy')

                if model == "CNN":
                    net = CNN(num_features=len(features)-1, hidden_layer=int(params['hidden_layer_size'].iloc[0]),
                              layer_num=params['layers'].iloc[0])
                elif model == "LSTM":
                    net = LSTM(num_features=len(features)-1, batch_size=bs,
                               hidden_layer=params['hidden_layer_size'].iloc[0],
                               layer_num=params['layers'].iloc[0])

                elif model == "MLP":
                    print(int(params['layers'].iloc[0]))
                    net = MLP(num_features=len(features)-1, hidden_layer=int(params['hidden_layer_size'].iloc[0]),layer_num=int(params['layers'].iloc[0]))
                net = net.double()

                net.load_state_dict(torch.load('/home/francesco/Documents/Thesis_project/Models/' + model + '_' + tag_train + '.pt',
                                               map_location=torch.device('cpu')))


                net = net.eval()

                if tag_test in ['fx', 'fxL','ps','pcpPINJ1','pcpPINJ2','pcpDC1','pcpDC2'] or tag in ['fx', 'fxL','ps','pcpPINJ1','pcpPINJ2','pcpDC1','pcpDC2']:
                    varied = True

                df_train,df_val, df_test = training_data_gen.build_training_test_data(features=features,
                                                                               tag_train=tag_train, tag_test=tag_test
                                                                               , step_sz=10
                                                                               , set_1=set_1, set_2=set_2,
                                                                               rng=[-3, 3],
                                                                               varied=varied)


                print(tag_train)
                print(tag_test)
                train_data = df_to_array(df_train)
                val_data = df_to_array(df_val)
                test_data = df_to_array(df_test)



                _,_,test_data = min_max_norm_array(train_data,val_data ,test_data, a=-3, b=3)
                #test_data = shuffle_array(test_data)

                test_data = data_to_tensors(test_data,varied)

                test_loader = torch.utils.data.DataLoader(test_data, batch_size=bs,drop_last=True)  # ,sampler=test_sampler)

                eval = pd.DataFrame(columns=['class_error', 'F1', 'ROC','acc'], index=['a'])
                eval['class_error'].iloc[0],eval['F1'].iloc[0],eval['ROC'].iloc[0],eval['acc'].iloc[0],ones,twos,threes,conf_mtrx = predict(net,test_loader,model)

                print("Classification Error: ",eval['class_error'].iloc[0])
                print("ROC score: ", eval['ROC'].iloc[0])
                print("F1 score: ",eval['F1'].iloc[0])
                print("Class 1: ",ones)
                print("Class 2: ", twos)
                print("Class 3: ", threes)
                print("Confusion matrix:\n",conf_mtrx)
                df_conf_mtrx= pd.DataFrame(conf_mtrx,columns= ['0.5Gbps','5Gbps','9Gbps'],
                                           index=['0.5Gbps','5Gbps','9Gbps'])
                df_conf_mtrx.to_csv('/home/francesco/Documents/Thesis_project/Results/ML/Confusion_matrix/'+ model +'_' + tag_train +'.csv')
                #eval.to_csv('/home/francesco/Documents/Thesis_project/Results/ML/Feature_strength'+ model + '/evaluation_' + tag_train + '_'+tag_test+'.csv')
                df_results.loc[tag_train , tag_test + ' ROC'] = round(eval['ROC'].iloc[0], 3)#("ROC: "+str(round(eval['ROC'].iloc[0], 3)),"F1: " +str(round(eval['F1'].iloc[0], 3)))
                df_results.loc[tag_train , tag_test + ' acc'] = round(eval['acc'].iloc[0], 3)
            df_results.to_csv('/home/francesco/Documents/Thesis_project/Results/ML/'+ model +'/ROC_matrix_new.csv')