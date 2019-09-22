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

import training_data_gen_GPU

from imblearn.under_sampling import RandomUnderSampler
from collections import Counter



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
                 layer_num = int(3),
                 num_classes: int = 3):
        self.num_features = num_features
        self.hidden_layer = hidden_layer
        self.layers = layer_num
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(self.num_features*50, self.hidden_layer)
        self.linears = nn.ModuleList([nn.Linear(self.hidden_layer, self.hidden_layer) for i in range(self.layers-1)])
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
        self.multiples = [5,5,4,4,4,3,3,3,2,2,2,1,1,1]
        self.num_features = num_features
        self.hidden_layer = hidden_layer
        self.layers = layer_num
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=self.num_features, out_channels=self.hidden_layer, kernel_size=5, stride=3)
        self.convs = nn.ModuleList([nn.Conv1d(in_channels=self.hidden_layer, out_channels=int(self.hidden_layer),
                                                kernel_size=2,stride=1) for i in range(self.layers - 1)])


        self.fc1 = nn.Linear(int(self.hidden_layer*self.multiples[self.layers-1]), num_classes)

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
                 hidden_layer,GPU,
                 layer_num = 3,
                 num_classes: int = 3):
        self.GPU = GPU
        self.hidden_layer = hidden_layer
        self.layers = layer_num
        self.batch_size = batch_size
        self.hidden = self.init_hidden()

        super(LSTM, self).__init__()
        self.lstm1 = nn.LSTM(num_features, self.hidden_layer,self.layers,batch_first = True,dropout=0.1)
        self.fc1 = nn.Linear(50 * self.hidden_layer,num_classes)

    def init_hidden(self):
    # This is what we'll initialise our hidden state as
        return [Variable(torch.zeros(self.layers, self.batch_size, self.hidden_layer,requires_grad=True).double().to(self.GPU)),\
               Variable(torch.zeros(self.layers, self.batch_size, self.hidden_layer,requires_grad=True).double().to(self.GPU))]

    def forward(self,x):
        ln = len(x[1])
        x, (self.hidden[0],self.hidden[1]) = self.lstm1(x,(self.hidden[0],self.hidden[1]))

        for i,hidden in enumerate(self.hidden):
            hidden.detach_()
            hidden = hidden.detach()
            self.hidden[i] = Variable(hidden, requires_grad=True)

        x = F.relu(x)
        x = x.contiguous().view(x.size(0), -1)
        #x = x.contiguous().view(x.shape[1], x.shape[0] * x.shape[2])  # ???
        x = self.fc1(x)
        return x

class Transformer(nn.Module):
    def __init__(self,
                 num_features,
                 hidden_layer,
                 batch_size,
                 num_classes: int = 4):
        self.batch_size = batch_size
        self.num_features = num_features
        self.hidden_layer = hidden_layer
        self.target = self.init_target()

        super(Transformer, self).__init__()
        self.transformer = nn.Transformer(d_model=self.num_features,nhead = 27)
        self.fc1 = nn.Linear(self.num_features * 50, num_classes)

    def init_target(self):
    # This is what we'll initialise our hidden state as
        return Variable(torch.zeros(50, self.batch_size, self.num_features,requires_grad=True).double())

    def forward(self, x):
        x = self.transformer(x,self.target)

        self.target = x
        self.target = self.target.detach_()

        x = F.relu(x)
        x = x.view(x.size(1), -1)
        x = self.fc1(x)
        return x



def train(model,GPU,train_loader,val_loader,learn_rate,feature_no,epoch_no,hidden_layer,layer_num,batch_size,test_loader=0,learning_curve = False):
    ## Call instant of model

    if model == "CNN":
        net = CNN(num_features = feature_no,hidden_layer = hidden_layer,layer_num=layer_num)

    elif model == "LSTM":

        net = LSTM(num_features = feature_no,batch_size = batch_size,hidden_layer = hidden_layer,GPU=GPU
                   ,layer_num=layer_num)
        #net.hidden = net.init_hidden(hidden_layer)
    elif model == "MLP":
        net = MLP(num_features = feature_no,hidden_layer = hidden_layer,layer_num=layer_num)

    elif model == "Transformer":
        net = Transformer(num_features = feature_no,hidden_layer = hidden_layer,batch_size = batch_size)
    net = net.double()
    device = torch.device(GPU if torch.cuda.is_available() else "cpu")

    #if torch.cuda.device_count() > 1:
    #    net = nn.DataParallel(net)#,device_ids=['cuda:0','cuda:2','cuda:3'])
    #    print("Using multi GPUs")
    net = net.to(device=device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=learn_rate)  # use adam, it is faster

    train_error = []
    val_error = []
    running_losses = []
    val_loss = []
    test_loss =[]
    ## TRAIN MODEL
    net = net.train()
    for epoch in range(epoch_no):  # loop over the dataset multiple times

        running_loss = 0.0

        for i, data in enumerate(train_loader):
            # train in batches
            inputs, labels = data

            #if model == "LSTM":
            #    aux = inputs.numpy()
            #    aux = aux.swapaxes(0, 1)
            #    inputs = torch.from_numpy(aux)

            if model == 'CNN':
                aux = inputs.numpy()
                aux = aux.swapaxes(1, 2)
                inputs = torch.from_numpy(aux)

            elif model == 'Transformer':
                aux = inputs.numpy()
                aux = aux.swapaxes(0, 1)
                inputs = torch.from_numpy(aux)

            labels = labels.to(device=device, dtype=torch.int64)
            labels = labels[:, 0]
            labels = torch.reshape(labels, (len(labels),))
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize

            inputs = inputs.to(device=device)

            y_pred = net(inputs)

            loss = criterion(y_pred, labels)
            # backwards pass
            loss.backward()

            # update weigths
            optimizer.step()

            # print statistics
            running_loss = loss.item()
            #print(running_loss)
            #if i % 10 == 9:
            if learning_curve == True:
                net.eval()
                val_loss.append(np.mean(calc_loss(model,val_loader,net,criterion,device)))
                test_loss.append(np.mean(calc_loss(model,test_loader,net,criterion,device)))
                net.train()
                #print('[%d, %5d] loss: %.3f' % (
                #epoch + 1, i + 1, running_loss))  # INFO: you need to make sure that this loss decreases
                running_losses.append(running_loss)
                running_loss = 0.0

    #print('Finished Training')
    return net,running_losses,val_loss,test_loss


def calc_loss(model,loader,NN,criterion,device):
    losses = []
    with torch.no_grad():
        for j, data in enumerate(loader):
            input, label = data
            if model == 'CNN':
                aux = input.numpy()
                aux = aux.swapaxes(1, 2)
                input = torch.from_numpy(aux)
            elif model == 'Transformer':
                aux = inputs.numpy()
                aux = aux.swapaxes(0, 1)
                inputs = torch.from_numpy(aux)

            label = label.to(device=device, dtype=torch.int64)
            label = label[:, 0]
            label = torch.reshape(label, (len(label),))
            input = input.to(device=device)
            pred = NN(input)
            loss = criterion(pred, label)

            #r_loss += loss.item()
            #if j % 10 == 9:
            losses.append(loss.item())# / 10)
            #r_loss = 0.0
    return losses


def predict(net, loader, model,GPU):
    device = torch.device(GPU)
    lab = 0
    correct = 0
    total = 0
    one = 0
    two = 0
    three = 0
    f1 = []
    ROC = []
    acc=[]
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

            #conf_mtrx += confusion_matrix(labels.numpy(), predicted.numpy())

            one +=(predicted == 0).sum().item()
            two +=(predicted == 1).sum().item()
            three +=(predicted == 2).sum().item()
            # lab += (predicted == 0).sum().item()
            f1.append(f1_score(labels.cpu(), predicted.cpu(),average='micro'))
            ROC.append(multiclass_roc_auc_score(labels.cpu(), predicted.cpu()))
            acc.append(accuracy_score(labels.cpu(), predicted.cpu()))
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


    models = [sys.argv[1]]
    bs = int(sys.argv[2])
    epoch = int(sys.argv[3])
    GPU = sys.argv[4]
    data_types = [sys.argv[5]]
    hidden_layers = [int(sys.argv[6])]
    layer_nums = [int(sys.argv[7])]
    learning_rates = [float(sys.argv[8])]
    V = sys.argv[9]

    #path_training = sys.argv[3]
    #path_test = sys.arg[4]

    # set random seed for repetability
    random.seed(30)
    torch.manual_seed(30)

    #epoch = 5
    #bs = 800  ## TODO: pass this as a parameter

    #learning_rates = [1e-2]

    #hidden_layers = [10]

    #hidden_layers = [102400,51200,25600]
    #models = ['MLP']  ## TODO: pass this as a parameter
    #data_types = ['fx', 'fxL', 'ps', 'pcp1', 'pcp2', 'pcpDC', 'pcpDC2','pcpRV1','pcpRV2','pcpDCRV1','pcpDCRV2']
    #data_types = ['ps']#,'pcpPINJ1','pcpPINJ2','pcpDC1','pcpDC2']

    #data_types = ['pcp1']
    df_results = pd.DataFrame(columns = ['fx', 'fxL', 'ps', 'pcp1', 'pcp2', 'pcpDC', 'pcpDC2','pcpRV1','pcpRV2','pcpDCRV1','pcpDCRV2'],
                              index = ['fx', 'fxL', 'ps', 'pcp1', 'pcp2', 'pcpDC', 'pcpDC2','pcpRV1','pcpRV2','pcpDCRV1','pcpDCRV2'])


    varied = True
    train_models = True

    for model in models:
        print(model)

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


            for f in range(1,len(features)):
                features = ['time_stamp', 'instructions',
                            'branches', 'branch-misses', 'branch-load-misses',
                            'cache-misses', 'cache-references',
                            'cycles', 'context-switches', 'minor-faults', 'page-faults',
                            'L1-dcache-load-misses', 'L1-dcache-loads', 'L1-dcache-stores', 'L1-icache-load-misses',
                            'LLC-load-misses', 'LLC-store-misses', 'LLC-stores', 'LLC-loads',
                            'dTLB-stores', 'dTLB-load-misses', 'dTLB-store-misses',
                            'iTLB-loads', 'iTLB-load-misses', 'node-load-misses',
                            'node-loads', 'node-store-misses', 'node-stores']

                leave_out = features[f]
                if features[f] in features: features.remove(features[f])
            #features = ['time_stamp', 'instructions']


    ####################################### TRAIN MODELS AGAINST VALIDATION SET ###########################################

                if train_models == True:
                    df_train,df_val, df_test = training_data_gen_GPU.build_training_test_data(features=features,
                                                                                   tag_train=tag_train, tag_test=tag_test
                                                                                   , step_sz=10
                                                                                   , set_1=1, set_2=set_2,
                                                                                   rng=[-3, 3],
                                                                                   varied=varied)
                    for i in range(2,4):
                        aux,_,_ = training_data_gen_GPU.build_training_test_data(features=features,
                                                                                       tag_train=tag_train,
                                                                                       tag_test=tag_test
                                                                                       , step_sz=10
                                                                                       , set_1=i, set_2=set_2,
                                                                                       rng=[-3, 3],
                                                                                       varied=varied)
                        df_train = df_train.append(aux)

                    train_data = df_to_array(df_train)
                    val_data =  df_to_array(df_val)
                    test_data = df_to_array(df_test)
                    # for i in range(2800,2936):
                    #     plt.plot(train_data[i,:,0])
                    #     print(train_data[i,:,27])
                    #     plt.show()
                    train_data,val_data,test_data  = min_max_norm_array(train_data, val_data,test_data, a=-3, b=3)
                    #train_data, test_data = min_max_norm_array(train_data, test_data, a=-3, b=3)


                    train_data = data_to_tensors(train_data,varied)
                    val_data = data_to_tensors(val_data, varied)
                    test_data = data_to_tensors(test_data, varied)


                    val_loader = torch.utils.data.DataLoader(val_data, batch_size=bs,drop_last=True)  # ,sampler=val_sampler)
                    train_loader = torch.utils.data.DataLoader(train_data, batch_size=bs,drop_last=True,shuffle=False)



                    if V == 'v':
                        epochs = [15,20,21,22,23,24,26,50,100]
                        #hidden_layers = [200,100,50]
                        #layer_nums = [1,2,3,4,6,8,10]
                        layer_nums = [6, 8, 10]

                        #learning_rates = [1e-2, 1e-3]
                        acc_scores = np.zeros((len(layer_nums), len(hidden_layers),len(epochs)))
                        ROC_scores = np.zeros((len(layer_nums), len(hidden_layers),len(epochs)))
                        F1_scores = np.zeros((len(layer_nums), len(hidden_layers),len(epochs)))

                        for t, layer_num in enumerate(layer_nums):
                            print("Learning rate: ", learning_rates[0])
                            for h,hidden_layer in enumerate(hidden_layers):
                                print("Hidden layer size: ", hidden_layer)
                                for e, epoch in enumerate(epochs):
                                    fin = 0
                                    #while(1):
                                        #for GPU in ["cuda:0","cuda:1","cuda:2","cuda:3"]:

                                            #try:
                                    net,running_losses,_,_ = train(model,GPU,train_loader,val_loader,learning_rates[0],len(features)-1,
                                                                   epoch,hidden_layer,layer_num,bs)
                                                #fin = 1
                                                #break
                                            #except RuntimeError:
                                                #print(GPU," in use")
                                        #if fin == 1:
                                            #break
                                    net = net.eval()
                                    eval = {}
                                    aux,eval['F1'], eval['ROC'], eval['acc'], ones, twos, threes, conf_mtrx= predict(net, val_loader, model,GPU)
                                    acc_scores[t, h, e] = eval['acc']
                                    ROC_scores[t, h, e] = eval['ROC']
                                    F1_scores[t, h, e] = eval['F1']

                                    net = net.train()

                        print("Training, max ROC score: ", np.max(ROC_scores))
                        val_ROC = np.max(ROC_scores)
                        val_acc = np.max(acc_scores)
                        print("Training, max ROC score: ", np.max(acc_scores))
                        params_ind = np.unravel_index(np.argmax(ROC_scores, axis=None), ROC_scores.shape)
                        params_ind = np.unravel_index(np.argmax(acc_scores, axis=None), acc_scores.shape)
                        print('hidden layer: ',hidden_layers[params_ind[1]])
                        #print('learning rate: ', learning_rates[params_ind[0]])
                        #print("Class 1: ", ones)
                        #print("Class 2: ", twos)
                        #print("Class 3: ", threes)
                        epoch = epochs[params_ind[2]]

                        print(1)

    ################################## FINAL TRAIN MODEL WITH BEST HYPERPARAMETERS #########################################
                    #test_data = data_to_tensors(test_data,varied)
                        test_loader = torch.utils.data.DataLoader(test_data, batch_size=bs,drop_last=True)  # ,sampler=test_sampler)
                        net, loss,val_loss,test_loss = train(model,GPU,train_loader,val_loader, learning_rates[0],len(features)-1,epoch,
                                                            hidden_layer=hidden_layers[params_ind[1]],layer_num=layer_nums[params_ind[0]],
                                                            batch_size=bs,test_loader=test_loader,learning_curve=True)

                        fig, axes = plt.subplots(1, 1, figsize=(8, 6))
                        axes.plot(loss, label="Training loss")
                        axes.plot(val_loss, label="Validation loss")
                        axes.plot(test_loss, label="Test loss")
                        axes.set_ylabel('loss', fontsize=15)
                        axes.set_xlabel('batches', fontsize=15)
                        axes.set_title(model + ' learning curves ' + tag_train, fontsize=15)
                        axes.tick_params(labelsize=15)
                        plt.legend()
                        # plt.show()

                        net = net.eval()

                        torch.save(net.state_dict(),
                                   '/home/infres/cshelbourne/Models_fs/' + model + '_' + tag_train +'.pt')
                        # np.save('/home/francesco/Documents/Thesis_project/Models/'+model+'_'+tag_train+'_min_max.npy',min_max)
                        h_params = pd.DataFrame(columns=['learning_rate', 'hidden_layer_size','layers'], index=['a'])
                        h_params.iloc[0] = [learning_rates[0], hidden_layers[params_ind[1]],layer_nums[params_ind[0]]]
                        h_params.to_csv('/home/infres/cshelbourne/Models_fs/' + model + '_' + tag_train + '.csv')

                        # plt.savefig(
                        #     '/home/infres/cshelbourne/Results/Learning_curves/' + model + '/' + model + '_' + tag_train + '_LR_' + str(
                        #         learning_rates[0]) + '_HL_' + str(hidden_layers[params_ind[1]]) +'_ROC_'+str(val_ROC)+'_layers_'+str(layer_nums[params_ind[0]])+'_epoch_'+str(epoch)+'.png')
                        plt.savefig(
                            '/home/infres/cshelbourne/Results/Learning_curves/' + model + '/' + model + '_' + tag_train + '_LR_' + str(
                                learning_rates[0]) + '_HL_' + str(hidden_layers[params_ind[1]]) + '_acc_' + str(
                                val_acc) + '_layers_' + str(layer_nums[params_ind[0]]) + '_epoch_' + str(epoch) + '.png')
                        #print("Training, max ROC score: ", np.max(ROC_scores))
                        print("Training, max acc score: ", np.max(acc_scores))

                    else:

                        test_loader = torch.utils.data.DataLoader(test_data, batch_size=bs,
                                                                  drop_last=True)  # ,sampler=test_sampler)
                        net, loss, val_loss, test_loss = train(model, GPU, train_loader, val_loader,
                                                               learning_rates[0], len(features) - 1, epoch,
                                                               hidden_layer=hidden_layers[0],layer_num=layer_nums[0], batch_size=bs,
                                                               test_loader=test_loader, learning_curve=True)

                        net = net.eval()

                        eval = {}
                        aux, eval['F1'], eval['ROC'],eval['acc'], ones, twos, threes, conf_mtrx = predict(net, val_loader, model, GPU)
                        #ROC_list.append(eval['ROC'])
                        #F1_list.append(eval['F1'])
                        #val_error.append(aux)
                        net = net.train()
                        print("Training, max ROC score: ", eval['ROC'])
                        print("Training, max acc score: ", eval['acc'])
                        val_ROC = eval['ROC']
                        val_acc = eval['acc']

                        fig, axes = plt.subplots(1, 1, figsize=(8, 6))
                        axes.plot(loss, label="Training loss")
                        axes.plot(val_loss, label="Validation loss")
                        axes.plot(test_loss, label="Test loss")
                        axes.set_ylabel('loss', fontsize=15)
                        axes.set_xlabel('batches', fontsize=15)
                        axes.set_title(model + ' learning curves ' + tag_train, fontsize=15)
                        axes.tick_params(labelsize=15)
                        plt.legend()
                        #plt.show()

                        print("saving model " +leave_out)
                        torch.save(net.state_dict(), '/home/infres/cshelbourne/Models_fs/'+model+'_'+tag_train+'_'+leave_out+'.pt')
                        #np.save('/home/francesco/Documents/Thesis_project/Models/'+model+'_'+tag_train+'_min_max.npy',min_max)
                        h_params = pd.DataFrame(columns=['learning_rate', 'hidden_layer_size','layers'], index=['a'])
                        h_params.iloc[0] = [learning_rates[0],hidden_layers[0],layer_nums[0]]
                        h_params.to_csv('/home/infres/cshelbourne/Models_fs/'+ model + '_' + tag_train+'_'+leave_out+'.csv')


                        #plt.savefig('/home/infres/cshelbourne/Results/Learning_curves/' + model + '/' + model + '_' + tag_train +'_LR_'+str(learning_rates[0])+'_HL_'+str(hidden_layers[0])+'_ROC_'+str(val_ROC)+'_layers_'+str(layer_num)+'_epoch_'+str(epoch)+'.png')
                        plt.savefig(
                            '/home/infres/cshelbourne/Results/Learning_curves/' + model + '/' + model + '_' + tag_train +'_'+leave_out+ '_LR_' + str(
                                learning_rates[0]) + '_HL_' + str(hidden_layers[0]) + '_acc_' + str(
                                val_ROC) + '_layers_' + str(layer_nums[0]) + '_epoch_' + str(epoch) + '.png')

                    features = ['time_stamp', 'instructions',
                                'branches', 'branch-misses', 'branch-load-misses',
                                'cache-misses', 'cache-references',
                                'cycles', 'context-switches', 'minor-faults', 'page-faults',
                                'L1-dcache-load-misses', 'L1-dcache-loads', 'L1-dcache-stores',
                                'L1-icache-load-misses',
                                'LLC-load-misses', 'LLC-store-misses', 'LLC-stores', 'LLC-loads',
                                'dTLB-stores', 'dTLB-load-misses', 'dTLB-store-misses',
                                'iTLB-loads', 'iTLB-load-misses', 'node-load-misses',
                                'node-loads', 'node-store-misses', 'node-stores']



