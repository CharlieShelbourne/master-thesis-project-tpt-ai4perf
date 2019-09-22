import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def build_df(raw_data,features):
    data = pd.DataFrame()
    time_stamps = raw_data['Count'].unique()
    data['time_stamp'] = raw_data['Count'].unique()

    for label in raw_data['Labels'].iloc[0:31]:
        aux = pd.Series(raw_data['Values'][raw_data['Labels'] == label], name=label)
        aux = aux.reset_index(drop=True)
        data = data.join(aux, how='right')

    for column in list(data):
        data[column] = pd.to_numeric(data[column], errors='coerce').fillna(0).astype(np.int64)

    data = data[features]

    return data


def build_training(files,features):
    window_size = 100

    if len(files) == 1:
        training_data = build_df(pd.read_csv('/home/francesco/charlie_thesis_dir/generated_traces/' + files[0] + '.csv'),
                                 features)

        training_data = training_data.rolling(window_size).mean()
        training_data = training_data[window_size:len(training_data)]
    else:
        for i, file in enumerate(files):
            if i == 0:
                training_data = build_df(pd.read_csv('/home/francesco/charlie_thesis_dir/generated_traces/' + file + '.csv'),
                                         features)

                training_data = training_data.rolling(window_size).mean()
                training_data = training_data[window_size:len(training_data)]

                training_data['label'] = i
                training_data = training_data.reset_index(drop=True)
                #print(training_data.head())
            else:
                data = build_df(pd.read_csv('/home/francesco/charlie_thesis_dir/generated_traces/' + file + '.csv'),
                                features)
                data = data.rolling(window_size).mean()
                data = data[window_size:len(data)]

                data['label'] = i
                #print(data.head())
                training_data = pd.concat([training_data, data])
                training_data = training_data.reset_index(drop=True)
        print(training_data.shape)
    return training_data


def two_norm_array(x):
    for i in range(x.shape[2]-1):
       x[:,:,i] = x[:,:,i] / np.linalg.norm(x[:,:,i], ord=2, axis=1, keepdims=True)
    return x

def min_max_norm_array(x,x2,a,b):
    min_max = np.zeros((x.shape[2],2))

    x2[:, :, 0] = a + ((x2[:, :, 0] - np.min(x[:, :, 0])) * (b - a) / (np.max(x[:, :, 0]) - np.min(x[:, :, 0])))
    min_max[0, 0] = np.min(x[:, :, 0])
    min_max[0, 1] = np.max(x[:, :, 0])
    for i in range(1, x.shape[2] - 2):
        x2[:, :, i] = a + ((x2[:, :, i] - np.min(x[:, :, i])) * (b - a) / (np.max(x[:, :, i]) - np.min(x[:, :, i])))
        # x2[:, :, i] = a + ((x2[:, :, i] - mn) * (b - a) / (mx - mn))
        min_max[i,0] = np.min(x[:, :, i])
        min_max[i,1] = np.max(x[:, :, i])
    x[:,:,0] = a + ((x[:,:,0] - np.min(x[:,:,0]))*(b-a)/(np.max(x[:,:,0])-np.min(x[:,:,0])))
    for i in range(1,x.shape[2]-2):
       x[:,:,i] = a + ((x[:,:,i] - np.min(x[:,:,i]))*(b-a)/(np.max(x[:,:,i])-np.min(x[:,:,i])))
       #x[:, :, i] = a + ((x[:, :, i] - mn) * (b - a) / (mx - mn))

    return x,x2,min_max


def shuffle_array(array):
    for i in [11, 34, 56, 3]:
        np.random.seed(i)
        np.random.shuffle(array)

    return array


def df_to_array(df,seq_len,step_sz):
    cut_off = (len(df) % 100)
    df = df.iloc[0:len(df) - cut_off]

    feat_no = len(list(df))-1

    #df['instructions'].plot()
    #plt.show()

    #df_new = pd.DataFrame(columns = list(df))

    array = np.asarray(df[list(df)[1:len(list(df))]])#.reshape((int(df.shape[0] / 1), 1, len(list(df))))
    array_resample = np.zeros((int(array.shape[0]/step_sz), seq_len, feat_no+1))
    for i,j in tqdm(enumerate(range(0,array.shape[0]-seq_len,step_sz))):
        array_resample[i,:,0:feat_no] = array[j:j+seq_len,:]
        array_resample[i,:,feat_no] = i

    array = array_resample.copy().reshape((array_resample.shape[0] * array_resample.shape[1] , array_resample.shape[2]))
    array_resample = shuffle_array(array_resample)
    #array = array[:,:,0:array.shape[2]-1]
    aux = pd.DataFrame(columns=df.columns[1:len(df.columns)])
    aux['seq_id'] = pd.Series()

    df_new = pd.DataFrame(array,columns=aux.columns)

    return array_resample, df_new

def pre_process_df(df,step_sz,seq_len):
    rates_array = np.asarray(
        [2,1,2,0,2,2,0,0,0,1,1,2,0,0,1,0,2,0,2,0,0,2,2,1,1,2,0,2,2,2,1,1,0,2,0,1,2,2,0,2,1,1,1,0,2,0,1,0,2,2,0,0,2,1,1,
         0,1,1,0,0,1,0,1,1,1,0,1,0,2,2,2,0,2,1,0,0,0,1,1,1,2,2,0,2,1,0,1,0,2,2,2,1,1,1,0])#,1,1,2])

    cutoff = 420
    window = 310
    df = df.iloc[cutoff:len(df) ]
    df= df.reset_index(drop=True)
    df = df.iloc[0:window*int(len(df)/window)]
    df = df.reset_index(drop=True)
    #df['instructions'].plot()
    #plt.show()
    df['label'] = 3
    start = 0
    stop = window
    for label in rates_array:
        df['label'].iloc[start:stop] = label
        start = start + window
        stop = stop + window
        #print(label)
    #cut_off = (len(df) % 100)
    #df = df.iloc[0:len(df) - cut_off]


    feat_no = len(list(df)) - 1


    # df_new = pd.DataFrame(columns = list(df))

    array = np.asarray(df)  # .reshape((int(df.shape[0] / 1), 1, len(list(df))))
    array_resample = np.zeros((int((array.shape[0] / step_sz)-(seq_len/step_sz))+1, seq_len, feat_no + 2))
    for i, j in tqdm(enumerate(range(0, (array.shape[0]-seq_len)+step_sz, step_sz))):

        array_resample[i, :, 0:feat_no +1] = array[j:j + seq_len, :]
        array_resample[i, :, feat_no +1] = i

    array = array_resample.copy().reshape((array_resample.shape[0] * 50, array_resample.shape[2]))
    aux = pd.DataFrame(columns=df.columns)
    aux['seq_id'] = pd.Series()

    df_new = pd.DataFrame(array, columns=aux.columns)
    #df_new['label'] = 3
    # lab = 0
    # seg = 1550
    # #seg = 3000
    #
    # count_0 = 0
    # count_1 = 0
    # count_2 = 0
    #
    # df_new['label'].iloc[0:2200] = 0
    # low = True


    # for i in np.arange(2200,df_new.shape[0],seg):
    #     if lab == 0:
    #         df_new['label'].iloc[i:i+seg]= 0
    #         lab = 1
    #         low = True
    #         count_0 = count_0 + 1
    #     elif lab == 1:
    #
    #         df_new['label'].iloc[i:i+seg] = 1
    #         if low == True:
    #             lab = 2
    #         else:
    #             lab = 0
    #         count_1 = count_1 + 1
    #     elif lab == 2:
    #
    #         df_new['label'].iloc[i:i+seg] = 2
    #         lab = 1
    #         low = False
    #         count_2 = count_2 + 1
    ## Add sequence numbers
    # col = pd.Series(name="seq_id")
    # df_new = df_new.join(col)
    # print("Class 1 num: ",count_0*seg)
    # print("Class 2 num: ", count_1*seg)
    # print("Class 3 num: ", count_2*seg)

    df_new = df_new.drop(['time_stamp'], axis=1)

    return df_new

##### Build Training set
def build_training_test_data(features,
                             seq_len=50,step_sz=10,
                             tag_train = 'ps',tag_test ='pcp',
                             set_1=1,set_2=1,
                             rng =[-3,3],
                             varied=False):
    if varied == True:
        window_size = 100

        if tag_train in ['fx', 'fxL','ps','pcpPINJ1','pcpPINJ2','pcpDC1','pcpDC2'] and tag_test in ['fx', 'fxL','ps','pcpPINJ1','pcpPINJ2','pcpDC1','pcpDC2'] :

            file = '/home/francesco/charlie_thesis_dir/generated_traces/varied_rates/'+ tag_train + '_trainingData_Rvaried'+str(set_1)+'.csv'
            df_train = build_df(pd.read_csv(file),features)
            df_train = df_train.rolling(window_size).mean()
            df_train = df_train[window_size:len(df_train)]

            file = '/home/francesco/charlie_thesis_dir/generated_traces/varied_rates/' + tag_test + '_trainingData_Rvaried' + str(4) + '.csv'
            df_val = build_df(pd.read_csv(file), features)
            df_val = df_val.rolling(window_size).mean()
            df_val = df_val[window_size:len(df_val)]

            file = '/home/francesco/charlie_thesis_dir/generated_traces/varied_rates/' + tag_test + '_trainingData_Rvaried' + str(5) + '.csv'
            df_test = build_df(pd.read_csv(file), features)
            df_test = df_test.rolling(window_size).mean()

            df_test = df_test[window_size:len(df_test)]

            df_train = pre_process_df(df_train, seq_len=seq_len, step_sz=step_sz)
            df_val = pre_process_df(df_val, seq_len=seq_len, step_sz=step_sz)
            df_test = pre_process_df(df_test, seq_len=seq_len, step_sz=step_sz)
        elif tag_test in ['pcpRV1','pcpRV2','pcpDCRV1','pcpDCRV2']:
            files = [tag_train + '_500mbps_trainData' + str(set_1), tag_train + '_5000mbps_trainData' + str(set_1),
                     tag_train + '_9000mbps_trainData' + str(set_1)]
            df_train = build_training(files, features)

            files = [tag_train + '_500mbps_trainData' + str(4), tag_train + '_5000mbps_trainData' + str(4),
                     tag_train + '_9000mbps_trainData' + str(4)]
            df_val = build_training(files, features)

            file = '/home/francesco/charlie_thesis_dir/generated_traces/varied_rates/'+ tag_test + '_trainingData_Rvaried' + str(5)+'.csv'
            df_test = build_df(pd.read_csv(file),features)
            df_test = df_test.rolling(window_size).mean()

            df_test = df_test[window_size:len(df_test)]

            df_test = pre_process_df(df_test,seq_len=seq_len,step_sz=step_sz)
        else:
            file = '/home/francesco/charlie_thesis_dir/generated_traces/varied_rates/' + tag_train + '_trainingData_Rvaried' + str(
                set_1) + '.csv'
            df_train = build_df(pd.read_csv(file), features)
            df_train = df_train.rolling(window_size).mean()
            df_train = df_train[window_size:len(df_train)]

            file = '/home/francesco/charlie_thesis_dir/generated_traces/varied_rates/' + tag_train + '_trainingData_Rvaried' + str(
                4) + '.csv'
            df_val = build_df(pd.read_csv(file), features)
            df_val = df_train.rolling(window_size).mean()
            df_val = df_train[window_size:len(df_train)]

            files = [tag_test + '_500mbps_trainData' + str(5), tag_test + '_5000mbps_trainData' + str(5),
                     tag_test + '_9000mbps_trainData' + str(5)]
            df_test = build_training(files, features)

            array_test, df_test = df_to_array(df_test, seq_len, step_sz)
    else:
        files = [tag_train+'_500mbps_trainData'+str(set_1),tag_train+'_5000mbps_trainData'+str(set_1),tag_train+'_9000mbps_trainData'+str(set_1)]
        df_train= build_training(files,features)

        files = [tag_train + '_500mbps_trainData' + str(4), tag_train + '_5000mbps_trainData' + str(4),
                 tag_train + '_9000mbps_trainData' + str(4)]
        df_val = build_training(files, features)

        files = [tag_test+'_500mbps_trainData'+str(5),tag_test+'_5000mbps_trainData'+str(5),tag_test+'_9000mbps_trainData'+str(5)]
        df_test= build_training(files,features)

        array_train, df_train = df_to_array(df_train,seq_len,step_sz)

        array_val, df_val = df_to_array(df_val, seq_len, step_sz)

        array_test, df_test = df_to_array(df_test,seq_len,step_sz)

        array_train,array_test,min_max = min_max_norm_array(array_train,array_test, rng[0],rng[1])

#for i in range(10):
#        plt.plot(array_train[i, :, 0])
#        plt.show()
#        plt.plot(array_train[i, :, 1])
#        plt.show()
#        plt.plot(array_train[i, :, 2])
#        plt.show()
#        print(array_train[i, 0, 26])


        #np.save('/home/francesco/Documents/Thesis_project/Data/training_data.npy',array_train)
        #np.save('/home/francesco/Documents/Thesis_project/Data/test_data.npy',array_test)
    return df_train,df_val,df_test#,min_max

if __name__ == '__main__':

    build_training_test_data()





