""" Make datasets

"""

__author__ = 'Atoosa Parsa'
__copyright__ = 'Copyright 2024, Atoosa Parsa'
__credits__ = 'Atoosa Parsa'
__license__ = 'MIT License'
__version__ = '2.0.0'
__maintainer__ = 'Atoosa Parsa'
__email__ = 'atoosa.parsa@gmail.com'
__status__ = "Dev"


import torch
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedKFold

def loadData(dev, params, shuffle=True):

    if params['dataset'] == "simple":
        window_length = params['window_length']
        batchSize = params['batchSize']

        ## build a simple dataset
        timesteps = params['timeSteps']
        At = 1e-3     # excitation amplitude

        t = np.arange(0, timesteps*params['dt'], params['dt'])
        Nt = timesteps
        data_all = []
        target_all = [] # index of the probe for the desired output
        n_classes = 2 # number of probes for the output
        freqs = [7, 15]
        for f in freqs:
            data_all.append([At*np.sin(f*params['dt']*np.arange(Nt))])        
            if f == 7:
                target_all.append(0)
            if f == 15:
                target_all.append(1)

        train_dl = make_dataset(dev, data_all, n_classes, target_all, window_length, batch_size=batchSize)
        test_dl = None

    elif params['dataset'] == "and":
        window_length = params['window_length']
        batchSize = params['batchSize']

        ## build a simple dataset
        timesteps = params['timeSteps']
        At = 1e-3     # excitation amplitude

        t = np.arange(0, timesteps*params['dt'], params['dt'])
        Nt = timesteps
        data_all = []
        target_all = [] # index of the probe for the desired output
        n_classes = 1 # number of probes for the output
        f = 7
        
        data_all.append([np.zeros(Nt), At*np.sin(f*params['dt']*np.arange(Nt))])
        target_all.append([np.zeros(Nt)])
        
        data_all.append([At*np.sin(f*params['dt']*np.arange(Nt)), np.zeros(Nt)])
        target_all.append([np.zeros(Nt)])
        
        data_all.append([At*np.sin(f*params['dt']*np.arange(Nt)), At*np.sin(f*params['dt']*np.arange(Nt))])
        target_all.append([At*np.sin(f*params['dt']*np.arange(Nt))])
        
        train_dl = make_dataset2(dev, data_all, n_classes, target_all, window_length, batchSize, shuffle)
        test_dl = None

    elif params['dataset'] == "and3":
        window_length = params['window_length']
        batchSize = params['batchSize']

        ## build a simple dataset
        timesteps = params['timeSteps']
        At = 1e-3     # excitation amplitude

        t = np.arange(0, timesteps*params['dt'], params['dt'])
        Nt = timesteps
        data_all = []
        target_all = [] # index of the probe for the desired output
        n_classes = 1 # number of probes for the output
        f = 15
        
        # case 01
        data_all.append([At*np.sin(f*params['dt']*np.arange(Nt)), np.zeros(Nt)])
        target_all.append([np.zeros(Nt)])        
        
        # case 11
        data_all.append([At*np.sin(f*params['dt']*np.arange(Nt)), \
                         At*np.sin(f*params['dt']*np.arange(Nt))])
        target_all.append([At*np.sin(f*params['dt']*np.arange(Nt))])
        
        # case 10
        data_all.append([np.zeros(Nt), At*np.sin(f*params['dt']*np.arange(Nt))])
        target_all.append([np.zeros(Nt)])

        # auxiliary cases
        data_all.append([At*np.sin(20*params['dt']*np.arange(Nt)), \
                         At*np.sin(20*params['dt']*np.arange(Nt))])
        target_all.append([np.zeros(Nt)])
        
        data_all.append([At*np.sin(30*params['dt']*np.arange(Nt)), \
                         At*np.sin(30*params['dt']*np.arange(Nt))])
        target_all.append([np.zeros(Nt)])

        train_dl = make_dataset2(dev, data_all, n_classes, target_all, window_length, \
                                 batchSize, shuffle)
        test_dl = None
    
    elif params['dataset'] == "and2":
        window_length = params['window_length']
        batchSize = params['batchSize']

        ## build a simple dataset
        timesteps = params['timeSteps']
        At = 1e-3     # excitation amplitude

        t = np.arange(0, timesteps*params['dt'], params['dt'])
        Nt = timesteps
        data_all = []
        target_all = [] # index of the probe for the desired output
        n_classes = 1 # number of probes for the output
        f = 15

        data_all.append([np.zeros(Nt), At*np.sin(f*params['dt']*np.arange(Nt))])
        target_all.append([np.zeros(Nt)])
        
        data_all.append([At*np.sin(f*params['dt']*np.arange(Nt)), np.zeros(Nt)])
        target_all.append([np.zeros(Nt)])
        
        data_all.append([At*np.sin(f*params['dt']*np.arange(Nt)), At*np.sin(f*params['dt']*np.arange(Nt))])
        target_all.append([At*np.sin(f*params['dt']*np.arange(Nt))])
        
        train_dl = make_dataset2(dev, data_all, n_classes, target_all, window_length, batchSize, shuffle)
        test_dl = None
        
    elif params['dataset'] == "and4":
        window_length = params['window_length']
        batchSize = params['batchSize']

        ## build a simple dataset
        timesteps = params['timeSteps']
        At = 1e-3     # excitation amplitude

        t = np.arange(0, timesteps*params['dt'], params['dt'])
        Nt = timesteps
        data_all = []
        target_all = [] # index of the probe for the desired output
        n_classes = 1 # number of probes for the output
        f = 15

        data_all.append([np.zeros(Nt), At*np.sin(f*params['dt']*np.arange(Nt))])
        target_all.append([np.zeros(Nt)])
        
        data_all.append([At*np.sin(f*params['dt']*np.arange(Nt)), np.zeros(Nt)])
        target_all.append([np.zeros(Nt)])
        
        data_all.append([At*np.sin(f*params['dt']*np.arange(Nt)), At*np.sin(f*params['dt']*np.arange(Nt))])
        target_all.append([At*np.sin(f*params['dt']*np.arange(Nt))])
        
        data_all.append([At*np.sin(f*params['dt']*np.arange(Nt)), At*np.sin(f*params['dt']*np.arange(Nt))])
        target_all.append([At*np.sin(f*params['dt']*np.arange(Nt))])
        
        train_dl = make_dataset2(dev, data_all, n_classes, target_all, window_length, batchSize, shuffle)
        test_dl = None
        
    elif params['dataset'] == "xor":
        window_length = params['window_length']
        batchSize = params['batchSize']

        ## build a simple dataset
        timesteps = params['timeSteps']
        At = 1e-3     # excitation amplitude

        t = np.arange(0, timesteps*params['dt'], params['dt'])
        Nt = timesteps
        data_all = []
        target_all = [] # index of the probe for the desired output
        n_classes = 1 # number of probes for the output
        f = 10

        data_all.append([np.zeros(Nt), At*np.sin(f*params['dt']*np.arange(Nt))])
        target_all.append([At*np.sin(f*params['dt']*np.arange(Nt))])
        
        data_all.append([At*np.sin(f*params['dt']*np.arange(Nt)), np.zeros(Nt)])
        target_all.append([At*np.sin(f*params['dt']*np.arange(Nt))])
        
        data_all.append([At*np.sin(f*params['dt']*np.arange(Nt)), At*np.sin(f*params['dt']*np.arange(Nt))])
        target_all.append([np.zeros(Nt)])

        train_dl = make_dataset2(dev, data_all, n_classes, target_all, window_length, batchSize, shuffle)
        test_dl = None
        
    elif params['dataset'] == "xor2":
        window_length = params['window_length']
        batchSize = params['batchSize']

        ## build a simple dataset
        timesteps = params['timeSteps']
        At = 1e-3     # excitation amplitude

        t = np.arange(0, timesteps*params['dt'], params['dt'])
        Nt = timesteps
        data_all = []
        target_all = [] # index of the probe for the desired output
        n_classes = 1 # number of probes for the output
        f = 15
        
        data_all.append([np.zeros(Nt), At*np.sin(f*params['dt']*np.arange(Nt))])
        target_all.append([At*np.sin(f*params['dt']*np.arange(Nt))])

        data_all.append([At*np.sin(f*params['dt']*np.arange(Nt)), np.zeros(Nt)])
        target_all.append([At*np.sin(f*params['dt']*np.arange(Nt))])
        
        data_all.append([At*np.sin(f*params['dt']*np.arange(Nt)), At*np.sin(f*params['dt']*np.arange(Nt))])
        target_all.append([np.zeros(Nt)])
        
        train_dl = make_dataset2(dev, data_all, n_classes, target_all, window_length, batchSize, shuffle)
        test_dl = None
        
    elif dataset == "vowels":
        sampling_rate = params['sampling_rate']
        seed = params['seed']
        window = params['window']
        window_size = params['window']
        batch_size = params['batchSize']
        
        vwl = ['ei', 'oa']  # 'iy'
        N_classes = len(vwl)

        X, Y, _ = vowels.load_all_vowels(vwl, "both", sr=sampling_rate, normalize=True, max_samples=30, random_state=seed)

        skf = StratifiedKFold(n_splits=3, random_state=seed, shuffle=True)
        samps = [y.argmax().item() for y in Y]

        train_dl = []
        test_dl = []
        for num, (train_index, test_index) in enumerate(skf.split(np.zeros(len(samps)), samps)):

            if window:
                x_train = torch.nn.utils.rnn.pad_sequence([window_data(X[i], window_size)\
                                                           for i in train_index], batch_first=True) * torch.tensor(1e-1)
            else:
                x_train = torch.nn.utils.rnn.pad_sequence([X[i] for i in train_index], batch_first=True) * torch.tensor(1e-1)

            x_test = torch.nn.utils.rnn.pad_sequence([X[i] for i in test_index], batch_first=True)
            y_train = torch.nn.utils.rnn.pad_sequence([Y[i] for i in train_index], batch_first=True)
            y_test = torch.nn.utils.rnn.pad_sequence([Y[i] for i in test_index], batch_first=True)

            x_train = x_train.to(dev)
            x_test  = x_test.to(dev)
            y_train = y_train.to(dev)
            y_test  = y_test.to(dev)

            train_ds = TensorDataset(x_train, y_train)
            test_ds  = TensorDataset(x_test, y_test)

            train_dl.append(DataLoader(train_ds, batch_size=batch_size, shuffle=True))
            test_dl.append(DataLoader(test_ds, batch_size=batch_size))
        
    return train_dl, test_dl
    

def make_dataset(dev, data_all, n_classes, target_all, window_length, batch_size=1):
    xs = []
    ys = []

    for data, target in zip(data_all, target_all):
        for i in range(0, data[0].shape[0], window_length):
            xi = []
            yi = []
            for j in range(0, len(data)):
                xi.append(data[j][i:i+window_length])
                yi.append(np.eye(n_classes)[target])
            xs.append(xi)
            ys.append(yi)

    X = [torch.tensor(x, dtype=torch.get_default_dtype()) for x in xs]
    Y = [torch.tensor(y, dtype=torch.get_default_dtype()) for y in ys]

    X_data = torch.nn.utils.rnn.pad_sequence([X[i] for i in range(len(X))], batch_first=True).to(dev)
    Y_data = torch.nn.utils.rnn.pad_sequence([Y[i] for i in range(len(Y))], batch_first=True).to(dev)

    print("data size: ")
    print(X_data.size())
    print(Y_data.size())
    
    train_ds = TensorDataset(X_data, Y_data)
    
    train_dl = DataLoader(train_ds, batch_size = batch_size, shuffle=True)
    
    return train_dl

def make_dataset2(dev, data_all, n_classes, target_all, window_length, batch_size, shuffle=True):
    xs = []
    ys = []

    for data, target in zip(data_all, target_all):
        for i in range(0, data[0].shape[0], window_length):
            xi = []
            for j in range(0, len(data)):
                xi.append(data[j][i:i+window_length])
            yi = []
            for j in range(0, len(target)):
                yi.append(target[j][i:i+window_length])
            xs.append(xi)
            ys.append(yi)

    X = [torch.tensor(x, dtype=torch.get_default_dtype()) for x in xs]
    Y = [torch.tensor(y, dtype=torch.get_default_dtype()) for y in ys]

    X_data = torch.nn.utils.rnn.pad_sequence([X[i] for i in range(len(X))], batch_first=True).to(dev)
    Y_data = torch.nn.utils.rnn.pad_sequence([Y[i] for i in range(len(Y))], batch_first=True).to(dev)

    print("data size: ")
    print(X_data.size())
    print(Y_data.size())
    
    train_ds = TensorDataset(X_data, Y_data)
    
    train_dl = DataLoader(train_ds, batch_size = batch_size, shuffle=shuffle)
    
    return train_dl

def window_data(X, window_length):
    return X[int(len(X) / 2 - window_length / 2):int(len(X) / 2 + window_length / 2)]