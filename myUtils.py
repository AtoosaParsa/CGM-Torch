""" IO

"""

__author__ = 'Atoosa Parsa'
__copyright__ = 'Copyright 2024, Atoosa Parsa'
__credits__ = 'Atoosa Parsa'
__license__ = 'MIT License'
__version__ = '2.0.0'
__maintainer__ = 'Atoosa Parsa'
__email__ = 'atoosa.parsa@gmail.com'
__status__ = "Dev"


import time
import gc
import psutil
import sys
import os

import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

usecolortext = True

def memReport():
    for obj in gc.get_objects():
        if torch.is_tensor(obj):
            print(type(obj), obj.size())
    
def cpuStats():
        print(sys.version)
        print(psutil.cpu_percent())
        print(psutil.virtual_memory())  # physical memory usage
        pid = os.getpid()
        py = psutil.Process(pid)
        memoryUse = py.memory_info()[0] / 2. ** 30  # memory use in GB...
        print('memory GB:', memoryUse)

def cudaStats(msg=''):
    print('GPU memory usage ' + msg + ':')
    print('allocated: %dM (max %dM), cached: %dM (max %dM)'
          % (torch.cuda.memory_allocated() / 1024 / 1024,
             torch.cuda.max_memory_allocated() / 1024 / 1024,
             torch.cuda.memory_reserved() / 1024 / 1024,
             torch.cuda.max_memory_reserved() / 1024 / 1024))


def normalize_X(dev, X):

    X_sum = torch.sum(X, dim=1, keepdim=True)
    if X_sum.all():
        power = X / torch.sum(X, dim=1, keepdim=True)
    # all values are zero
    else:
        power = X
    #print(power)
    #out = torch.nn.Softmax(power, dim=0) no softmax needed
    #out = power / torch.sum(power, dim=0, keepdim=True) # no normalization needed
    return power.to(dev)


def accuracy_onehot(y_pred, y_label):
    return (y_pred.argmax(dim=1) == y_label).float().mean().item()


def tic():
    global tic_t0, tic_t1
    tic_t0 = tic_t1 = time.perf_counter()


def toc():
    if usecolortext:
        green = '\x1b[1;32m'
        blue =  '\x1b[1;34m'
        end =   '\x1b[0m'
    else:
        green=blue=end=''
    t = time.perf_counter()
    global tic_t1
    if (t - tic_t1) > 0.01:
        print('Elapsed time %s%.2f s%s,' % (green, (t - tic_t0), end),
              '(dt = %s%.2f s%s)' % (blue,(t - tic_t1), end))
    else:
        print('Elapsed time %s%.2f s%s,' % (green, (t - tic_t0), end),
              '(dt = %s%.3f ms%s)' % (blue,((t - tic_t1)*1000), end))
    tic_t1 = t
