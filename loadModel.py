""" Load the data from the files and make plots

"""

__author__ = 'Atoosa Parsa'
__copyright__ = 'Copyright 2024, Atoosa Parsa'
__credits__ = 'Atoosa Parsa'
__license__ = 'MIT License'
__version__ = '2.0.0'
__maintainer__ = 'Atoosa Parsa'
__email__ = 'atoosa.parsa@gmail.com'
__status__ = "Dev"

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import time
import copy
import argparse
import sys
import os
import io

import pickle
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib

import CGMtorch as CGMtorch
import myUtils as myUtils
import myIO as myIO
import loadData

font = {'family' : 'sans-serif'}  
plt.rc('font', **font) 
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

sns.set_style("ticks", {"xtick.major.size": 8, "ytick.major.size": 8, 'font.family': 'serif', 'font.serif': 'Times New Roman'})
colors = ["cyan", "dark pink", "ocean green", "tan", "light red", "grey", "apricot"]
sns.set_palette(sns.xkcd_palette(colors), desat=.9)


parser = argparse.ArgumentParser() 
parser.add_argument('--name', type=str, default=time.strftime('%Y%m%d%H%M%S'),
                    help='Name to use when saving or loading the model file.')
parser.add_argument('--savedir', type=str, default='./exp/',\
                    help='Directory in which the model file is saved.')
parser.add_argument('--seed', type=int, default=5,\
                    help='Number of seeds')
parser.add_argument('--output', type=list, default='plots',\
                    help='Output directory')
parser.add_argument('--epoch', type=int, default=None,\
                    help='Epoch.')
parser.add_argument('--plotName', type=str, default=None,\
                    help='Plot name.')

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)

if __name__ == '__main__':
    
    args = parser.parse_args()
    
    if not os.path.exists(args.savedir):
        print("input directory does not exist.")
        
    if args.epoch != None:
        modelName = args.name+'_'+str(args.epoch)+'.pt'
    else:
        modelName = args.name+'_final'+'.pt'

    savepath = os.path.join(args.savedir, 'plots')
    if not os.path.exists(savepath):
        os.makedirs(savepath)

    seeds = np.arange(1, int(args.seed)+1)
    losses = []
    case = ['case 01', 'case 10', 'case 11']
    stiffnesses = []
    for s in seeds:
        
        plotPath = os.path.join(savepath, f'seed{s}/')
        if not os.path.exists(plotPath):
            os.makedirs(plotPath)
        
        path = os.path.join(args.savedir, f'seed{s}/')
        
        model, dynamics, history, history_state, params = myIO.load_model(path+modelName)

        dev = torch.device('cpu')

        print('Running on ', dev)

        torch.set_default_dtype(torch.float64)

        f = open(path+"dataset.dat", 'rb')    
        data = CPU_Unpickler(f).load()
        f.close()

        datasetParams = data['params']
        datasetParams['timeSteps'] = datasetParams['window_length']
        
        train_dl, test_dl = loadData.loadData(dev, datasetParams, shuffle=False)
        
        N_epochs = np.array(history.loc[:, 'epoch'])[-1]
        state_size = (datasetParams['batchSize'],) + dynamics.initial_X.size()
    
        N_epochs = np.array(history.loc[:, 'epoch'])[-1]
        
        figsize = 8
        fig = plt.figure(figsize=(1.5*figsize, 1*figsize))
        gs = fig.add_gridspec(3, 2, hspace=0, wspace=0, top=0.9)
        axs = gs.subplots(sharey=True, sharex=True)
        #fig.subplots_adjust(wspace=0, hspace=0)
        
        for num, (xb, yb) in enumerate(train_dl):
            X = torch.zeros(state_size, device=dev)
            Y = torch.zeros(state_size, device=dev)
            V_x = torch.zeros(state_size, device=dev)
            V_y = torch.zeros(state_size, device=dev)
            A_x = torch.zeros(state_size, device=dev)
            A_y = torch.zeros(state_size, device=dev)

            with torch.no_grad():
                if num >= 3:
                    break
                
                outs, X, Y, V_x, V_y, A_x, A_y = model(xb, X, Y, V_x, V_y, A_x, A_y, params['fire_reset'], propagate=True)               
                for i in range(xb.size(0)):
                    x = torch.transpose(xb[i, :, :], 1, 0).detach().cpu().numpy()
                    axs[num][0].plot(x[:, 0], color='blue', linestyle='-', linewidth=2, label='input 1')
                    axs[num][0].plot(x[:, 1], color='green', linestyle='--', linewidth=2, label='input 2')
                    axs[num][0].grid(which='major', color='gray', linestyle='-', linewidth=0.3)
                    axs[num][0].set_axisbelow(True)
                    axs[num][0].tick_params(axis='both', which='major', labelsize=7, color='k')
                    if num == 0:
                        axs[num][0].legend(loc='upper right', fontsize=16)
                    axs[num][0].text(0, 0.0008, case[num], fontsize=16, weight='demibold', backgroundcolor='w')
                    axs[num][0].set_ylim((-1.1e-3, 1.1e-3))
                                        
                    y = outs[i, :, :].detach().cpu().numpy()
                    axs[num][1].plot(y[:, 0], color='tomato', linestyle='-', linewidth=2, label='output')
                    axs[num][1].grid(which='major', color='gray', linestyle='-', linewidth=0.3)
                    axs[num][1].set_axisbelow(True)
                    axs[num][1].tick_params(axis='both', which='major', labelsize=7, color='k')
                    if num == 0:
                        axs[num][1].legend(loc='upper right', fontsize=16)
                    axs[num][1].set_ylim((-1.1e-3, 1.1e-3))
        
        fig.text(0.5, 0.07, "Time Steps", ha='center', va='center', fontsize=14)
        fig.text(0.07, 0.5, "Displacement", ha='center', va='center', rotation='vertical', fontsize=14)
        fig.suptitle(f"Particle Displacements - {args.plotName}", fontsize=16, y=0.93, fontweight='bold')
        for ax in fig.get_axes():
            ax.label_outer()
        plt.tight_layout()
        savename = 'waves_seed'+str(s)+'.png'
        saving = os.path.join(plotPath, savename)
        plt.show()
        fig.savefig(saving, bbox_inches='tight', dpi=300) #, transparent=False)
        plt.close(fig)
        
        print("Seed: "+str(s))
        stiffnesses.append(dynamics.material.K_padded.detach().cpu().numpy().tolist())
        #print(dynamics.material.K_padded)
        CGMtorch.plot.plot_config(N_epochs, dynamics.material.N_, dynamics.material.X_ini, dynamics.material.Y_ini,\
                                  dynamics.material.D,\
                                  dynamics.material.Lx, dynamics.material.Ly, dynamics.material.K_padded, \
                                  params['k_min'], params['k_max'], dynamics.sources, dynamics.probes,\
                                  plotdir=plotPath)
        
        losses.append(np.array(history.loc[:, 'loss_train']))
        print('Seed: '+str(s))
        times = np.array(history.loc[:, 'time'])
        print('Total Time: %.2f min\n' % ((times[-1] - times[0]) / 60))
    
    print(stiffnesses)
    losses = np.array(losses)
    epochs = np.array(history.loc[:, 'epoch'])
    
    mean = np.mean(losses, axis=0)
    std = np.std(losses, axis=0)
    
    fig, ax = plt.subplots(1, 1)
    fig.set_figwidth(8)
    fig.set_figheight(4)
    for i in range(losses.shape[0]):
        ax.plot(epochs, losses[i], color='blue', linewidth=1, alpha=0.15)
    ax.plot(epochs, mean, color='blue', linewidth=2.5, label='$Mean$')
    ax.plot(epochs, mean-std, color='blue', linewidth=1.5, linestyle='dashed', label='$\pm 1 \; STD$')
    ax.plot(epochs, mean+std, color='blue', linewidth=1.5, linestyle='dashed')

    ax.set_axisbelow(True)
    #ax.grid(which='minor', color='gray', linestyle=':', linewidth=0.2)
    ax.grid(which='major', color='gray', linestyle='-', linewidth=0.2)
    ax.minorticks_on()
    ax.set_xlabel('Epochs', fontsize=14, fontweight='bold')
    ax.set_ylabel('Loss', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=12, width=2)
    ax.tick_params(axis='both', which='minor', labelsize=12, width=1)
    plt.setp(ax.get_xticklabels(), visible=True)
    plt.setp(ax.get_yticklabels(), visible=True)

    plt.tight_layout()

    fig.savefig(savepath+'/loss.png', bbox_inches='tight', dpi=300, transparent=False)
    plt.show()
    plt.close(fig)