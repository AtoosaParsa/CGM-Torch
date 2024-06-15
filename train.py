""" Gradient-based optimization of granular crystals
    The stiffness of the particles in a granular system is optimized to achieve a desireable wave response.
    
    Source code for the AND experiment in section 4.2.1 (Figure 7) in the following paper:
    Parsa, A., O'Hern, C. S., Kramer-Bottiglio, R., & Bongard, J. (2024). Gradient-based Design of Computational Granular 
    Crystals. arXiv preprint arXiv:2404.04825.

    Partially inspired by the following repositories: [1] https://github.com/a-papp/SpinTorch
                                                      [2] https://github.com/fancompute/wavetorch
                                                      
    AND gate with uniform initial distribution and adaptive learning rate
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
import pickle
import sys
import os

import torch
import numpy as np
import pandas as pd

import CGMtorch as CGMtorch
import myUtils as myUtils
import myIO as myIO
import loadData as loadData


parser = argparse.ArgumentParser() 
parser.add_argument('--name', type=str, default=time.strftime('%Y%m%d%H%M%S'),
                    help='Name to use when saving or loading the model file.')
parser.add_argument('--savedir', type=str, default='./exp/',\
                    help='Directory in which the model file is saved.')
parser.add_argument('--seed', type=int, default=0,\
                    help='Random seed for the experiments.')

def fft_mag(X, freq = 15.0, dt=5e-3):
    X_fft = torch.fft.rfft(X, norm='forward', dim=2) #dim 0 is batch, dim 1 is number of inputs/outputs
    X_mag = torch.abs(X_fft)
    fft_freqs = 2 * np.pi * torch.fft.rfftfreq(X.size(2), d=dt)
    ind = torch.argwhere(fft_freqs>freq)
    index = ind[0][0]
    fft_target = X_mag[:, :, index-1] + (X_mag[:, :, index]-X_mag[:, :, index-1]) * \
          ((freq-fft_freqs[index-1])/(fft_freqs[index]-fft_freqs[index-1]))
    
    return fft_target

if __name__ == '__main__':
    
    args = parser.parse_args()

    if not os.path.exists(args.savedir):
        os.makedirs(args.savedir)
    
    directory = args.savedir+'seed'+str(args.seed)+'/'
    if not os.path.exists(directory):
        os.makedirs(directory)
        
    ## setup for torch
    torch.manual_seed(int(args.seed))

    if torch.cuda.is_available():
        dev = torch.device('cuda')
    else:
        dev = torch.device('cpu')
    
    print('Running on ', dev)

    # check if cudnn is enabled
    print(torch.backends.cudnn.version())
    
    # it has to be float64 not float32, I was having problems with the dphi=0 case, I think because of the pi value...
    torch.set_default_dtype(torch.float64)
    #torch.set_default_tensor_type(torch.DoubleTensor)

    #torch.set_printoptions(profile="full")

    ## Configuration, sources, probes, model definitions
    # dt needs to be small enough for the numerical integration
    # we need some background damping to get clean sine waves
    params = {"N_x": 10, "N_y": 11, "mass": 1.0, "d": 0.1, "dphi": 0.1, "b": 1.0, "k_init": "uniform-01", "k_max": 10.0, \
              "k_min": 1.0, "b_pp": 0.0, "b_pw": 0.0, "b_pp_pad": 0.0, "k_w": 1.0, "k_pad": 1.0, "padding": 0, \
              "encoding": "float-01", "clipped": "min-max", "lattice": "hex", "type": "float64", "device": dev, \
              "source": [19, 76], "probes": [56], "dt": 5e-3, "fire_reset": True, "Nt_fire": 1e5}

    
    ## Dataset and training parameters
    
    datasetParams = {"timeSteps": 6000,
                     "window": True,
                     "window_length": 3000,
                     "sampling_rate": 10000,
                     "dt": 5e-3,
                     "dataset": 'and2',
                     "seed": int(args.seed),
                     "batchSize": 1}
    
    train_dl, test_dl = loadData.loadData(dev, datasetParams)
    
    dataset = {'data': [train_dl, test_dl], 'params': datasetParams}
    
    f = open(directory+"dataset.dat", 'wb')
    pickle.dump(dataset, f)
    f.close()


    ## Define the optimizer and loss function
    N_epochs = 500
    learning_rate = 0.001

    
    ## set up the model
    config = CGMtorch.VariableStiffnessParticles(dev, params['N_x'], params['N_y'], params['mass'], params['d'],\
                                                 params['dphi'], params['b'], params['k_init'], params['k_max'],\
                                                 params['k_min'], params['b_pp'], params['b_pw'], params['b_pp_pad'],\
                                                 params['k_w'], params['k_pad'], params['padding'], params['lattice'],\
                                                 params['encoding'], params['clipped'])
    
    srcs = []
    Ss = params['source']
    for s in Ss:
        srcs.append(CGMtorch.Source(s, params['N_x'], params['N_y'], params['padding'], params['lattice']))
        
    probes = []
    Ps = params['probes']
    for p in Ps:
        probes.append(CGMtorch.IntensityProbe(p, params['N_x'], params['N_y'], params['padding'], params['lattice']))

    dynamics = CGMtorch.Dynamics(dev, config, srcs, probes, params['dt'], params['Nt_fire'], hertzian=True)
    
    CGMtorch.plot.plot_config(0, config.N_, config.X_ini, config.Y_ini, config.D, config.Lx, config.Ly, config.K_padded, \
                             params['k_min'], params['k_max'], srcs, probes, plotdir=directory)

    model = CGMtorch.MDSolver(dev, dynamics, srcs, probes)

    # sending model to GPU/CPU
    model.to(dev)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, foreach=False)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150, 300, 400], gamma=0.1)
    
    # using dtype=float64 gives me errors: foreach=False 
    
    criterion = torch.nn.L1Loss(reduction='mean')


    for name, param in model.named_parameters():
        if param.requires_grad:
            print("Model parameters:")
            print(name)
            print(param.data)
    
    # minibatches bigger than two, messes this up, because we are changing the Ks midway through the traj, but fixing the Xs
    # truncated backprop through time?!
    miniBatch = 3000
    state_size = (datasetParams['batchSize'],) + dynamics.initial_X.size()
    
    history = pd.DataFrame(columns=['time', 'epoch', 'loss_train', 'acc_train'])

    history_model_state = []
    acc_iter = [0]
    loss_iter = []

    ## Train the network
    myUtils.tic()
    t_start = time.time()
    for epoch in range(0, N_epochs + 1):
        #with torch.autograd.set_detect_anomaly(False):
        loss_sum = []
        acc = [0]
        if epoch % 10 == 0:
            myUtils.cudaStats(f'epoch {epoch}')
            myUtils.memReport()
            myUtils.cpuStats()
        for num, (xb, yb) in enumerate(train_dl):
            print(f"batch: {num}")
            X = torch.zeros(state_size, device=dev)
            Y = torch.zeros(state_size, device=dev)
            V_x = torch.zeros(state_size, device=dev)
            V_y = torch.zeros(state_size, device=dev)
            A_x = torch.zeros(state_size, device=dev)
            A_y = torch.zeros(state_size, device=dev)
            
            fire = params['fire_reset']

            for i, (xbi, ybi) in enumerate(zip(xb.chunk(int(xb.size(2)/miniBatch), dim=2), yb.chunk(int(yb.size(2)/miniBatch), dim=2))):
                print(f"minibatch: {i}")
                if epoch == 0:
                    with torch.no_grad():
                        outs, X, Y, V_x, V_y, A_x, A_y = model(xbi, X, Y, V_x, V_y, A_x, A_y, fire, propagate=False)
                        xb_mag = torch.sum(torch.sum(torch.pow(torch.abs(xbi[:, :, 2000:]), 2), dim=2), dim=1)/xbi.size(1)
                        yb_pred_mag = torch.sum(torch.transpose(outs[:, 2000:, :], 2, 1), dim=2)
                        #print(yb_pred_mag.size())
                        #print(yb_pred_mag)
                        yb_mag = torch.sum(torch.pow(torch.abs(ybi[:, :, 2000:]), 2), dim=2)
                        #print(yb_mag)
                        
                        loss = criterion(yb_pred_mag, yb_mag)
                        
                        print(loss)
                        loss_sum.append(loss.item())
                        X.detach_(); Y.detach_(); V_x.detach_(); V_y.detach_(); A_x.detach_(); A_y.detach_()
                        outs.detach_(); loss.detach_();
                else:
                    optimizer.zero_grad()
                    outs, X, Y, V_x, V_y, A_x, A_y = model(xbi, X, Y, V_x, V_y, A_x, A_y, fire, propagate=False)
                    xb_mag = torch.sum(torch.sum(torch.pow(torch.abs(xbi[:, :, 2000:]), 2), dim=2), dim=1)/xbi.size(1)
                    yb_pred_mag = torch.sum(torch.transpose(outs[:, 2000:, :], 2, 1), dim=2)
                    yb_mag = torch.sum(torch.pow(torch.abs(ybi[:, :, 2000:]), 2), dim=2)
                    loss = criterion(yb_pred_mag, yb_mag)
                    
                    if True:
                        # don't back prop in the transient part of the signal
                        loss.backward()
                        optimizer.step()
                        loss_sum.append(loss.item())
                    # make sure the stiffness is within the desired range
                    # or we can add a regularization term to the loss
                    dynamics.material.constrain_K()
                    X.detach_(); Y.detach_(); V_x.detach_(); V_y.detach_(); A_x.detach_(); A_y.detach_()
                    outs.detach_(); loss.detach_(); #yb_pred.detach_()
                    
            del X; del Y; del V_x; del V_y; del A_x; del A_y
        
        loss_iter.append(np.mean(loss_sum))
        scheduler.step()

        print("Epoch finished: %d -- Loss: %.6f, Acc: %.6f" % (epoch, np.mean(loss_sum), np.mean(acc)))

        myUtils.toc()
        

        if epoch % 10 == 0:
            CGMtorch.plot.plot_loss(loss_iter, directory)
            print(config.K)
        
        if epoch % 20 == 0:
            history = history._append({'time': time.time(),
                                       'epoch': epoch,
                                       'loss_train': np.mean(loss_sum),
                                       'acc_train': np.mean(acc)},
                                       ignore_index=True)

            history_model_state.append(copy.deepcopy(dynamics.material.state_reconstruction_args()))
            with torch.no_grad():
                for num, (xb, yb) in enumerate(train_dl):
                    # just plot the first one in each batch!                
                    X = torch.zeros(state_size, device=dev)
                    Y = torch.zeros(state_size, device=dev)
                    V_x = torch.zeros(state_size, device=dev)
                    V_y = torch.zeros(state_size, device=dev)
                    A_x = torch.zeros(state_size, device=dev)
                    A_y = torch.zeros(state_size, device=dev)

                    fire = params['fire_reset']
                    outs, X, Y, V_x, V_y, A_x, A_y = model(xb, X, Y, V_x, V_y, A_x, A_y, fire, propagate=True)
                    xb_mag = torch.sum(torch.sum(torch.pow(torch.abs(xb[:, :, 2000:]), 2), dim=2), dim=1)/xb.size(1)
                    yb_pred_mag = torch.sum(torch.transpose(outs[:, 2000:, :], 2, 1), dim=2)
                    yb_mag = torch.sum(torch.pow(torch.abs(yb[:, :, 2000:]), 2), dim=2)
                    loss = criterion(yb_pred_mag, yb_mag)
                    
                    CGMtorch.plot.plot_waves(epoch, torch.transpose(xb[0, :, :], 1, 0), \
                                             ["input-"+str(i) for i in range(xb.size(2))],\
                                             directory, name=str(num)+'_'+'mbatch_0_input')
                    CGMtorch.plot.plot_waves(epoch, outs[0, :, :], ["port-"+str(i) for i in range(outs.size(2))],\
                                                  directory, name=str(num)+'_'+'mbatch_0_output')
                    
                CGMtorch.plot.plot_config(epoch, config.N_, config.X_ini, config.Y_ini, config.D, config.Lx, config.Ly,\
                                          config.K_padded, params['k_min'], params['k_max'], srcs, probes, \
                                          plotdir=directory)

        if epoch % 50 == 0:
            myIO.save_model(model, args.name+"_"+str(epoch), directory, history, history_model_state, optimizer, \
                            cfg=params, verbose=False)
    
    myIO.save_model(model, args.name+"_final", directory, history, history_model_state, optimizer, cfg=params, verbose=True)
    
    CGMtorch.plot.plot_loss(loss_iter, directory)
    
    print(config.K)
    
    with torch.no_grad():        
        for num, (xb, yb) in enumerate(train_dl):
            X = torch.zeros(state_size, device=dev)
            Y = torch.zeros(state_size, device=dev)
            V_x = torch.zeros(state_size, device=dev)
            V_y = torch.zeros(state_size, device=dev)
            A_x = torch.zeros(state_size, device=dev)
            A_y = torch.zeros(state_size, device=dev)
            outs, X, Y, V_x, V_y, A_x, A_y = model(xb, X, Y, V_x, V_y, A_x, A_y, params['fire_reset'], propagate=True)
            xb_mag = torch.sum(torch.sum(torch.pow(torch.abs(xb[:, :, 2000:]), 2), dim=2), dim=1)/xb.size(1)
            yb_pred_mag = torch.sum(torch.transpose(outs[:, 2000:, :], 2, 1), dim=2)
            yb_mag = torch.sum(torch.pow(torch.abs(yb[:, :, 2000:]), 2), dim=2)
            loss = criterion(yb_pred_mag, yb_mag)
            
            for i in range(xb.size(0)):
                CGMtorch.plot.plot_waves(epoch, torch.transpose(xb[i, :, :], 1, 0),\
                                         ["input-"+str(i) for i in range(xb.size(2))], directory, \
                                          name='batch_'+str(num)+'_'+'minibatch_'+str(i)+'_input')

                CGMtorch.plot.plot_waves(epoch, outs[i, :, :], ["port-"+str(p) for p in range(outs.size(2))],\
                                          directory, name='batch_'+str(num)+'_'+'minibatch_'+str(i)+'_output')

    CGMtorch.plot.plot_config(epoch, config.N_, config.X_ini, config.Y_ini, config.D, config.Lx, config.Ly,\
                              config.K_padded, params['k_min'], params['k_max'], srcs, probes, plotdir=directory)

    print('Total Time: %.2f min\n' % ((time.time() - t_start) / 60))
                