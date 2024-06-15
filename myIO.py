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


import sys
import os
import copy

import torch

import CGMtorch
import myUtils

cwd = os.getcwd()
sys.path.append(cwd+'/CGMtorch')

def save_model(model,
               name,
               savedir,
               history,
               history_config_state,
               optimizer,
               cfg=None,
               verbose=True):
    # Save the model state and history to a file
    
    str_filename = name + '.pt'
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    str_savepath = savedir + str_filename

    if history_config_state is None:
        history_config_state = [model.dynamics.material.state_reconstruction_args()]

    data = {'model_config_class_str': model.dynamics.material.__class__.__name__,
            'model_state': model.state_dict(),
            'history': history,
            'history_config_state': history_config_state,
            'optimizer_state_dict': optimizer.state_dict(),
            'cfg': cfg}

    if verbose:
        print("Saving model to %s" % str_savepath)
    torch.save(data, str_savepath)
    
    return True


def new_config(class_str, state):
    #print(class_str)
    #configClass = getattr(material, class_str)
    config_state = copy.deepcopy(state)
    if not torch.cuda.is_available():
        dev = torch.device('cpu')
        config_state['device'] = dev
    return CGMtorch.VariableStiffnessParticles(**config_state)


def load_model(str_filename, which_iteration=-1):
    # Load a previously saved model and its history from a file

    print("Loading model from %s" % str_filename)
    
    if not torch.cuda.is_available():
        data = torch.load(str_filename, map_location=torch.device('cpu'))
        data['cfg']['device'] = torch.device('cpu')        
    else:
        data = torch.load(str_filename)

    torch.set_default_dtype(torch.float64)

    new_config_ = new_config(data['model_config_class_str'], data['history_config_state'][which_iteration])

    model_state = copy.deepcopy(data['model_state'])

    params = data['cfg']    
    
    new_srcs = []
    Ss = params['source']
    for s in Ss:
        new_srcs.append(CGMtorch.Source(s, params['N_x'], params['N_y'], params['padding'], params['lattice']))
        
    new_probes = []
    Ps = params['probes']
    for p in Ps:
        new_probes.append(CGMtorch.IntensityProbe(p, params['N_x'], params['N_y'], params['padding'], params['lattice']))

    new_dym = CGMtorch.Dynamics(params['device'], new_config_, new_srcs, new_probes, params['dt'], \
                                params['Nt_fire'], hertzian=True)
    
    new_model = CGMtorch.MDSolver(params['device'], new_dym, new_srcs, new_probes)

    new_model.eval()

    return new_model, new_dym, data['history'], data['history_config_state'], data['cfg']
