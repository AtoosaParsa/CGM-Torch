""" Molecular Dynamics simulation with backpropagation

"""

__author__ = 'Atoosa Parsa'
__copyright__ = 'Copyright 2024, Atoosa Parsa'
__credits__ = 'Atoosa Parsa'
__license__ = 'MIT License'
__version__ = '2.0.0'
__maintainer__ = 'Atoosa Parsa'
__email__ = 'atoosa.parsa@gmail.com'
__status__ = "Dev"


from typing import Tuple

from numba import jit
import torch 
import numpy as np

class SkipMDSolver(torch.nn.Module):

    def __init__(self, device, dynamics, sources: list, probes: list):
        
        super().__init__()
        
        self.device = device

        self.dynamics = dynamics
        
        # input and output ports        
        self.sources = torch.nn.ModuleList(sources)
        self.probes = torch.nn.ModuleList(probes)
        
        
    def forward(self, signal, X, Y, V_x, V_y, A_x, A_y, fire=False, propagate=False):  
        
        #device = "cuda" if next(self.parameters()).is_cuda else "cpu"
        
        batch_size = signal.shape[0]
        
        K_ij = self.dynamics.material.K_ij
        
        # energy minimization for a statically stable packing
        initial_X, initial_Y = self.dynamics.init_state(fire)
        
        outputs = []
        for i, sig in enumerate(signal.chunk(signal.size(2), dim=2)):
            
            if i % 2 == 0:

                X, Y, V_x, V_y, A_x, A_y = self.dynamics(K_ij, X, Y, V_x, V_y, A_x, A_y, initial_X, initial_Y, sig)

                # read the output probes, displacement of the particles at the output ports
                probe_values = []
                for i, prb in enumerate(self.probes):
                    if not propagate:
                        probe_values.append(prb(X))
                    else:
                        probe_values.append(prb.displacements(X))

                outputs.append(torch.stack(probe_values, dim=-1))
                
            else:
                with torch.no_grad():
                    X_, Y_, V_x_, V_y_, A_x_, A_y_ = self.dynamics(K_ij.detach(), X.detach(), Y.detach(), V_x.detach(),\
                                                                   V_y.detach(), A_x.detach(), A_y.detach(), \
                                                                   initial_X.detach(), initial_Y.detach(), sig.detach())
 
                    X.copy_(X_); Y.copy_(Y_); V_x.copy_(V_x_);
                    V_y.copy_(V_y_); A_x.copy_(A_x_); A_y.copy_(A_y_) 
                    # read the output probes, displacement of the particles at the output ports
                    #probe_values = []
                    #for i, prb in enumerate(self.probes):
                    #    if not propagate:
                    #        probe_values.append(prb(X))
                    #    else:
                    #        probe_values.append(prb.displacements(X))

                    #outputs.append(torch.stack(probe_values, dim=-1))
                
            #torch.set_printoptions(profile="full")
            #if(X.isnan().any()):
            #    print("X")
            #    print(self.X)
            #if(Y.isnan().any()):
            #    print("Y")
            #    print(self.Y)
            #if(V_x.isnan().any()):
            #    print("V_x")
            #    print(V_x)
            #if(V_y.isnan().any()):
            #    print("V_y")
            #    print(V_y)
            #torch.set_printoptions(profile="default")
            
        return torch.stack(outputs, dim=1), X, Y, V_x, V_y, A_x, A_y # outputs: [batch, timesteps, probes]
    
    