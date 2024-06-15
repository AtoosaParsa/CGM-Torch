""" Assign the outputs

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

class Probe(torch.nn.Module):
    
    def __init__(self, index, Nx, Ny, pad=0, lattice="hex"):
        
        super().__init__()
        
        real_index = self.index_init(Nx + pad * 2, Ny + pad * 2, pad, lattice, index)           
        self.register_buffer('index', torch.tensor(real_index, dtype=torch.int64))
        
    def forward(self, state):
        return state[:, self.index] #.unsqueeze(0).expand(state.size(0), 1)

    def coordinates(self):
        return self.index.detach().cpu().numpy()

    def index_init(self, Nx, Ny, pad, lattice, index):
        
        ind = 0 # index in the padded lattice
        ind_= 0 # index in the design area

        pad_indices = []
        k_indices = []
        
        real_index = -1
        
        if lattice == "square":
            for i in range(int(Ny)):
                for j in range(int(Nx)):
                    if (i < pad) or (i >= pad + int(Ny)):
                        pad_indices.append(ind)
                    elif (j < pad) or (j >= pad + int(Nx)):
                        pad_indices.append(ind)
                    else:
                        k_indices.append(ind_)
                        if index == ind_:
                            real_index = ind
                            return real_index
                        ind_ = ind_ + 1
                    ind = ind + 1
        
        elif lattice == "hex":
            for i in range(int(Ny)): # rows
                if i % 2 == 0: # on even rows
                    for j in range(int(Nx)): #columns
                        if (i < pad) or (i >= pad + int(Ny)):
                            pad_indices.append(ind)
                        elif (j < pad) or (j >= pad + int(Nx)):
                            pad_indices.append(ind)
                        else:
                            k_indices.append(ind_)
                            if index == ind_:
                                real_index = ind
                                return real_index
                            ind_ = ind_ + 1
                        ind = ind + 1
                else: # on odd rows
                    for j in range(int(Nx)-1):
                        if (i < pad) or (i >= pad + int(Ny)):
                            pad_indices.append(ind)
                        elif (j < pad) or (j >= pad + int(Nx) - 1):
                            pad_indices.append(ind)
                        else:
                            k_indices.append(ind_)
                            if index == ind_:
                                real_index = ind
                                return real_index
                            ind_ = ind_ + 1
                        ind = ind + 1
        
        return real_index
    
class IntensityProbe(Probe):
    def __init__(self, index, Nx, Ny, pad=0, lattice="hex"):
        super().__init__(index, Nx, Ny, pad, lattice)

    def forward(self, state):
        # return the wave intensity
        return super().forward(state).pow(2) #.sum().pow(2).unsqueeze(0)
    
    def displacements(self, state):
        # return the displacement, for checking the signal at the output port
        return super().forward(state)