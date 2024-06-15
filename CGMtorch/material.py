""" Material properties

"""

__author__ = 'Atoosa Parsa'
__copyright__ = 'Copyright 2024, Atoosa Parsa'
__credits__ = 'Atoosa Parsa'
__license__ = 'MIT License'
__version__ = '2.0.0'
__maintainer__ = 'Atoosa Parsa'
__email__ = 'atoosa.parsa@gmail.com'
__status__ = "Dev"


import copy
import torch
import numpy as np

from .binarize import binarize

class GranularMaterial(torch.nn.Module):
    
    def __init__(self, device: torch.device, N_x: int, N_y: int, mass: float, d: float, dphi: float, k_max: float, \
                 k_min: float, lattice: str, padding: int):
        
        super().__init__()

        self.device = device
        
        self.mass = mass
        
        self.lattice = lattice
        
        # padding around the main design area to prevent wave reflections
        self.padding = padding
        
        # number of particles in x and y directions
        self.register_buffer("N_x", torch.tensor(N_x, dtype=torch.int64))
        self.register_buffer("N_y", torch.tensor(N_y, dtype=torch.int64))

        if lattice == "hex":
            self.register_buffer("N", torch.tensor(((N_x-1) * (N_y // 2) + (N_x) * (N_y - (N_y // 2))), dtype=torch.int64))
        else:
            self.register_buffer("N", torch.tensor(N_x * N_y, dtype=torch.int64))
        

        # number of particles after adding the padding
        N_x_ = N_x + padding * 2
        N_y_ = N_y + padding * 2
        self.register_buffer("N_x_", torch.tensor(N_x_, dtype=torch.int64))
        self.register_buffer("N_y_", torch.tensor(N_y_, dtype=torch.int64))

        if lattice == "hex":
            self.register_buffer("N_", torch.tensor(((N_x_-1) * (N_y_ // 2) + (N_x_) * (N_y_ - (N_y_ // 2))), \
                                                    dtype=torch.int64))
        else:
            self.register_buffer("N_", torch.tensor(N_x_ * N_y_, dtype=torch.int64))
   
        # maximum and minimum for the stiffness
        self.k_max = torch.tensor(k_max)
        self.k_min = torch.tensor(k_min)
        
        # mass matrix: N * 1
        self.register_buffer("M", torch.ones(self.N_.item(), device=device) * torch.tensor(mass))      

        
        # initial particle diameter
        self.register_buffer("d0", torch.tensor(d))
        
        # packing fraction
        self.register_buffer("dphi", torch.tensor(dphi))
        
        # dimensions of the box
        Lx = d * N_x_
        if lattice == "hex":
            Ly = (N_y_-1) * np.sqrt(3)/2*d + d # for hexagonal lattice
        elif lattice == "square":
            Ly = d * N_y_ # for square lattice

        self.register_buffer("Lx", torch.tensor(Lx))
        self.register_buffer("Ly", torch.tensor(Ly))
        #print("{0:0.10f}".format(Lx))
        
        # initial packing fraction with no compression
        phi0 = self.N_.item() * np.pi*d**2/4/(Lx*Ly)
        # new particle diameter after applying the compression
        d_ini = d * np.sqrt(1+dphi/phi0)
        
        #print("{0:0.10f}".format(d))
        #print("{0:0.10f}".format(d_ini))
        
       
        # particle diameter after adjusting the compression
        self.register_buffer("D", torch.tensor(d_ini))
        
        #torch.set_printoptions(precision=10)
        #print(self.D)
        #print(self.d0)
        #print(torch.tensor(d_ini))
        #print(torch.tensor(d_ini, dtype=torch.float32))
        #print(self.Lx)
     
    def state_reconstruction_args(self):
        return {"device": self.device,
                "mass": self.mass,
                "d": self.d0.item(),
                "dphi": self.dphi.item(),
                "k_max": self.k_max.item(),
                "k_min": self.k_min.item(),
                "N_x": self.N_x.item(),
                "N_y": self.N_y.item(),
                "padding": self.padding,
                "lattice": self.lattice}

    def forward(self):
        raise NotImplementedError


class VariableStiffnessParticles(GranularMaterial):
    
    def __init__(self, device, N_x: int, N_y: int, mass: float, d: float, dphi: float, b: float, k_init: str, k_max: float, \
                 k_min: float, b_pp: float, b_pw: float, b_pp_pad: float, k_w: float, k_pad: float, padding: int, \
                 lattice: str, encoding: str, clipped: str, initialize=None):

        super().__init__(device, N_x, N_y, mass, d, dphi, k_max, k_min, lattice, padding)
               
        # initial stiffness
        self.k_init = k_init
        
        self.encoding = encoding
        self.clipped = clipped
        
        # background damping
        self.register_buffer("B", torch.tensor(b))
        
        # particle-particle damping
        self.register_buffer("B_pp", torch.tensor(b_pp))
        
        # particle-wall damping
        self.register_buffer("B_pw", torch.tensor(b_pw))    
        
        # particle-particle damping in the padding area
        self.register_buffer("B_pp_pad", torch.tensor(b_pp_pad))

        # wall stiffness
        self.register_buffer("K_w", torch.tensor(k_w))
        
        # particle stiffness in the padding area
        self.register_buffer("k_pad", torch.tensor(k_pad))
        
        # stiffness matrix: N * 1
        if isinstance(initialize, torch.Tensor):
            self.K = torch.nn.Parameter(initialize.clone().detach())            
        elif isinstance(initialize, np.ndarray):
            self.K = torch.nn.Parameter(torch.from_numpy(initialize))            
        elif k_init == "uniform-float":
            self.K = torch.nn.Parameter(torch.ones(self.N.item(), dtype=torch.get_default_dtype(), device=device) * \
                                        torch.tensor(k_min + k_max / 2))        
        elif k_init == "uniform-01":
            self.K = torch.nn.Parameter(torch.ones(self.N.item(), dtype=torch.get_default_dtype(), device=device) * \
                                        torch.tensor(k_min/2, dtype=torch.float64))
        elif k_init == "random":
            self.K = torch.nn.Parameter((k_max - k_min) * torch.rand(self.N.item(), \
                                                                     dtype=torch.get_default_dtype(), device=device) + k_min)
            
        elif k_init == "random-01":
            self.K = torch.nn.Parameter(torch.rand(self.N.item(), dtype=torch.get_default_dtype(), device=device))
        
        elif k_init == "random-int":
            self.K = torch.nn.Parameter(torch.randint(int(k_min), int(k_max), size=(self.N.item(), 1),\
                                                device=device).squeeze().type(torch.get_default_dtype()))
        elif k_init == "binary":
            self.K = torch.nn.Parameter(torch.round(torch.rand((self.N.item(), 1), device=device)).squeeze())
        
        # make the mask for the main particles on the lattice
        self.mask = self.masking()
        
        # effective pairwise particle-particle damping matrix
        # It needs to be computed once because damping doesn't change during the training
        B_ij_ = self.B_ij_init()
        self.register_buffer("B_ij_", B_ij_)
         
        # initial positions
        X_ini, Y_ini = self.XY_init()
        self.register_buffer("X_ini", X_ini)
        self.register_buffer("Y_ini", Y_ini)
        
        #torch.set_printoptions(precision=10)
        #print(self.d0)
        #print(self.D)
        #print(self.X_ini)
        #print(self.Y_ini)

    def state_reconstruction_args(self):
        my_args = {"b": self.B.item(),
                   "k_init": self.k_init,
                   "encoding": self.encoding,
                   "clipped": self.clipped,
                   "b_pp": self.B_pp.item(),
                   "k_w": self.K_w.item(),
                   "b_pw": self.B_pw.item(),
                   "b_pp_pad": self.B_pp_pad.item(),
                   "k_pad": self.k_pad.item(),
                   "initialize": copy.deepcopy(self.K.detach())}
        
        return {**super().state_reconstruction_args(), **my_args}
    
    @property
    def K_ij(self):
        """ Effective stiffness between pairs of particles """
        K = self.K_model()
        temporary = torch.ones(self.N_.item(), device=self.device, dtype=torch.get_default_dtype()) * self.k_pad
        k_padded = temporary.masked_scatter(self.mask, K)
        temp = torch.abs(k_padded.reshape(-1, 1) - k_padded) #ki - kj
        K_ij = k_padded.unsqueeze(1).repeat(1, int(self.N_.item())) * torch.less(temp, 1e-3) + \
        torch.div(k_padded.reshape(-1, 1) * k_padded, k_padded.reshape(-1, 1) + k_padded) * torch.ge(temp, 1e-3)
            
        #if K_ij.isnan().any():
        #    print("K_ij is nan")
        #    print(self.K)
        
        return K_ij

    @property
    def B_ij(self):
        return self.B_ij_

    @property
    def K_padded(self):
        # for plotting!
        with  torch.no_grad():
            K = self.K_model()        
            temp = torch.ones(self.N_.item(), device=self.device, dtype=torch.get_default_dtype()) * -1
            k_padded = temp.masked_scatter(self.mask, K)
        
        return k_padded
        
    def constrain_K(self):
        if self.clipped == "min-max":            
            if self.encoding == 'float-01' or self.encoding == 'binary':
                with torch.no_grad():
                    self.K[self.K > 1.0] = 1.0
                    self.K[self.K < 0.0] = 0.0            
                #self.K.clamp(min=0.0, max=1.0)
                #w = self.K.data
                #w.sub_(torch.min(w)).div_(torch.max(w) - torch.min(w))
            
            else:
                with torch.no_grad():
                    self.K[self.K > self.k_max] = self.k_max
                    self.K[self.K < self.k_min] = self.k_min
                    
        elif self.clipped == "min":            
            if self.encoding == 'float-01' or self.encoding == 'binary':
                with torch.no_grad():
                    self.K[self.K < 0.0] = 0.0
            
            else:
                with torch.no_grad():
                    self.K[self.K < self.k_min] = self.k_min
                    
        return True

    def K_model(self):
        
        if self.encoding == "float-01":
            K = self.k_min.item() + (self.k_max.item() - self.k_min.item()) * self.K

        elif self.encoding == "float":
            K = self.K

        elif self.encoding == "smooth":
            K = self.smooth()

        elif self.encoding == "binary":
            K = self.k_min.item() + (self.k_max.item() - self.k_min.item()) * binarize(self.K)
            
        return K
            
    def smooth(self):
        """Low pass filter to increase + projection"""
        # ref. ...
        
        blur_N = 1
        blur_radius = 1
        blur_kernel = torch.ones(2*blur_radius+1, dtype=torch.get_default_dtype())
        blur_kernel=blur_kernel/blur_kernel.sum().unsqueeze(0).unsqueeze(0).unsqueeze(0)
        
        K_blurr = torch.nn.functional.conv1d(self.K.unsqueeze(0).unsqueeze(0),blur_kernel, \
                                             padding=blur_kernel.shape[-1]//2).squeeze()
        
        eta = 0.5
        beta = 1000
        
        return (np.tanh(beta * eta) + torch.tanh(beta * (K_blurr - eta))) / (
                np.tanh(beta * eta) + np.tanh(beta * (1 - eta)))
    
    def masking(self):

        mask = torch.zeros(self.N_.item(), device=self.device, dtype=torch.uint8)

        k_indices = []
        pad_indices = []
        
        ind = 0
        ind_= 0

        if self.lattice == "square":
            for i in range(int(self.N_y_.item())):
                for j in range(int(self.N_x_.item())):
                    if (i < self.padding) or (i >= self.padding + int(self.N_y.item())):
                        pad_indices.append(ind)
                    elif (j < self.padding) or (j >= self.padding + int(self.N_x.item())):
                        pad_indices.append(ind)
                    else:
                        mask[ind] = 1
                        k_indices.append(ind_)
                        ind_ = ind_ + 1
                    ind = ind + 1
        
        elif self.lattice == "hex":
            for i in range(int(self.N_y_.item())): # rows
                if i % 2 == 0: # on even rows
                    for j in range(int(self.N_x_.item())): #columns
                        if (i < self.padding) or (i >= self.padding + int(self.N_y.item())):
                            pad_indices.append(ind)
                        elif (j < self.padding) or (j >= self.padding + int(self.N_x.item())):
                            pad_indices.append(ind)
                        else:
                            mask[ind] = 1
                            k_indices.append(ind_)
                            ind_ = ind_ + 1
                        ind = ind + 1
                else: # on odd rows
                    for j in range(int(self.N_x_.item())-1):
                        if (i < self.padding) or (i >= self.padding + int(self.N_y.item())):
                            pad_indices.append(ind)
                        elif (j < self.padding) or (j >= self.padding + int(self.N_x.item()) - 1):
                            pad_indices.append(ind)
                        else:
                            mask[ind] = 1
                            k_indices.append(ind_)
                            ind_ = ind_ + 1
                        ind = ind + 1
                        
        return mask.bool()
    
    def B_ij_init(self):
        
        # returns a matrix like K_ij but with dampings: if i,j in design region: B_pp else: B_padding        
        temporary = torch.ones(self.N_.item(), device=self.device, dtype=torch.get_default_dtype()) * self.B_pp_pad
        Bpp = torch.ones(self.N.item(), device=self.device, dtype=torch.get_default_dtype()) * self.B_pp
        b_padded = temporary.masked_scatter(self.mask, Bpp)
        
        temp = torch.abs(b_padded.reshape(-1, 1) - b_padded)
        if b_padded.all():
            B_ij = b_padded.unsqueeze(1).repeat(1, int(self.N_.item())) * torch.less(temp, 1e-3) + \
            torch.div(b_padded.reshape(-1, 1) * b_padded, b_padded.reshape(-1, 1) + b_padded) * torch.ge(temp, 1e-3)
        else:
            B_ij = b_padded.unsqueeze(1).repeat(1, int(self.N_.item()))

        if B_ij.isnan().any():
            print("B_ij is nan")
        
        return B_ij.to(self.device).type(torch.get_default_dtype())
    
    def XY_init(self):
        
        X_ini = np.zeros(self.N_.item())
        Y_ini = np.zeros(self.N_.item())
        
        
        # particles on a hexagonal lattice with fixed boundary
        if self.lattice == "hex":
            ind = 0
            for i_row in range(1, int(self.N_y_.item())+1):
                if i_row % 2 == 1:
                    for i_col in range(1, int(self.N_x_.item())+1):
                        X_ini[ind] = (i_col-1) * self.d0.item() + 0.5 * self.d0.item()
                        Y_ini[ind] = (i_row-1) * np.sqrt(3) / 2 * self.d0.item() + 0.5 * self.d0.item()
                        ind = ind + 1

                else:
                    for i_col in range(1, int(self.N_x_.item())):
                        X_ini[ind] = (i_col-1) * self.d0.item() + self.d0.item()
                        Y_ini[ind] = (i_row-1) * np.sqrt(3) / 2 * self.d0.item() + 0.5 * self.d0.item()
                        ind = ind + 1
        
        # particles on a square lattice
        if self.lattice == "square":
            for i_row in range(1, int(self.N_y_.item())+1):
                for i_col in range(1, int(self.N_x_.item())+1):
                    ind = (i_row - 1) * int(self.N_x_.item()) + i_col - 1
                    X_ini[ind] = (i_col-1) * self.d0.item() + 0.5 * self.d0.item()
                    Y_ini[ind] = (i_row-1) * self.d0.item() + 0.5 * self.d0.item()
                    
        return torch.from_numpy(X_ini).type(dtype=torch.get_default_dtype()).to(self.device), \
               torch.from_numpy(Y_ini).type(dtype=torch.get_default_dtype()).to(self.device)
