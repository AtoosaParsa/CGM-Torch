""" Binarization layer with pseudo-gradient

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

class Binarize(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x):
        return torch.ones(x.size())*torch.less(x, 5.0) + 10 * torch.ones(x.size())*torch.ge(x, 5.0)
    @staticmethod
    def backward(ctx,grad_output):
        return grad_output

def binarize(x):
    return Binarize.apply(x)
