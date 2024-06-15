"""
    Partially adopted from: [1] https://github.com/a-papp/SpinTorch
                            [2] https://github.com/fancompute/wavetorch
"""

__author__ = 'Atoosa Parsa'
__copyright__ = 'Copyright 2024, Atoosa Parsa'
__credits__ = 'Atoosa Parsa'
__license__ = 'MIT License'
__version__ = '2.0.0'
__maintainer__ = 'Atoosa Parsa'
__email__ = 'atoosa.parsa@gmail.com'
__status__ = "Dev"


from .solver import MDSolver
from .skip_solver import SkipMDSolver
from .dynamics import Dynamics
from .material import GranularMaterial, VariableStiffnessParticles
from .probe import Probe, IntensityProbe
from .source import Source
from .binarize import Binarize, binarize
from . import plot