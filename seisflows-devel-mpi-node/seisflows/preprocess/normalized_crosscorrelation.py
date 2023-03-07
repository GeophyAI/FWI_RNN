
import sys
import numpy as np

from os.path import exists
from obspy.core import Stream, Trace

from seisflows.plugins import adjoint, misfit
from seisflows.tools import unix
from seisflows.tools.config import ParameterError, custom_import

PAR = sys.modules['seisflows_parameters']
PATH = sys.modules['seisflows_paths']

system = sys.modules['seisflows_system']


class normalized_crosscorrelation(custom_import('preprocess', 'base')):
    """ Normalized cross-correlation misfit function's data processing class

      Adds Normalized cross-correlation misfit  data misfit functions to base class
    """

    def check(self):
        """ Checks parameters, paths, and dependencies
        """
        super(normalized_crosscorrelation, self).check()

        assert PAR.MISFIT in [
            'NormalizedCrossCorrelation']


    def sum_residuals(self,paths):
        """ Sums squares of residuals
        """
        total_misfit = 0.
        for path in paths:
            total_misfit += np.sum(np.loadtxt(path))
        return total_misfit


