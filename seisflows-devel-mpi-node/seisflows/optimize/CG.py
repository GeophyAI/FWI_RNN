
"""
ConjugateGradient
"""
import os
import numpy as np
from glob import glob

from seisflows import logger
from seisflows.optimize.gradient import Gradient
from seisflows.tools import unix
from seisflows.tools.msg import DEG
from seisflows.tools.math import angle
from seisflows.tools.specfem import read_fortran_binary
from seisflows.plugins import line_search as line_search_dir

class CG(Gradient):

    """
        Deterministic step-length.
        Gauthier et al.,1986, Geophysics, Vol 51, No.7

        Step <1>. Calculate the search direction p_new.
        Step <2>. Calculate the <alpha_try> based on the search direction.
        Step <3>. m_try = current_model + alpha_try * p_new
        Step <4>. Forward simulation based on m_try and calculate the alpha.
        Step <5>. m_new = current_model + alpha * p_new
    """
    __doc__ = Gradient.__doc__ + __doc__
    def __init__(self, **kwargs):
        """Instantiate CG specific parameters"""

        self.grad_history = []
        self.iteration = 0

        super().__init__(**kwargs)

        # Overwrite user-chosen line search. L-BFGS requires 'Backtrack'ing LS
        if self.line_search_method.title() != "Customer":
            logger.warning(f"CG optimization requires 'Customer'ing line "
                           f"search. Overwriting '{self.line_search_method}'")
            self.line_search_method = "Customer"
            self._line_search = getattr(
                line_search_dir, self.line_search_method)()

    def compute_direction(self):
        """
        Call on the L-BFGS optimization machinery to compute a search
        direction using internally stored memory of previous gradients.

        The potential outcomes when computing direction with L-BFGS:
        1. First iteration of L-BFGS optimization, search direction is defined
            as the inverse gradient
        2. L-BFGS internal iteration ticks over the maximum allowable number of
            iterations, force a restart condition, search direction is the
            inverse gradient
        3. New search direction vector is too far from previous direction,
            force a restart, search direction is inverse gradient
        4. New search direction is acceptably angled from previous,
            becomes the new search direction

        TODO do we need to precondition L-BFGS?

        :rtype: seisflows.tools.specfem.Model
        :return: search direction as a Model instance
        """
        # Load gradients
        grad_history_files = sorted(glob(os.path.join(self.path.output, "GRADIENT_*", "*")))
        self.iteration = len(grad_history_files)

        #assert self.iteration==len(grad_history), "Iteration number error, compute direction failed"

        # Load the current gradient direction, which is the CG search
        # direction if this is the first iteration
        g = self.load_vector("g_new")
        p_new = g.copy()

        if self.iteration == 1:
            logger.info("CG info:first iteration, default to 'Gradient' descent")
            p_new.update(vector=-1 * g.vector)
            restarted = False
        # Normal CG direction computation
        else:
            logger.info("CG info: using conjugate gradient")
            """beta: PRP method."""
            """beta(k) = g(k)*[g(k)-g(k-1)]/[g(k-1)*g(k-1)]"""
            gk0 = read_fortran_binary(grad_history_files[-1]) # current gradient, i.e. p_new
            gk1 = read_fortran_binary(grad_history_files[-2]) # last gradient 
            up = np.sum(gk0*(gk0-gk1))
            down = np.sum(gk1*gk1)
            beta = up / down
            p_new.update(vector= -gk0 + beta * gk1)

        return p_new
