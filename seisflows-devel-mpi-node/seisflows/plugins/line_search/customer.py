import numpy as np
from seisflows.plugins.line_search.bracket import Bracket

class Customer(Bracket):

    def calculate_step_length(self, model, gradient, status='Try'):
        """Calculate alpha_try."""
        """max{alpha_try * g_n} <= (max{M(x)_n})/(100)"""
        if status == 'Try':
            max_grad = np.abs(gradient).max()
            max_vel = np.abs(model).max()
            alpha = max_vel / (max_grad * 100)
        return alpha, "PASS"
