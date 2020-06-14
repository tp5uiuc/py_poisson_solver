import numpy as np
import pytest

from poisson_solver import PeriodicPoissonSolver, PoissonOrder
from trial_functions import make_periodic_profile


class TestPeriodicPoissonSolver:
    def test_error_at_high_resolution(self):
        n_points = 256
        dx = 1.0 / n_points
        x = np.linspace(0.5 * dx, 1.0 - 0.5 * dx, n_points)
        X, Y = np.meshgrid(x, x)

        fx = 12.0
        fy = 18.0
        psi_analytical, vort = make_periodic_profile(X, Y, fx, fy)

        poisson_solver = PeriodicPoissonSolver(n_points, dx)
        psi_numerical = poisson_solver.solve(vort)

        psi_net_error = np.linalg.norm(
            np.abs(psi_numerical - psi_analytical), np.inf
        ) / np.linalg.norm(np.abs(psi_analytical), np.inf)

        assert psi_net_error < 1e-2
