from poisson_solver import PoissonOrder, UnboundedPoissonSolver
import pytest
import numpy as np
from trial_functions import make_bump

class TestUnboundedPoissonSolver:

    @pytest.mark.parametrize("order", range(0, 11, 2))
    def test_error_at_high_resolution(self, order):
        n_points = 256
        dx = 1.0 / n_points
        x = np.linspace(0.5 * dx, 1.0 - 0.5 * dx, n_points)
        X, Y = np.meshgrid(x, x)

        # bump properties
        bump_radius = 0.5
        bump_center = [0.5, 0.5]

        psi_analytical, vort = make_bump(X, Y, bump_center, bump_radius)

        poisson_solver = UnboundedPoissonSolver(n_points, dx, PoissonOrder(order))
        psi_numerical = poisson_solver.solve(vort)

        psi_net_error = np.linalg.norm(
            np.abs(psi_numerical - psi_analytical), np.inf
        ) / np.linalg.norm(np.abs(psi_analytical), np.inf)

        assert psi_net_error < 1e-2
