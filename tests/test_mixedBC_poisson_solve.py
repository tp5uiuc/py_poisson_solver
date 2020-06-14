import numpy as np
import pytest

from poisson_solver import MixedBCPoissonSolver, PoissonOrder
from trial_functions import make_bumps_in_y, make_oned_periodic_bump


class TestMixedBCPoissonSolver:
    @pytest.mark.parametrize("order", range(0, 11, 2))
    @pytest.mark.parametrize("periodicity_direction", ["x", "y"])
    def test_error_at_high_resolution(self, order, periodicity_direction):
        n_points = 256
        dx = 1.0 / n_points
        x = np.linspace(0.5 * dx, 1.0 - 0.5 * dx, n_points)
        X, Y = np.meshgrid(x, x)

        # bump properties
        bump_radius = 0.5
        bump_center = [0.5, 0.5]

        psi_analytical, vort = make_oned_periodic_bump(X, Y, bump_center, bump_radius)
        if periodicity_direction == "x":
            psi_analytical = psi_analytical.T
            vort = vort.T

        poisson_solver = MixedBCPoissonSolver(
            n_points, dx, PoissonOrder(order), periodicity_direction
        )
        psi_numerical = poisson_solver.solve(vort)

        psi_net_error = np.linalg.norm(
            np.abs(psi_numerical - psi_analytical), np.inf
        ) / np.linalg.norm(np.abs(psi_analytical), np.inf)

        assert psi_net_error < 1e-2

    @pytest.mark.parametrize("order", range(0, 11, 2))
    @pytest.mark.parametrize("periodicity_direction", ["x", "y"])
    def test_error_with_mean_mode(self, order, periodicity_direction):
        n_points = 256
        dx = 1.0 / n_points
        x = np.linspace(0.5 * dx, 1.0 - 0.5 * dx, n_points)
        X, Y = np.meshgrid(x, x)

        psi_analytical, vort = make_bumps_in_y(X, Y)

        if periodicity_direction == "x":
            psi_analytical = psi_analytical.T
            vort = vort.T

        poisson_solver = MixedBCPoissonSolver(
            n_points, dx, PoissonOrder(order), periodicity_direction
        )
        psi_numerical = poisson_solver.solve(vort)

        psi_net_error = np.linalg.norm(
            np.abs(psi_numerical - psi_analytical), np.inf
        ) / np.linalg.norm(np.abs(psi_analytical), np.inf)

        # TODO : Change error tolerance
        assert psi_net_error < 2e-2
