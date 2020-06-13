import numpy as np
from numpy.fft import fft2, ifft2, fftfreq, fft

try:
    from poisson_solver_order import PoissonOrder
except ImportError:
    from .poisson_solver_order import PoissonOrder


class PeriodicPoissonSolver:
    def __init__(
        self, grid_size: int, dx: float, order: PoissonOrder = PoissonOrder(0)
    ):
        self.grid_size = grid_size

        kx = np.tile(np.fft.fftfreq(self.grid_size), (self.grid_size, 1))
        ky = kx.T

        self.laplacian_in_fourier_domain = (
            -4 * np.pi ** 2 * (kx ** 2 + ky ** 2) / dx / dx
        )
        self.laplacian_in_fourier_domain[0, 0] = np.inf

    def solve(self, rhs):
        r""" solve \delta^2 u = -f
        Careful : pass in -f to be solved!
        """
        sol = ifft2(fft2(-rhs) / self.laplacian_in_fourier_domain)
        return sol


def make_periodic_profile(inp_X, inp_Y, fx, fy):
    psi_expected = 1e2 * np.sin(2 * np.pi * fx * inp_X) * np.sin(2 * np.pi * fy * inp_Y)
    psi_expected /= (4 * np.pi ** 2) * (fx ** 2 + fy ** 2)
    vorticity_to_be_fed_in = (
        1e2 * np.sin(2 * np.pi * fx * inp_X) * np.sin(2 * np.pi * fy * inp_Y)
    )

    return psi_expected, vorticity_to_be_fed_in


def visualize_solution(func, n_points=512):
    dx = 1.0 / n_points
    x = np.linspace(0.5 * dx, 1.0 - 0.5 * dx, n_points)
    X, Y = np.meshgrid(x, x)

    psi_e, vort = func(X, Y)

    normalized_psi = psi_e / np.amax(np.abs(psi_e))
    normalized_vort = vort / np.amax(np.abs(vort))

    from matplotlib import pyplot as plt

    fig = plt.figure(1, figsize=(8, 8))
    ax = fig.add_subplot(221)
    contr = ax.contourf(X, Y, normalized_psi)
    plt.colorbar(contr)

    slice_idx = n_points // 3
    ax = fig.add_subplot(222)
    # Take slice at center
    ax.plot(x, normalized_psi[slice_idx], "b-")

    ax = fig.add_subplot(223)
    contr = ax.contourf(X, Y, normalized_vort)
    plt.colorbar(contr)

    ax = fig.add_subplot(224)
    # Take slice at center
    ax.plot(x, normalized_vort[slice_idx], "r-")

    plt.show()


def test_poisson_solve_periodic(rhs_func, n_points=64, draw=False):
    dx = 1.0 / n_points
    x = np.linspace(0.5 * dx, 1.0 - 0.5 * dx, n_points)
    X, Y = np.meshgrid(x, x)

    psi_analytical, vort = rhs_func(X, Y)

    poisson_solver = PeriodicPoissonSolver(n_points, dx)
    psi_numerical = poisson_solver.solve(vort)

    if draw:
        from matplotlib import pyplot as plt

        fig = plt.figure(1, figsize=(10, 5))
        ax = fig.add_subplot(121)
        field_error = np.abs(psi_numerical - psi_analytical)
        contr = ax.contourf(X, Y, field_error, cmap="magma")
        plt.colorbar(contr)

        ax = fig.add_subplot(122)
        ax.plot(
            x, psi_analytical[n_points // 2], "k--", x, psi_numerical[n_points // 2]
        )

        plt.show()

    psi_net_error = np.linalg.norm(
        np.abs(psi_numerical - psi_analytical), np.inf
    ) / np.linalg.norm(np.abs(psi_analytical), np.inf)
    print(dx, "\t", psi_net_error)


def report_errors_in_solving_periodic_poisson(rhs_func):
    for i in range(4, 11):
        n_points = 2 ** i
        test_poisson_solve_periodic(rhs_func, n_points, draw=False)


if __name__ == "__main__":
    from functools import partial

    # solution properties
    fx = 12.0
    fy = 18.0
    func = partial(make_periodic_profile, fx=fx, fy=fy)

    # visualize_solution(func=func, n_points=512)

    # test_poisson_solve_periodic(
    #     func, n_points=64, draw=True
    # )

    # spectrally accurate, no need order of accuracy
    report_errors_in_solving_periodic_poisson(func)
