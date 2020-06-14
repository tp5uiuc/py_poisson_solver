import numpy as np
from numpy.fft import fft2, fftfreq, ifft2

try:
    from poisson_solver_order import PoissonOrder
except ImportError:
    from .poisson_solver_order import PoissonOrder


class UnboundedPoissonSolver:
    def __init__(self, grid_size: int, dx: float, order: PoissonOrder):
        if not isinstance(order, PoissonOrder):
            raise RuntimeError("order passed in is not a valid PoissonOrder!")
        self.fourier_GF = UnboundedPoissonSolver.construct_greens_function(
            grid_size, dx, order
        )

        self.dx = dx
        self.grid_size = grid_size
        self.rhs_doubled = np.zeros((2 * self.grid_size, 2 * self.grid_size))

    @staticmethod
    def construct_greens_function(grid_size: int, dx: float, order: PoissonOrder):
        x_double = np.linspace(0, 2.0 - dx, 2 * grid_size)
        X_double, Y_double = np.meshgrid(x_double, x_double)
        even_reflected_distance = np.sqrt(
            np.minimum(X_double, 2.0 - X_double) ** 2
            + np.minimum(Y_double, 2.0 - Y_double) ** 2
        )

        # non-regularized
        if order is PoissonOrder.NON_REGULARIZED:
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                GF = np.log(even_reflected_distance) / 2 / np.pi
            # Regularization term
            GF[0, 0] = (2 * np.log(dx / np.sqrt(np.pi)) - 1) / 4 / np.pi

        else:
            from scipy.special import exp1

            regularize_epsilon = 2.0 * dx
            eulers_constant = np.euler_gamma
            zero_val_common = 0.5 * eulers_constant - np.log(
                np.sqrt(2.0) * regularize_epsilon
            )
            if order is PoissonOrder.SECOND_ORDER:
                zero_val = zero_val_common + 0.0

                def q_m(dist):
                    return 0.0 + 0.0 * dist

            elif order is PoissonOrder.FOURTH_ORDER:
                zero_val = zero_val_common + 0.5

                def q_m(dist):
                    return 0.5 + 0.0 * dist

            elif order is PoissonOrder.SIXTH_ORDER:
                zero_val = zero_val_common + 0.75

                def q_m(dist):
                    return 0.75 - dist ** 2 / 8.0

            elif order is PoissonOrder.EIGTH_ORDER:
                zero_val = zero_val_common + 11.0 / 12.0

                def q_m(dist):
                    return 11.0 / 12.0 + dist ** 2 * (dist ** 2 / 48.0 - 7.0 / 24.0)

            elif order is PoissonOrder.TENTH_ORDER:
                zero_val = zero_val_common + 25.0 / 24.0

                def q_m(dist):
                    return 25.0 / 24.0 + dist ** 2 * (
                        dist ** 2 * (13.0 / 192.0 - dist ** 2 / 384.0) - 23.0 / 48.0
                    )

            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                # Compared to paper, add a negative sign
                GF = (
                    np.log(even_reflected_distance)
                    - (
                        q_m(even_reflected_distance / regularize_epsilon)
                        * np.exp(
                            -(even_reflected_distance ** 2)
                            / 2.0
                            / regularize_epsilon ** 2
                        )
                    )
                    + 0.5
                    * exp1(even_reflected_distance ** 2 / 2.0 / regularize_epsilon ** 2)
                )
            GF[0, 0] = -zero_val
            GF /= 2.0 * np.pi

        return fft2(GF)

    def solve(self, rhs):
        r""" solve \delta^2 u = -f
        Careful : pass in -f to be solved!
        """
        self.rhs_doubled[: self.grid_size, : self.grid_size] = -rhs.copy()
        sol = ifft2((fft2(self.rhs_doubled) * self.fourier_GF))
        return sol[: self.grid_size, : self.grid_size] * self.dx * self.dx


def make_bump(inp_X, inp_Y, center, radius):
    R = np.sqrt((inp_X - center[0]) ** 2 + (inp_Y - center[1]) ** 2)
    c = 20.0
    prefactor = 1.0
    idx = R < radius
    psi_expected = 0.0 * R
    psi_expected[idx] = prefactor * np.exp(
        -(radius ** 2) * c / (radius ** 2 - R[idx] ** 2)
    )
    """
    import sympy as sym
    from sympy.abc import R, c, r
    expr = sym.exp(-c * r**2/(r**2 - R**2))
    res = sym.diff(R * sym.diff(expr, R), R) / R
    sym.simplify(res)
    """
    vorticity_to_be_fed_in = 0.0 * R
    vorticity_to_be_fed_in[idx] = -prefactor * (
        4
        * c
        * radius ** 2
        * (
            R[idx] ** 2 * c * radius ** 2
            + 2 * R[idx] ** 2 * (R[idx] ** 2 - radius ** 2)
            - (R[idx] ** 2 - radius ** 2) ** 2
        )
        * np.exp(c * radius ** 2 / (R[idx] ** 2 - radius ** 2))
        / (R[idx] ** 2 - radius ** 2) ** 4
    )

    return psi_expected, vorticity_to_be_fed_in


def visualize_bump(n_points=512):
    dx = 1.0 / n_points
    x = np.linspace(0.5 * dx, 1.0 - 0.5 * dx, n_points)
    X, Y = np.meshgrid(x, x)

    # bump properties
    bump_radius = 0.4
    bump_center = [0.5, 0.5]

    psi_e, vort = make_bump(X, Y, bump_center, bump_radius)

    normalized_psi = psi_e / np.amax(np.abs(psi_e))
    normalized_vort = vort / np.amax(np.abs(vort))

    from matplotlib import pyplot as plt

    fig = plt.figure(1, figsize=(8, 8))
    ax = fig.add_subplot(221)
    contr = ax.contourf(X, Y, normalized_psi)
    plt.colorbar(contr)

    ax = fig.add_subplot(222)
    # Take slice at center
    ax.plot(x, normalized_psi[n_points // 2], "b-")

    ax = fig.add_subplot(223)
    contr = ax.contourf(X, Y, normalized_vort)
    plt.colorbar(contr)

    ax = fig.add_subplot(224)
    # Take slice at center
    ax.plot(x, normalized_vort[n_points // 2], "r-")

    plt.show()


def test_poisson_solve_unbounded(n_points=64, draw=False, order=PoissonOrder(0)):
    dx = 1.0 / n_points
    x = np.linspace(0.5 * dx, 1.0 - 0.5 * dx, n_points)
    X, Y = np.meshgrid(x, x)

    # bump properties
    bump_radius = 0.5
    bump_center = [0.5, 0.5]

    psi_analytical, vort = make_bump(X, Y, bump_center, bump_radius)

    poisson_solver = UnboundedPoissonSolver(n_points, dx, order)
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


def report_errors_in_solving_unbounded_poisson(order):
    for i in range(4, 11):
        n_points = 2 ** i
        test_poisson_solve_unbounded(n_points, draw=False, order=order)


if __name__ == "__main__":
    # visualize_bump()
    # test_poisson_solve_unbounded(n_points=64, draw=True, order=PoissonOrder(2))
    # report_errors_in_solving_unbounded_poisson(order=PoissonOrder(2))
    for i_order in range(0, 11, 2):
        print("-------")
        print(i_order)
        print("-------")
        report_errors_in_solving_unbounded_poisson(order=PoissonOrder(i_order))
