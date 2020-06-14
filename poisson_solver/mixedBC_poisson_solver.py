import numpy as np
from numpy.fft import fft, fft2, fftfreq, ifft2

try:
    from poisson_solver_order import PoissonOrder
except ImportError:
    from .poisson_solver_order import PoissonOrder


class MixedBCPoissonSolver:
    def __init__(
        self,
        grid_size: int,
        dx: float,
        order: PoissonOrder,
        periodicity_direction: str = "y",
    ):
        self.grid_size = grid_size
        self.dx = dx

        if periodicity_direction not in ["x", "y"]:
            raise RuntimeError("periodicity should be either in x or y")

        # By default, periodicity in y direction
        x_size = 2 * self.grid_size
        y_size = self.grid_size
        self.x_slice = slice(None, self.grid_size)
        self.y_slice = slice(None, None)

        if not isinstance(order, PoissonOrder):
            raise RuntimeError("order passed in is not a valid PoissonOrder!")

        self.fourier_GF = MixedBCPoissonSolver.construct_greens_function(
            grid_size, dx, order
        )

        if periodicity_direction == "x":
            x_size, y_size = y_size, x_size
            self.x_slice, self.y_slice = self.y_slice, self.x_slice
            self.fourier_GF = (self.fourier_GF.T).copy()

        # to be consistent with np.meshgrid, do y first then x
        self.rhs_doubled = np.zeros((y_size, x_size))

    @staticmethod
    def construct_greens_function(grid_size: int, dx: float, order: PoissonOrder):
        x_double = np.linspace(0, 2.0 - dx, 2 * grid_size)
        k_y = fftfreq(grid_size).reshape(-1, 1) * 2.0 * np.pi / dx
        even_reflected_distance = np.minimum(x_double, 2.0 - x_double)
        DIST, KY = np.meshgrid(even_reflected_distance, k_y)

        # non-regularized
        if order is PoissonOrder.NON_REGULARIZED:
            GF = 0.0 * DIST
            GF[1:] = -0.5 * np.exp(-np.abs(KY[1:]) * DIST[1:]) / np.abs(KY[1:])

            # Instead of leaving it as 0.0, we solve the 1D poisson equation
            # here, using the corresponding non-regularized greens function
            # In 1D, no regularization is needed at [0,0] because the kernel
            # is non-singular
            GF[0] = 0.5 * np.abs(np.minimum(x_double, 2.0 - x_double))
        else:
            from scipy.special import erf, erfc

            regularize_epsilon = 2.0 * dx
            DIST /= regularize_epsilon
            KY *= regularize_epsilon
            KY = np.abs(KY)

            if order is PoissonOrder.SECOND_ORDER:

                def P_m(s, dist):
                    return 0.0 + 0.0 * dist + 0.0 * s

                def P_m_1D_poisson(dist):
                    return 1.0 + 0.0 * dist

            elif order is PoissonOrder.FOURTH_ORDER:

                def P_m(s, dist):
                    return 0.25 + 0.0 * dist + 0.0 * s

                def P_m_1D_poisson(dist):
                    return 0.5 + 0.0 * dist

            elif order is PoissonOrder.SIXTH_ORDER:

                def P_m(s, dist):
                    return 0.3125 + 0.0625 * (s ** 2 - dist ** 2)

                def P_m_1D_poisson(dist):
                    return (dist ** 2 + 3.0) / 8.0

            elif order is PoissonOrder.EIGTH_ORDER:

                def P_m(s, dist):
                    return (
                        11.0 / 32.0
                        + s ** 2 / 12.0
                        - dist ** 2 / 8.0
                        - dist ** 2 * s ** 2 / 48.0
                        + (s ** 4 + dist ** 4) / 96.0
                    )

                def P_m_1D_poisson(dist):
                    return -(dist ** 4 - 12.0 * dist ** 2 - 15.0) / 48.0

            elif order is PoissonOrder.TENTH_ORDER:

                def P_m(s, dist):
                    return (
                        93.0 / 256.0
                        + 73.0 / 768.0 * s ** 2
                        - 47.0 / 256.0 * dist ** 2
                        - 17.0 / 384.0 * dist ** 2 * s ** 2
                        + 11.0 / 768.0 * s ** 4
                        + 23.0 / 768.0 * dist ** 4
                        + (s ** 2 * dist ** 4 - s ** 4 * dist ** 2) / 256.0
                        + (s ** 6 - dist ** 6) / 768.0
                    )

                def P_m_1D_poisson(dist):
                    return (
                        dist ** 6 - 23.0 * dist ** 4 + 141.0 * dist ** 2 + 105.0
                    ) / 384.0

            # Compared to paper, add a negative sign
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                GF = -(
                    regularize_epsilon
                    / 4.0
                    / KY
                    * (
                        erfc((KY - DIST) / np.sqrt(2.0)) * np.exp(-KY * DIST)
                        + erfc((KY + DIST) / np.sqrt(2.0))
                        * np.minimum(np.exp(KY * DIST), 1e10)  # minimum to prevent nans
                    )
                    + regularize_epsilon
                    * np.sqrt(2.0)
                    / np.sqrt(np.pi)
                    * P_m(KY, DIST)
                    * np.exp(-0.5 * (KY ** 2 + DIST ** 2))
                )
                # Instead of leaving it the zero mode as 0.0,
                # we solve the 1D poisson equation, regulraized upto
                # the order of the solution needed
                GF[0] = regularize_epsilon * (
                    0.5 * erf(DIST[0] / np.sqrt(2.0)) * DIST[0]
                    + P_m_1D_poisson(DIST[0])
                    * np.exp(-0.5 * DIST[0] ** 2)
                    / np.sqrt(2.0 * np.pi)
                )
                # GF[0] = 0.5 * np.abs(np.minimum(x_double, 2.0 - x_double))

        greens_function_in_fourier_domain = np.zeros_like(GF, dtype=np.complex)
        # Do FT of rows only
        for i in range(GF.shape[0]):
            greens_function_in_fourier_domain[i] = fft(GF[i])

        # Set mean mode to 0, dosen't affect solution
        # if order is not PoissonOrder.NON_REGULARIZED:
        #     greens_function_in_fourier_domain[0] = 0.0

        return greens_function_in_fourier_domain

    def solve(self, rhs):
        r""" solve \delta^2 u = -f
        Careful : pass in -f to be solved!
        """
        self.rhs_doubled[self.y_slice, self.x_slice] = -rhs.copy()
        sol = ifft2((fft2(self.rhs_doubled) * self.fourier_GF))
        return sol[self.y_slice, self.x_slice] * self.dx


def make_oned_periodic_smooth(inp_X, inp_Y):
    alpha = 40.0
    psi_expected = np.exp(-alpha * (inp_X - 0.5) ** 2) * np.sin(4.0 * np.pi * inp_Y)
    # print(expected_answer[:, 0])
    vorticity_to_be_fed_in = (
        -(4.0 * alpha ** 2 * (inp_X - 0.5) ** 2 - 2.0 * alpha - 16.0 * np.pi ** 2)
        * psi_expected
    )
    return psi_expected, vorticity_to_be_fed_in


def make_oned_periodic_bump(inp_X, inp_Y, center, radius):
    R = radius
    c = 10.0
    k = 1.0
    shifted_X = inp_X - center[0]
    idx = np.logical_and((shifted_X > -R), (shifted_X < R))
    prefactor = 1.0
    """
    import sympy as sym
    from sympy.abc import x, c, r, k, y
    expr = sym.exp(c * (1 - r**2/(r**2 - x**2))) * sym.sin(2.0 * sym.pi * k * y)
    res = sym.diff(expr, x, 2) + sym.diff(expr , y, 2)
    sym.simplify(res)
    """
    psi_expected = 0.0 * shifted_X
    psi_expected[idx] = (
        prefactor
        * np.exp(c * (-(R ** 2) / (R ** 2 - shifted_X[idx] ** 2) + 1))
        * np.sin(2.0 * np.pi * k * inp_Y[idx])
    )
    vorticity_to_be_fed_in = 0.0 * shifted_X
    vorticity_to_be_fed_in[idx] = -prefactor * (
        -(
            2
            * c
            * R ** 2
            * (
                -2 * c * R ** 2 * shifted_X[idx] ** 2
                + 4 * shifted_X[idx] ** 2 * (R ** 2 - shifted_X[idx] ** 2)
                + (R ** 2 - shifted_X[idx] ** 2) ** 2
            )
            + 4.0 * np.pi ** 2 * k ** 2 * (R ** 2 - shifted_X[idx] ** 2) ** 4
        )
        * np.exp(-c * shifted_X[idx] ** 2 / (R ** 2 - shifted_X[idx] ** 2))
        * np.sin(2.0 * np.pi * k * inp_Y[idx])
        / (R ** 2 - shifted_X[idx] ** 2) ** 4
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


def test_poisson_solve_mixedBC(
    rhs_func, periodicity_direction="y", n_points=64, draw=False, order=PoissonOrder(0)
):
    dx = 1.0 / n_points
    x = np.linspace(0.5 * dx, 1.0 - 0.5 * dx, n_points)
    X, Y = np.meshgrid(x, x)

    # # bump properties
    # bump_radius = 0.5
    # bump_center = [0.5, 0.5]

    # psi_analytical, vort = make_bump(X, Y, bump_center, bump_radius)
    psi_analytical, vort = rhs_func(X, Y)

    poisson_solver = MixedBCPoissonSolver(n_points, dx, order, periodicity_direction)
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


def report_errors_in_solving_mixedBC_poisson(rhs_func, periodic_direction, order):
    for i in range(4, 11):
        n_points = 2 ** i
        test_poisson_solve_mixedBC(
            rhs_func, periodic_direction, n_points, draw=False, order=order
        )


if __name__ == "__main__":
    from functools import partial

    periodicity_direction = "y"
    if periodicity_direction == "y":

        # # bump properties
        # bump_radius = 0.3
        # bump_center = [0.5, 0.5]
        # func = partial(make_oned_periodic_bump, center=bump_center, radius=bump_radius)

        # non-bump
        # func = make_oned_periodic_smooth

        # multiple bumps in
        bump_radius = 0.3
        bump_centers = []

        bump_y_start = 0.3
        bump_y_end = 0.7
        n_bumps = 2
        bump_y_increment = (bump_y_end - bump_y_start) / (n_bumps - 1)
        for i_bump in range(n_bumps):
            bump_centers.append([0.5, bump_y_start + bump_y_increment * i_bump])

        from unbounded_poisson_solver import make_bump

        # Wrap the individual bump functions into one big function
        def func(inp_X, inp_Y):
            psi = 0.0 * inp_X
            vort = 0.0 * inp_X
            for i_bump in range(n_bumps):
                temp_psi, temp_vort = make_bump(
                    inp_X, inp_Y, center=bump_centers[i_bump], radius=bump_radius
                )
                psi += 1e6 * temp_psi
                vort += 1e6 * temp_vort
            return psi, vort

        """
        from unbounded_poisson_solver import make_bump
        bump_radius = 0.3
        bump_center = [0.5, 0.5]
        # func = partial(make_bump, center=bump_center, radius=bump_radius)
        def func(inp_X, inp_Y):
            temp_psi, temp_vort = make_bump(
                inp_X, inp_Y, center=bump_center, radius=bump_radius
            )
            return 1e6 * temp_psi, 1e6 * temp_vort
        """

    if periodicity_direction == "x":
        # bump_properties
        # Wrap around and tranpose
        bump_radius = 0.3
        bump_center = [0.5, 0.5]

        def func(inp_X, inp_Y):
            psi, vort = make_oned_periodic_bump(
                inp_X, inp_Y, center=bump_center, radius=bump_radius
            )
            return psi.T, vort.T

    # visualize_solution(func=func, n_points=512)

    # test_poisson_solve_mixedBC(func, n_points=64, draw=True, order=PoissonOrder(0))

    # report_errors_in_solving_mixedBC_poisson(
    #     func, periodicity_direction, order=PoissonOrder(6)
    # )

    for i_order in range(0, 11, 2):
        print("-------")
        print(i_order)
        print("-------")
        report_errors_in_solving_mixedBC_poisson(
            func, periodicity_direction, order=PoissonOrder(i_order)
        )
