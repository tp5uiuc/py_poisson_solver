import numpy as np


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


def make_bumps_in_y(inp_X, inp_Y, **kwargs):
    bump_centers = kwargs.get("bump_centers", None)
    bump_radii = kwargs.get("bump_radii", None)

    n_bumps = 2
    if bump_centers is None:
        bump_centers = []
        bump_y_start = 0.3
        bump_y_end = 0.7
        bump_y_increment = (bump_y_end - bump_y_start) / (n_bumps - 1)
        for i_bump in range(n_bumps):
            bump_centers.append([0.5, bump_y_start + bump_y_increment * i_bump])

    if bump_radii is None:
        bump_radii = []
        bump_radii_start = 0.3
        bump_radii_end = 0.4
        bump_radii_increment = (bump_radii_end - bump_radii_start) / (n_bumps - 1)
        for i_bump in range(n_bumps):
            bump_radii.append(bump_radii_start + bump_radii_increment * i_bump)

    psi = 0.0 * inp_X
    vort = 0.0 * inp_X
    for center, radius in zip(bump_centers, bump_radii):
        temp_psi, temp_vort = make_bump(inp_X, inp_Y, center=center, radius=radius)
        psi += 1e2 * temp_psi
        vort += 1e2 * temp_vort
    return psi, vort


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


def make_periodic_profile(inp_X, inp_Y, fx, fy):
    psi_expected = 1e2 * np.sin(2 * np.pi * fx * inp_X) * np.sin(2 * np.pi * fy * inp_Y)
    psi_expected /= (4 * np.pi ** 2) * (fx ** 2 + fy ** 2)
    vorticity_to_be_fed_in = (
        1e2 * np.sin(2 * np.pi * fx * inp_X) * np.sin(2 * np.pi * fy * inp_Y)
    )

    return psi_expected, vorticity_to_be_fed_in
