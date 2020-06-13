__all__ = [
    "PoissonOrder",
    "PeriodicPoissonSolver",
    "UnboundedPoissonSolver",
    "MixedBCPoissonSolver",
    "make_poisson_solver",
]

from poisson_solver.poisson_solver_order import PoissonOrder
from poisson_solver.periodic_poisson_solver import PeriodicPoissonSolver
from poisson_solver.unbounded_poisson_solver import UnboundedPoissonSolver
from poisson_solver.mixedBC_poisson_solver import MixedBCPoissonSolver


def make_poisson_solver(
    grid_size: int,
    dx: float,
    x_boundary_condition: str,
    y_boundary_condition: str,
    order_of_accuracy: PoissonOrder = PoissonOrder(4),  # not put int to make explicit
):
    allowed_boundary_conditions = ["periodic", "unbounded"]
    if x_boundary_condition not in allowed_boundary_conditions:
        raise RuntimeError("x boundary condition should be periodic or unbounded")
    if y_boundary_condition not in allowed_boundary_conditions:
        raise RuntimeError("y boundary condition should be periodic or unbounded")
    try:
        if x_boundary_condition == "periodic":
            if y_boundary_condition == "periodic":
                solver = PeriodicPoissonSolver(grid_size, dx)
            else:
                solver = MixedBCPoissonSolver(
                    grid_size, dx, order=(order_of_accuracy), periodicity_direction="x",
                )
        else:
            if y_boundary_condition == "periodic":
                solver = MixedBCPoissonSolver(
                    grid_size, dx, order=(order_of_accuracy), periodicity_direction="y",
                )
            else:
                solver = UnboundedPoissonSolver(
                    grid_size, dx, order=(order_of_accuracy)
                )
    except:  # noqa
        # TODO : better handling
        print("Unexpected error")
        raise

    return solver
