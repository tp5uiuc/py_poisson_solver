from enum import Enum, unique


@unique
class PoissonOrder(Enum):
    NON_REGULARIZED = 0
    SECOND_ORDER = 2
    FOURTH_ORDER = 4
    SIXTH_ORDER = 6
    EIGTH_ORDER = 8
    TENTH_ORDER = 10
