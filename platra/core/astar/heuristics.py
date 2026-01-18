from numba import njit

from platra.typings import Number


@njit
def h_euclidian(c1: tuple[Number, Number], c2: tuple[Number, Number]):
    return ((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2) ** (1 / 2)
