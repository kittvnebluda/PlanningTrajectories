import numpy as np

from .types import CellState

o = CellState.OCCUPIED.value
f = CellState.FREE.value
s = CellState.START.value
g = CellState.GOAL.value
d = CellState.DENSE.value

_occup_map_lab1 = np.array(
    [
        [f, f, f, o, f, f, f, f, f, f],
        [f, f, f, o, f, f, f, f, f, f],
        [f, f, f, o, f, o, o, o, f, f],
        [f, f, f, o, f, f, f, o, o, f],
        [f, o, f, o, f, o, o, o, f, f],
        [f, o, f, o, f, o, f, f, f, f],
        [f, f, f, o, f, o, f, o, f, f],
        [f, f, d, o, f, o, f, o, o, o],
        [f, f, d, d, f, o, f, f, f, f],
        [f, f, f, f, f, o, f, f, f, f],
    ]
)
OCCUPANCY_MATS: dict[str, np.ndarray] = {
    "lab1": _occup_map_lab1,
}
