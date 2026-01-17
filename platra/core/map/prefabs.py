import numpy as np

from .map import CellState

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
n_rows, n_cols = 50, 100
_occup_map_parking_lot = np.full((n_rows, n_cols), f)
_occup_map_parking_lot[0, :] = o  # top border
_occup_map_parking_lot[-1, :] = o  # bottom border
_occup_map_parking_lot[:, 0] = o  # left border
_occup_map_parking_lot[:, -1] = o  # right border

_occup_map_parking_lot[17, :13] = o
_occup_map_parking_lot[15:20, 13] = o
_occup_map_parking_lot[15:20, 27] = o

_occup_map_parking_lot[40, :50] = o


OCCUPANCY_MATS: dict[str, np.ndarray] = {
    "lab1": _occup_map_lab1,
    "parking_lot": _occup_map_parking_lot,
}
