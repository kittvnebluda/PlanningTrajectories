from enum import Enum


class CellState(Enum):
    OCCUPIED = 0
    FREE = 1
    DENSE = 2
    START = 3
    GOAL = 4


cell_state_colors: dict[CellState, tuple[int, int, int]] = {
    CellState.OCCUPIED: (60, 60, 60),
    CellState.FREE: (255, 255, 255),
    CellState.DENSE: (200, 255, 200),
    CellState.START: (255, 0, 0),
    CellState.GOAL: (0, 0, 255),
}
