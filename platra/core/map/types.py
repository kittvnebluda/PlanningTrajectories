from enum import Enum


class CellState(Enum):
    OCCUPIED = 0
    FREE = 1
    DENSE = 2
    START = 3
    GOAL = 4
