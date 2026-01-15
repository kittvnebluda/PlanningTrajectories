from typing import Iterator

from platra.types import Cell

from ..map import Grid

DIRECTIONS_8 = [
    (-1, 0),
    (0, -1),
    (1, 0),
    (0, 1),
    (-1, -1),
    (1, -1),
    (1, 1),
    (-1, 1),
]


def grid_neighbors_8(grid: Grid, cell: Cell) -> Iterator[Cell]:
    i, j = cell

    for di, dj in DIRECTIONS_8:
        ni, nj = i + di, j + dj

        if not grid.is_free((ni, nj)):
            continue

        if di != 0 and dj != 0:
            if not (grid.is_free((i + di, j)) and grid.is_free((i, j + dj))):
                continue

        yield ni, nj
