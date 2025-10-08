from math import isclose

import numpy as np
import pytest
from pygame import Vector2

from platra.map import Grid


@pytest.mark.parametrize(
    "scale,shift_x,shift_y",
    [
        (1.0, 0.0, 0.0),
        (2.0, 10.0, -5.0),
        (0.5, -3.0, 7.0),
    ],
)
@pytest.mark.parametrize(
    "cell",
    [
        (0, 0),
        (1, 2),
        (3, 4),
        (10, 5),
    ],
)
def test_round_trip(scale, shift_x, shift_y, cell):
    grid = Grid(
        occupancy=np.array([[0] * 20 for _ in range(20)]),
    )

    pt = grid.to_world(cell, scale, shift_x, shift_y)
    back = grid.to_grid(pt, scale, shift_x, shift_y)

    assert back == cell, f"Round trip failed: {cell} -> {pt} -> {back}"


def test_to_world_coordinates():
    grid = Grid(
        occupancy=np.array([[0] * 5 for _ in range(5)]),
    )
    cell = (1, 3)  # row=1, col=3
    pt = grid.to_world(cell, scale=1.0, shift_x=0.0, shift_y=0.0)

    expected_x = 3
    expected_y = -1

    assert isclose(pt.x, expected_x), f"{pt.x} != {expected_x}"
    assert isclose(pt.y, expected_y), f"{pt.y} != {expected_y}"


def test_to_grid_coordinates():
    grid = Grid(
        occupancy=np.array([[0] * 5 for _ in range(5)]),
    )
    pt = Vector2(-9, 9)  # should map to (2, 3)
    cell = grid.to_grid(pt, 1, -10, 10)

    assert cell == (1, 1)
