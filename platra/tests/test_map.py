from math import isclose

import pytest
from core.map import grid_to_world, world_to_grid


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
    pt = grid_to_world(cell, scale, shift_x, shift_y)
    back = world_to_grid(pt, scale, shift_x, shift_y)

    assert back == cell, f"Round trip failed: {cell} -> {pt} -> {back}"


def test_to_world_coordinates():
    cell = (1, 3)  # row=1, col=3
    pt = grid_to_world(cell, scale=1.0, shift_x=0.0, shift_y=0.0)

    expected_x = 3
    expected_y = -1

    assert isclose(pt[0], expected_x), f"{pt[0]} != {expected_x}"
    assert isclose(pt[1], expected_y), f"{pt[1]} != {expected_y}"


def test_to_grid_coordinates():
    pt = (-9, 9)  # should map to (2, 3)
    cell = world_to_grid(pt, 1, -10, 10)

    assert cell == (1, 1)
