from __future__ import annotations

from sys import maxsize
from typing import Iterable

import numpy as np

from platra.types import Cell, Number

from .map import CellState
from .prefabs import OCCUPANCY_MATS


class Grid:
    def __init__(self, occupancy: np.ndarray):
        self.occupancy = occupancy.astype(int)
        self.rows, self.cols = self.occupancy.shape

    @property
    def shape(self):
        return self.cols, self.rows

    def in_bounds(self, cell: Cell) -> bool:
        return 0 <= cell[0] < self.rows and 0 <= cell[1] < self.cols

    def is_free(self, cell: Cell) -> bool:
        return (
            self.in_bounds(cell)
            and self.occupancy[cell[0], cell[1]] != CellState.OCCUPIED.value
        )

    def set_cell_state(self, cell: Cell, state: CellState):
        if self.in_bounds(cell):
            self.occupancy[cell[0], cell[1]] = state.value

    def set_obstacle(self, cell: Cell):
        self.set_cell_state(cell, CellState.OCCUPIED)

    def clear_cell(self, cell: Cell):
        self.set_cell_state(cell, CellState.FREE)

    def clear_all(self):
        self.occupancy.fill(0)

    def cost(self, current: Cell, next: Cell) -> int:
        if self.in_bounds(current) and self.in_bounds(next):
            return ((current[0] - next[0]) ** 2 + (current[1] - next[1]) ** 2) ** (
                1 / 2
            ) * self.occupancy[next[0], next[1]]
        return maxsize

    def add_rectangle(
        self, top_left: Cell, bottom_right: Cell, state: CellState = CellState.OCCUPIED
    ):
        for i in range(top_left[0], bottom_right[0]):
            for j in range(top_left[1], bottom_right[1]):
                self.set_cell_state((i, j), state)

    def add_circle(
        self, center: Cell, radius: int, state: CellState = CellState.OCCUPIED
    ):
        for i in range(center[0] - radius, center[0] + radius + 1):
            for j in range(center[1] - radius, center[1] + radius + 1):
                if (i - center[0]) ** 2 + (j - center[1]) ** 2 <= radius**2:
                    self.set_cell_state((i, j), state)

    def add_random_obstacles(self, density: float = 0.2, seed: int | None = None):
        rng = np.random.default_rng(seed)
        for i in range(self.rows):
            for j in range(self.cols):
                if rng.random() < density:
                    self.set_obstacle((i, j))

    @classmethod
    def from_name(cls, name: str) -> Grid:
        if name not in OCCUPANCY_MATS:
            raise Exception(
                f"No such grid prefab. Available: {', '.join(tuple(OCCUPANCY_MATS.keys()))}"
            )
        return Grid(OCCUPANCY_MATS[name])


def grid_to_world(
    cell: Cell, scale: Number, shift_x: Number, shift_y: Number
) -> tuple[float, float]:
    return (
        (cell[1]) * scale + shift_x,
        (-cell[0]) * scale + shift_y,
    )


def cell_array_to_world(
    cells: Iterable[Cell], scale: Number, shift_x: Number, shift_y: Number
) -> np.ndarray:
    return np.array([grid_to_world(cell, scale, shift_x, shift_y) for cell in cells])


def world_to_grid(
    pt: tuple[float, float], scale: Number, shift_x: Number, shift_y: Number
) -> Cell:
    return int((pt[1] - shift_y) / -scale + 1 / 2), int(
        (pt[0] - shift_x) / scale + 1 / 2
    )
