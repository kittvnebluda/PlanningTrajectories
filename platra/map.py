from enum import Enum
from typing import Callable, Iterable, Iterator

import numpy as np
from pygame import Rect, Surface, draw

from .utils import ScreenParams, to_screen
from .type_aliases import Number, Cell


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


class Grid:
    def __init__(self, occupancy: np.ndarray):
        self.occupancy = occupancy.astype(int)
        self.rows, self.cols = self.occupancy.shape
        self._directions = [
            (-1, 0),
            (0, -1),
            (1, 0),
            (0, 1),
            (-1, -1),
            (1, -1),
            (1, 1),
            (-1, 1),
        ]

    def in_bounds(self, cell: Cell) -> bool:
        return 0 <= cell[0] < self.rows and 0 <= cell[1] < self.cols

    def is_free(self, cell: Cell) -> bool:
        return (
            self.in_bounds(cell)
            and self.occupancy[cell[0], cell[1]] != CellState.OCCUPIED.value
        )

    def neighbors(self, cell: Cell) -> Iterator[Cell]:
        i, j = cell

        for di, dj in self._directions:
            ni, nj = i + di, j + dj

            if not self.is_free((ni, nj)):
                continue

            if di != 0 and dj != 0:
                if not (self.is_free((i + di, j)) and self.is_free((i, j + dj))):
                    continue

            yield ni, nj

    def set_cell_state(self, cell: Cell, state: CellState):
        if self.in_bounds(cell):
            self.occupancy[cell[0], cell[1]] = state.value

    def set_obstacle(self, cell: Cell):
        self.set_cell_state(cell, CellState.OCCUPIED)

    def clear_cell(self, cell: Cell):
        self.set_cell_state(cell, CellState.FREE)

    def cost(self, current: Cell, next: Cell) -> int:  # pyright: ignore[reportReturnType]
        if self.in_bounds(current) and self.in_bounds(next):
            return ((current[0] - next[0]) ** 2 + (current[1] - next[1]) ** 2) ** (
                1 / 2
            ) * self.occupancy[next[0], next[1]]

    @property
    def shape(self):
        return self.cols, self.rows

    def draw(
        self,
        surface: Surface,
        scale: Number,
        shift_x: Number,
        shift_y: Number,
        screen_params: ScreenParams,
    ):
        """Draw the grid occupancy map on a surface."""
        border_width = int(scale * 15)  # TODO: w/ low scale border is not thick enough
        d = scale * screen_params.meters2px
        dh = d // 2
        cell2screen = cell_to_screen_factory(scale, shift_x, shift_y, screen_params)
        for row in range(self.rows):
            for col in range(self.cols):
                x, y = cell2screen((row, col))
                rect = Rect(x - dh, y - dh, d, d)
                draw.rect(
                    surface,
                    cell_state_colors[CellState(self.occupancy[row, col])],
                    rect,
                )

        for r in range(self.rows + 1):
            start = cell2screen((r, 0))
            end = cell2screen((r, self.cols))
            draw.line(
                surface,
                (0, 0, 0),
                (start[0] - dh, start[1] - dh),
                (end[0] - dh, end[1] - dh),
                border_width,
            )

        for c in range(self.cols + 1):
            start = cell2screen((0, c))
            end = cell2screen((self.rows, c))
            draw.line(
                surface,
                (0, 0, 0),
                (start[0] - dh, start[1] - dh),
                (end[0] - dh, end[1] - dh),
                border_width,
            )

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

    def clear_all(self):
        self.occupancy.fill(0)


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


def cell_to_screen_factory(
    scale: Number, shift_x: Number, shift_y: Number, screen_params: ScreenParams
) -> Callable[[Cell], Cell]:
    def cell_to_screen(cell: Cell) -> Cell:
        x, y = grid_to_world(cell, scale, shift_x, shift_y)
        return to_screen(x, y, screen_params)

    return cell_to_screen


def fit_map_to_screen(
    screen_params: ScreenParams, cols: int, rows: int, fill=False
) -> tuple[Number, Number, Number]:
    """Returns scale, shift_x, shift_y for grid drawing"""
    scale_x = screen_params.width / cols / screen_params.meters2px
    scale_y = screen_params.height / rows / screen_params.meters2px
    choice_func = max if fill else min
    scale = choice_func(scale_x, scale_y)
    scale_half = scale / 2
    shift_x = -cols / 2 * scale + scale_half
    shift_y = rows / 2 * scale - scale_half
    return scale, shift_x, shift_y


def fit_screen_to_map(
    cols: int, rows: int, scale: float, meters2px: float = 100
) -> tuple[float, float, float, ScreenParams]:
    """
    Given number of grid columns, rows, and a scale (in meters),
    generate screen parameters so that the grid fits and is centered.

    Args:
        cols: number of columns in the grid
        rows: number of rows in the grid
        scale: scale factor (meters per cell)
        meters2px: conversion factor from meters to pixels

    Returns:
        (scale, shift_x, shift_y, ScreenParams)
    """

    map_width_m = cols * scale
    map_height_m = rows * scale

    screen_width_px = int(map_width_m * meters2px)
    screen_height_px = int(map_height_m * meters2px)

    shift_x = -map_width_m / 2 + scale / 2
    shift_y = map_height_m / 2 - scale / 2

    screen = ScreenParams(
        width=screen_width_px, height=screen_height_px, meters2px=meters2px
    )

    return scale, shift_x, shift_y, screen
