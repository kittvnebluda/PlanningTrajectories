from typing import Callable

from pygame import Rect, Surface, draw

from ..core.map.grid import CellState, Grid, cell_state_colors, grid_to_world
from ..types import Cell, Number
from .screen import ScreenParams, to_screen


def draw_grid(
    grid: Grid,
    surface: Surface,
    scale: Number,
    shift_x: Number,
    shift_y: Number,
    screen_params: ScreenParams,
):
    """Draw the grid occupancy map on a surface."""
    border_width = int(scale * 15)  # BUG: w/ low scale border is not thick enough
    d = scale * screen_params.meter2px
    dh = d // 2
    cell2screen = cell_to_screen_factory(scale, shift_x, shift_y, screen_params)
    for row in range(grid.rows):
        for col in range(grid.cols):
            x, y = cell2screen((row, col))
            rect = Rect(x - dh, y - dh, d, d)
            draw.rect(
                surface,
                cell_state_colors[CellState(grid.occupancy[row, col])],
                rect,
            )

    for r in range(grid.rows + 1):
        start = cell2screen((r, 0))
        end = cell2screen((r, grid.cols))
        draw.line(
            surface,
            (0, 0, 0),
            (start[0] - dh, start[1] - dh),
            (end[0] - dh, end[1] - dh),
            border_width,
        )

    for c in range(grid.cols + 1):
        start = cell2screen((0, c))
        end = cell2screen((grid.rows, c))
        draw.line(
            surface,
            (0, 0, 0),
            (start[0] - dh, start[1] - dh),
            (end[0] - dh, end[1] - dh),
            border_width,
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
    scale_x = screen_params.width / cols / screen_params.meter2px
    scale_y = screen_params.height / rows / screen_params.meter2px
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
        width=screen_width_px, height=screen_height_px, meter2px=meters2px
    )

    return scale, shift_x, shift_y, screen
