from enum import Enum
from typing import Callable, Iterator

import numpy as np
import pygame
from pygame import Rect, Surface, Vector2, draw

from .utils import METERS2PX, SCRN_HEIGHT, SCRN_WIDTH, vec2screen
from .types import Number, Cell


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

    @classmethod
    def to_world(
        cls, cell: Cell, scale: Number, shift_x: Number, shift_y: Number
    ) -> Vector2:
        return Vector2(cell[1] * scale + shift_x, -cell[0] * scale + shift_y)

    @classmethod
    def to_grid(
        cls, pt: Vector2, scale: Number, shift_x: Number, shift_y: Number
    ) -> Cell:
        return int((pt.y - shift_y) / -scale + scale / 2), int(
            (pt.x - shift_x) / scale + scale / 2
        )

    @classmethod
    def cell_to_screen_factory(
        cls, scale: Number, shift_x: Number, shift_y: Number
    ) -> Callable[[Cell], Cell]:
        def cell_to_screen(cell: Cell) -> Cell:
            return vec2screen(Grid.to_world(cell, scale, shift_x, shift_y))

        return cell_to_screen

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
    ):
        """Draw the grid occupancy map on a surface."""
        border_width = int(scale * 15)
        d = scale * METERS2PX
        dh = d // 2
        cell2screen = self.cell_to_screen_factory(scale, shift_x, shift_y)
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


def fit_map_to_screen(
    screen_size: tuple[int, int], cols: int, rows: int, fill=False
) -> tuple[Number, Number, Number]:
    """Returns scale, shift_x, shift_y for grid drawing"""
    scale_x = screen_size[0] / cols / METERS2PX
    scale_y = screen_size[1] / rows / METERS2PX
    choice_fun = max if fill else min
    scale = choice_fun(scale_x, scale_y)
    scale_half = scale / 2
    shift_x = -cols / 2 * scale + scale_half
    shift_y = rows / 2 * scale - scale_half
    return scale, shift_x, shift_y


def _test_grid_draw(args):
    grid = Grid(np.zeros((30, 54)))
    grid.add_random_obstacles(0.05)
    grid.add_rectangle((0, 0), (2, 2))
    grid.add_rectangle((28, 52), (30, 54))
    grid.add_circle((15, 40), 4)

    pygame.init()
    screen = pygame.display.set_mode((SCRN_WIDTH, SCRN_HEIGHT))

    screen.fill((255, 255, 255))
    grid.draw(
        screen,
        *fit_map_to_screen(
            screen.get_size(),
            *grid.shape,
        ),
    )
    pygame.display.flip()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False


def _test_grid_collections(args):
    grid = Grid(occupancy_mats[args.map_name])

    pygame.init()
    screen = pygame.display.set_mode((SCRN_WIDTH, SCRN_HEIGHT))

    screen.fill((255, 255, 255))
    grid.draw(
        screen,
        *fit_map_to_screen(
            screen.get_size(),
            *grid.shape,
        ),
    )
    pygame.display.flip()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False


test_func_dict: dict[str, Callable] = {
    "draw": _test_grid_draw,
    "collections": _test_grid_collections,
}
if __name__ == "__main__":
    from argparse import ArgumentParser
    from .map_collection import occupancy_mats

    parser = ArgumentParser()
    parser.add_argument(
        "test_func",
        type=str,
        choices=test_func_dict.keys(),
        help="Name of a test function",
    )
    parser.add_argument("-m", "--map_name", type=str, choices=occupancy_mats.keys())
    args = parser.parse_args()
    try:
        test_func_dict[args.test_func](args)
    except KeyboardInterrupt:
        pass
    finally:
        pygame.quit()
