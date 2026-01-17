from argparse import ArgumentParser
from typing import Callable

import numpy as np
import pygame
from core.map import OCCUPANCY_MATS, CellState, Grid
from disp import ScreenParams
from disp import gridviz as gv


def _test_grid_draw(args):
    grid = Grid(np.ones((30, 54)))
    grid.add_random_obstacles(0.05)
    grid.add_rectangle((0, 0), (2, 2))
    grid.add_rectangle((28, 52), (30, 54))
    grid.add_circle((15, 40), 4)

    screen_params = ScreenParams(1024, 720)

    pygame.init()
    screen = pygame.display.set_mode((screen_params.width, screen_params.height))

    screen.fill((255, 255, 255))
    grid_params = gv.fit_map_to_screen(
        screen_params,
        *grid.shape,
    )
    print(grid.occupancy)
    gv.draw_grid(
        grid,
        screen,
        *grid_params,
        screen_params,
    )
    pygame.display.flip()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False


def _test_grid_collections(args):
    grid = Grid(OCCUPANCY_MATS[args.map_name])
    grid.occupancy[1, 1] = CellState.START.value
    grid.occupancy[9, 9] = CellState.GOAL.value

    screen_params = ScreenParams(1024, 720)

    pygame.init()
    screen = pygame.display.set_mode((screen_params.width, screen_params.height))

    screen.fill((255, 255, 255))
    gv.draw_grid(
        grid,
        screen,
        *gv.fit_map_to_screen(
            screen_params,
            *grid.shape,
        ),
        screen_params,
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
    parser = ArgumentParser()
    parser.add_argument(
        "test_func",
        type=str,
        choices=test_func_dict.keys(),
        help="Name of a test function",
    )
    parser.add_argument("-m", "--map_name", type=str, choices=OCCUPANCY_MATS.keys())
    args = parser.parse_args()
    try:
        test_func_dict[args.test_func](args)
    except KeyboardInterrupt:
        pass
    finally:
        pygame.quit()
