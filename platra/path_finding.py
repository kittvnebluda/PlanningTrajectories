from queue import PriorityQueue

import numpy as np

from .map import Grid
from .types import Cell


def _heuristic(c1: Cell, c2: Cell):
    return ((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2) ** (1 / 2)


def _discover_map(map: Grid, start: Cell, goal: Cell) -> dict[Cell, Cell]:
    frontier = PriorityQueue()
    frontier.put((0, start))
    came_from = dict()
    cost_so_far = dict()
    came_from[start] = None
    cost_so_far[start] = 0

    while not frontier.empty():
        current = frontier.get()[1]

        if current == goal:
            return came_from

        neighbors = map.neighbors(current)
        for next in neighbors:
            new_cost = cost_so_far[current] + map.cost(current, next)
            if next not in cost_so_far or new_cost < cost_so_far[next]:
                frontier.put((new_cost + _heuristic(current, goal), next))
                came_from[next] = current
                cost_so_far[next] = new_cost

    return came_from


def find_path(map: Grid, start: Cell, goal: Cell) -> list[Cell]:
    came_from = _discover_map(map, start, goal)
    path = []
    current = goal
    try:
        while current != start:
            path.append(current)
            current = came_from[current]
    except KeyError:
        print("Can't reach goal")
        return []
    path.append(start)
    path.reverse()
    return path


if __name__ == "__main__":
    from itertools import cycle

    import pygame

    from .map import CellState, fit_map_to_screen
    from .trajectory import (
        Trajectory,
        TrajectoryC0,
        TrajectoryC1,
        TrajectoryC2,
        DrawOpts,
    )
    from .utils import SCRN_HEIGHT, SCRN_WIDTH, from_screen
    from .map_collection import occupancy_mats

    traj_cicle = cycle([TrajectoryC0, TrajectoryC1, TrajectoryC2])

    pygame.init()
    screen = pygame.display.set_mode((SCRN_WIDTH, SCRN_HEIGHT))

    map = Grid(occupancy_mats["lab1"])
    map_disp_params = fit_map_to_screen(
        screen.get_size(),
        *map.shape,
    )

    start = (1, 1)
    goal = (9, 9)
    map.occupancy[start[0], start[1]] = CellState.START.value
    map.occupancy[goal[0], goal[1]] = CellState.GOAL.value

    path = find_path(map, start, goal)
    waypoints = [map.to_world(c, *map_disp_params) for c in path]

    t: Trajectory = TrajectoryC2(waypoints)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_t:
                    t = next(traj_cicle)(waypoints)

        rmb_pressed, _, lmb_pressed = pygame.mouse.get_pressed()
        if rmb_pressed:
            x, y = pygame.mouse.get_pos()
            next_goal = map.to_grid(from_screen(x, y), *map_disp_params)

            if map.in_bounds(next_goal):
                map.occupancy[goal[0], goal[1]] = CellState.FREE.value
                goal = next_goal
                map.occupancy[goal[0], goal[1]] = CellState.GOAL.value

                path = find_path(map, start, goal)
                waypoints = [map.to_world(c, *map_disp_params) for c in path]
                t.waypoints = waypoints
        if lmb_pressed:
            x, y = pygame.mouse.get_pos()
            next_start = map.to_grid(from_screen(x, y), *map_disp_params)

            if map.in_bounds(next_start):
                map.occupancy[start[0], start[1]] = CellState.FREE.value
                start = next_start
                map.occupancy[start[0], start[1]] = CellState.START.value

                path = find_path(map, start, goal)
                waypoints = [map.to_world(c, *map_disp_params) for c in path]
                t.waypoints = waypoints

        screen.fill((255, 255, 255))

        map.draw(screen, *map_disp_params)
        t.draw_trajectory(screen, DrawOpts(draw_waypoints=False))

        pygame.display.flip()
