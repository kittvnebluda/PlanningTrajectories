from itertools import cycle

import numpy as np
import pygame
from matplotlib import pyplot as plt

from .astar import astar
from .draw import DrawParams, draw_line, draw_pts
from .map import (
    CellState,
    Grid,
    cell_array_to_world,
    grid_to_world,
    fit_screen_to_map,
    world_to_grid,
)
from .maps import occupancy_mats
from .trajectory import (
    TrajParams,
    compute_traj_curvature,
    interpolate_bsplines,
    interpolate_c0,
    interpolate_c1,
    interpolate_c2,
)
from .utils import from_screen

# Trajectory settings
traj_params = TrajParams(resolution=0.001, curvature_gain=1, bspline_degree=5)
traj_cycle = cycle(
    [interpolate_c0, interpolate_c1, interpolate_c2, interpolate_bsplines]
)

# Map settings
map = Grid(occupancy_mats["lab1"])
grid_scale, shitf_x, shift_y, screen_params = fit_screen_to_map(map.cols, map.rows, 0.6)
map_disp_params = (grid_scale, shitf_x, shift_y)

# Draw settings
draw_traj_params = DrawParams(size=int(17 * grid_scale), color=(255, 165, 0))
draw_pts_params = DrawParams(size=int(30 * grid_scale), color=(255, 165, 0))

# Initialization
pygame.init()
screen = pygame.display.set_mode((screen_params.width, screen_params.height))

start = (1, 1)
goal = (9, 9)
map.occupancy[start[0], start[1]] = CellState.START.value
map.occupancy[goal[0], goal[1]] = CellState.GOAL.value

path = astar(map, start, goal)
wps = cell_array_to_world(path, *map_disp_params)
interpolate_func = next(traj_cycle)
traj = interpolate_func(wps, traj_params)

# Main loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            match event.key:
                case pygame.K_t:
                    interpolate_func = next(traj_cycle)
                    traj = interpolate_func(wps, traj_params)
                case pygame.K_k:
                    curvature = compute_traj_curvature(traj)
                    t = np.arange(0, len(curvature))
                    f = plt.figure()
                    f.set_figwidth(15)
                    f.set_figheight(3)
                    plt.plot(t, curvature)
                    plt.grid()
                    plt.show()
                case pygame.K_l:
                    t1 = interpolate_c0(wps, traj_params)
                    t2 = interpolate_c1(wps, traj_params)
                    t3 = interpolate_c2(wps, traj_params)
                    t4 = interpolate_bsplines(wps, traj_params)
                    curvs = [compute_traj_curvature(t) for t in [t1, t2, t3, t4]]
                    lens = [len(c) for c in curvs]
                    max_len = max(lens)
                    xs = [np.linspace(0, max_len, lens[i]) for i in range(4)]
                    for i in range(4):
                        plt.plot(xs[i], curvs[i])
                    plt.grid()
                    plt.show()

    rmb_pressed, _, lmb_pressed = pygame.mouse.get_pressed()

    if rmb_pressed:
        x, y = pygame.mouse.get_pos()
        next_goal = world_to_grid(from_screen(x, y, screen_params), *map_disp_params)

        if map.in_bounds(next_goal):
            map.occupancy[goal[0], goal[1]] = CellState.FREE.value
            goal = next_goal
            map.occupancy[goal[0], goal[1]] = CellState.GOAL.value

            path = astar(map, start, goal)
            wps = cell_array_to_world(path, *map_disp_params)
            traj = interpolate_func(wps, traj_params)

    if lmb_pressed:
        x, y = pygame.mouse.get_pos()
        next_start = world_to_grid(from_screen(x, y, screen_params), *map_disp_params)

        if map.in_bounds(next_start):
            map.occupancy[start[0], start[1]] = CellState.FREE.value
            start = next_start
            map.occupancy[start[0], start[1]] = CellState.START.value

            path = astar(map, start, goal)
            wps = cell_array_to_world(path, *map_disp_params)
            traj = interpolate_func(wps, traj_params)

    screen.fill((255, 255, 255))

    map.draw(screen, *map_disp_params, screen_params=screen_params)
    draw_line(screen, traj, draw_traj_params, screen_params)
    # draw_pts(screen, wps, draw_traj_params, screen_params)
    draw_pts(
        screen,
        [grid_to_world(start, *map_disp_params), grid_to_world(goal, *map_disp_params)],
        draw_pts_params,
        screen_params,
    )

    pygame.display.flip()


""" Code Stash
        path_record.append(state.pose.pos.copy())
        target = t.get_target(state.pose.pos)
        e = target - state.pose.pos
        sp = min((e * 10).magnitude(), 1)
        phi = atan2(e.y, e.x)
        theta_err = 10 * fix_angle(phi - state.pose.theta)
        cmdvel(state, Twist(sp, 0, theta_err))
"""
