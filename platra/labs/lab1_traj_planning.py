from itertools import cycle
from math import cos, sin

import numpy as np
import pygame
from matplotlib import pyplot as plt

from platra.core.astar import astar
from platra.core.map.grid import (
    CellState,
    Grid,
    cell_array_to_world,
    grid_to_world,
    world_to_grid,
)
from platra.core.robot import AckermannConfigForStaticFeedback, create_robot_model
from platra.core.robot.base import RobotState
from platra.core.robot.configs import AckermannConfigForDynamicFeedback
from platra.core.robot.controllers import (
    DynamicFeedbackByStateController,
    StaticFeedbackByStateController,
)
from platra.core.symbolic.ackermann import (
    LambdifiedAckermannForDynamicFeedback,
    LambdifiedAckermannForStaticFeedback,
)
from platra.core.traj.offline import (
    TrajParams,
    compute_traj_curvature,
    compute_trajectory_length,
    interpolate_bsplines,
    interpolate_c0,
    interpolate_c1,
    interpolate_c2,
)
from platra.disp.draw import DrawParams, draw_pts, draw_pts_connected
from platra.disp.map import draw_grid, fit_screen_to_map
from platra.disp.robot import draw_ackermann
from platra.disp.screen import from_screen


class Lab1TrajPlanning:
    def __init__(self):
        self.traj_params = TrajParams(
            resolution=0.01, curvature_gain=1, bspline_degree=5
        )
        self.traj_cycle = cycle(
            [interpolate_c0, interpolate_c1, interpolate_c2, interpolate_bsplines]
        )

        self.map = Grid.from_name("lab1")
        self.grid_scale, self.shift_x, self.shift_y, self.screen_params = (
            fit_screen_to_map(self.map.cols, self.map.rows, 1)
        )
        self.map_disp_params = (self.grid_scale, self.shift_x, self.shift_y)

        self.draw_traj_params = DrawParams(
            size=int(17 * self.grid_scale), color=(255, 165, 0)
        )
        self.draw_pts_params = DrawParams(
            size=int(30 * self.grid_scale), color=(255, 165, 0)
        )

        self.start = (1, 1)
        self.goal = (9, 9)
        self.map.occupancy[self.start] = CellState.START.value
        self.map.occupancy[self.goal] = CellState.GOAL.value

        self.path = astar(self.map, self.start, self.goal)
        self.wps = cell_array_to_world(self.path, *self.map_disp_params)
        self.interpolate_func = next(self.traj_cycle)
        self.traj = self.interpolate_func(self.wps, self.traj_params)

        L1 = np.array([[20, 0], [0, 10]])
        L2 = np.array([[2, 0], [0, 2]])

        self.conf = AckermannConfigForStaticFeedback.from_symbolic(
            Lf=0.15,
            Ls=0.5,
            e=0.2,
            r=0.1,
            symbolic_model=LambdifiedAckermannForStaticFeedback,
        )
        start = cell_array_to_world([self.start], *self.map_disp_params)[0]
        self.reg = StaticFeedbackByStateController(self.conf, L1, L2)
        self.initial_state = RobotState(
            np.array([start[0], start[1], 0]), -self.conf.alpha3s, 0, 0
        )
        self.robot = create_robot_model(self.conf, self.initial_state)
        self.i = 0
        self.robot_draw_params = DrawParams()
        self.target = self.traj[self.i]
        self._traj_ended = False

    def handle_keyup(self, key: pygame.event.Event):
        pass

    def handle_keydown(self, key: pygame.event.Event):
        match key:
            case pygame.K_l:
                print(
                    f"Trajectory length: {compute_trajectory_length(self.traj):.3f} m"
                )
            case pygame.K_n:
                self.interpolate_func = next(self.traj_cycle)
                self.traj = self.interpolate_func(self.wps, self.traj_params)
                self.robot = create_robot_model(self.conf, self.initial_state)
                self.i = 0
            case pygame.K_c:
                curvature = compute_traj_curvature(self.traj)
                plt.figure(figsize=(15, 3))
                plt.plot(curvature)
                plt.grid()
                plt.show()
            case pygame.K_a:
                interpolators = [
                    interpolate_c0,
                    interpolate_c1,
                    interpolate_c2,
                    interpolate_bsplines,
                ]
                curvs = [
                    compute_traj_curvature(fn(self.wps, self.traj_params))
                    for fn in interpolators
                ]
                xs = [np.linspace(0, len(c), len(c)) for c in curvs]
                plt.figure(figsize=(15, 3))
                for x, c in zip(xs, curvs):
                    plt.plot(x, c)
                plt.grid()
                plt.show()

    def handle_mouse(self, surface: pygame.Surface):
        rmb_pressed, _, lmb_pressed = pygame.mouse.get_pressed()
        x, y = pygame.mouse.get_pos()

        if not rmb_pressed and not lmb_pressed:
            return

        new_cell = world_to_grid(
            from_screen(x, y, self.screen_params), *self.map_disp_params
        )

        if not self.map.in_bounds(new_cell):
            return

        if rmb_pressed:
            self.map.occupancy[self.goal] = CellState.FREE.value
            self.goal = new_cell
            self.map.occupancy[self.goal] = CellState.GOAL.value
        elif lmb_pressed:
            self.map.occupancy[self.start] = CellState.FREE.value
            self.start = new_cell
            self.map.occupancy[self.start] = CellState.START.value
        self.path = astar(self.map, self.start, self.goal)
        self.wps = cell_array_to_world(self.path, *self.map_disp_params)
        self.traj = self.interpolate_func(self.wps, self.traj_params)

    def draw(self, surface: pygame.Surface, dt: float):
        if np.linalg.norm(self.target - self.conf.h(self.robot.state)) < 0.1:
            if self.i < len(self.traj):
                shift = self.conf.Lf * np.array(
                    [
                        cos(self.robot.nu),
                        sin(self.robot.nu),
                    ]
                )
                self.target = self.traj[self.i] + shift
                self.i += 1
            else:
                if not self._traj_ended:
                    self._traj_ended = True
                    print("Trajectory ended")

        v = self.reg.update_control(self.robot.state, np.array([0]), self.target, dt)
        self.robot.step(v[0], v[1], dt)

        draw_grid(
            self.map, surface, *self.map_disp_params, screen_params=self.screen_params
        )
        draw_pts_connected(
            surface, self.traj, self.draw_traj_params, self.screen_params
        )
        draw_pts(
            surface,
            [
                grid_to_world(self.start, *self.map_disp_params),
                grid_to_world(self.goal, *self.map_disp_params),
            ],
            self.draw_pts_params,
            self.screen_params,
        )
        draw_ackermann(
            surface,
            self.robot,
            self.robot_draw_params,
            self.screen_params,
        )
