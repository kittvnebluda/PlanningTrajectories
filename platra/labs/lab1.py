from math import cos, sin

import numpy as np
import pygame
from core.astar import astar
from core.map.grid import (
    CellState,
    Grid,
    cell_array_to_world,
    grid_to_world,
    world_to_grid,
)
from core.robot.ackermann import AckermannState
from core.robot.configs import AckermannConfigForStaticFeedback
from core.robot.controllers import StaticFeedbackByStateController
from core.symbolic.ackermann import LambdifiedAckermannForStaticFeedback
from core.traj import (
    InterpType,
    TrajParams,
    WaypointsTrajectory,
    traj_curvature,
    traj_length,
)
from disp import (
    DrawParams,
    draw,
    fit_screen_to_map,
    screen,
)
from disp import gridviz as gv
from disp import robotviz as rv
from matplotlib import pyplot as plt

from .labs import Laboratory


class TrajPlanning(Laboratory):
    def __init__(self):
        # ---- Map ----
        self.map = Grid.from_name("lab1")
        self.grid_scale, self.shift_x, self.shift_y, self.screen_params = (
            fit_screen_to_map(self.map.cols, self.map.rows, 1)
        )
        self.map_disp_params = (self.grid_scale, self.shift_x, self.shift_y)

        # ---- Draw parameters ----
        self.traj_dp = DrawParams(size=int(17 * self.grid_scale), color=(255, 165, 0))
        self.pts_dp = DrawParams(size=int(30 * self.grid_scale), color=(255, 165, 0))

        # ---- Knowns points ----
        self.start = (1, 1)
        self.goal = (9, 9)
        self.map.occupancy[self.start] = CellState.START.value
        self.map.occupancy[self.goal] = CellState.GOAL.value

        # ---- Trajectory ----
        self.traj_params = TrajParams(
            resolution=0.01,
            curvature_gain=1,
            bspline_degree=5,
            interp_type=InterpType.BSpline,
        )
        self.path = astar(self.map, self.start, self.goal)
        self.wps = cell_array_to_world(self.path, *self.map_disp_params)
        self.traj = WaypointsTrajectory(self.wps, self.traj_params)

        # ---- Robot ----
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
        self.initial_state = AckermannState(
            np.array([start[0], start[1], 0]), -self.conf.alpha3s, 0, 0
        )
        self.robot = create_robot_model(self.conf, self.initial_state)
        self.robot_dp = DrawParams()
        self.target = self.traj.sample()
        self._traj_ended = False

    def handle_keydown(self, key: pygame.event.Event):
        match key:
            case pygame.K_l:
                print(f"Trajectory length: {traj_length(self.traj):.3f} m")
            case pygame.K_n:
                self.traj.next_interp_type()
                self.robot = create_robot_model(self.conf, self.initial_state)
                self.i = 0
            case pygame.K_c:
                curvature = traj_curvature(self.traj)
                plt.figure(figsize=(15, 3))
                plt.plot(curvature)
                plt.grid()
                plt.show()
            case pygame.K_a:
                curvs = []
                for _ in range(len(InterpType)):
                    curvs.append(traj_curvature(self.traj))
                    self.traj.next_interp_type()
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
            screen.from_px(x, y, self.screen_params), *self.map_disp_params
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
        self.traj.set_waypoints(self.wps)

    def update_target(self) -> None:
        try:
            if np.linalg.norm(self.target.pos - self.conf.h(self.robot.state)) < 0.1:
                self.target = self.traj.sample()
                shift = self.conf.Lf * np.array(
                    [cos(self.robot.nu), sin(self.robot.nu)]
                )
                self.target.pos = self.target.pos + shift
        except StopIteration:
            if not self._traj_ended:
                self._traj_ended = True
                print("Trajectory ended")

    def draw(self, surface: pygame.Surface, dt: float):
        self.update_target()
        v = self.reg.compute_control(self.robot.state, self.target, dt)
        self.robot.step(v[0], v[1], dt)

        gv.draw_grid(
            self.map, surface, *self.map_disp_params, screen_params=self.screen_params
        )
        draw.polyline(surface, self.traj.samples_pos, self.traj_dp, self.screen_params)
        draw.points(
            surface,
            [
                grid_to_world(self.start, *self.map_disp_params),
                grid_to_world(self.goal, *self.map_disp_params),
            ],
            self.pts_dp,
            self.screen_params,
        )
        rv.ackermann(
            surface,
            self.robot,
            self.robot_dp,
            self.screen_params,
        )
