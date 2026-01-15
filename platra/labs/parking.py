from math import cos, pi, sin

import numpy as np
import pygame
from pygame import Surface, Vector2

from platra.core.astar.hybrid_astar import Pose, pose_to_grid
from platra.core.map.grid import (
    CellState,
    Grid,
)
from platra.core.traj import (
    InterpType,
    TrajParams,
)
from platra.disp import DrawParams, ScreenParams, draw, fit_screen_to_map, screen
from platra.disp import gridviz as gv
from platra.disp import robotviz as rv

from .labs import Laboratory


class Parking(Laboratory):
    def __init__(self):
        # ---- Map ----
        self.grid = Grid.from_name("parking_lot")
        self.grid_scale, self.shift_x, self.shift_y, self.screen_params = (
            fit_screen_to_map(self.grid.cols, self.grid.rows, 1.4, 20)
        )
        self.grid_disp_params = (self.grid_scale, self.shift_x, self.shift_y)

        # ---- Draw parameters ----
        self.traj_dp = DrawParams(size=int(2 * self.grid_scale), color=(255, 165, 0))
        self.pts_dp = DrawParams(size=int(5 * self.grid_scale), color=(255, 165, 0))
        self.vec_dp = DrawParams(size=int(2 * self.grid_scale), color=(255, 100, 0))

        # ---- Knowns points ----
        self.start_pose = Pose(10, 10, pi / 2)
        self.goal_pose = Pose(25, 20, pi / 2)
        self.grid.occupancy[pose_to_grid(self.start_pose, *self.grid_disp_params)] = (
            CellState.START.value
        )
        self.grid.occupancy[pose_to_grid(self.goal_pose, *self.grid_disp_params)] = (
            CellState.GOAL.value
        )

        # ---- Trajectory ----
        self.traj_params = TrajParams(
            resolution=0.01,
            curvature_gain=1,
            bspline_degree=5,
            interp_type=InterpType.BSpline,
        )
        self._generated = False

        # self.traj = hybrid_astar(
        #     self.grid,
        #     self.start_pose,
        #     self.goal_pose,
        #     0.2,
        #     1,
        #     pi,
        #     *self.grid_disp_params,
        # )

    def draw_pose(
        self,
        surface: Surface,
        p: Pose,
        draw_params: DrawParams,
        screen_params: ScreenParams,
    ):
        vec = np.array([cos(p.theta), sin(p.theta)])
        draw.vector(surface, np.array(p.pos()), vec, draw_params, screen_params)

    def draw(self, surface: pygame.Surface, dt: float):
        gv.draw_grid(
            self.grid, surface, *self.grid_disp_params, screen_params=self.screen_params
        )
        self.draw_pose(surface, self.start_pose, self.vec_dp, self.screen_params)
        self.draw_pose(surface, self.goal_pose, self.vec_dp, self.screen_params)
