import logging
from math import cos, pi, sin

import numpy as np
import pygame
from matplotlib import pyplot as plt
from numpy.typing import NDArray
from pygame import Surface, event

from platra.constants import PI_DOUBLE, PI_DOUBLE_NEG
from platra.core.robot.ackermann import AckermannModel, AckermannState
from platra.core.robot.configs import AckermannConfigForStaticFeedback
from platra.core.robot.controllers import StaticFeedbackByStateController
from platra.core.symbolic.ackermann import LambdifiedAckermannForStaticFeedback
from platra.core.traj import (
    ArcPrimitive,
    SequenceTrajectory,
    StraightLinePrimitive,
    Trajectory,
    TrajSample,
)
from platra.disp.draw import DrawParams, draw_points
from platra.disp.robotviz import draw_ackermann
from platra.disp.screen import ScreenParams

from .labs import Laboratory

VEC2_ZERO = np.zeros(2)

logger = logging.getLogger(__name__)


class TrajTracking(Laboratory):
    """
    Implements a specific trajectory for a mobile robot.

    The trajectory consists of three main segments:
    1.  **Circular Motion 1:** The robot moves along a circle of radius `radius_1`,
        with the tracking point's azimuth changing by `delta` radians in a
        specified direction (positive or negative).
    2.  **Straight Line Motion:** The robot then turns by `alpha` radians and
        proceeds in a straight line for `time` seconds.
    3.  **Circular Motion 2:** Finally, the robot performs another circular motion
        of radius R2 (implied by the overall description) in a specified direction
        (clockwise or counter-clockwise).

    All trajectory parameters are typically provided in an external 'Table 1'.

    Args:
        initial_xsi (NDArray): The initial position of the robot.
        radius_1 (float): The radius of the first circular motion.
        delta (float): The change in azimuth for the first circular motion in radians.
        alpha (float): The turn angle in radians before the straight-line movement.
        radius_2 (float): The radius of the second circular motion.
        clockwise (bool): Direction for the recond circular motion.
    """

    def __init__(
        self,
        initial_xsi: NDArray = np.array([6, 7, -2 * pi / 3]),
        radius_1: float = 6,
        delta: float = -1 * pi,
        alpha: float = pi / 3,
        radius_2: float = 10,
        clockwise: bool = True,
    ) -> None:
        self.path = []
        self.vs = []
        self.path_err = []

        L1 = np.array([[20, 0], [0, 10]])
        L2 = np.array([[2, 0], [0, 2]])

        self.conf = AckermannConfigForStaticFeedback.from_symbolic(
            0.35, 1, 0.4, 0.2, LambdifiedAckermannForStaticFeedback
        )
        self.reg = StaticFeedbackByStateController(self.conf, L1, L2)

        initial_state = AckermannState(initial_xsi.copy(), -self.conf.alpha3s, 0, 0)
        self.robot = AckermannModel(self.conf, initial_state)

        self.traj_draw_params = DrawParams(size=2, color=(0, 0, 255))
        self.path_draw_params = DrawParams(size=2, color=(0, 0, 0))
        self.robot_draw_params = DrawParams()
        self.target_draw_params = DrawParams(size=4, color=(255, 0, 0))

        self.screen_params = ScreenParams(1920, 1080, 30, shift=(0, 500))

        self.target = TrajSample(pos=initial_xsi[:2])
        self.traj = self._traj_generator(
            initial_xsi,
            radius_1,
            delta,
            alpha,
            radius_2,
            clockwise,
        )

    def _traj_generator(
        self, initial_xsi, R1, delta, alpha, R2, clockwise
    ) -> Trajectory:
        angle1 = alpha + initial_xsi[2] + delta
        angle2 = PI_DOUBLE_NEG if clockwise else PI_DOUBLE

        res = 0.1
        vel_norm = 0.5

        return SequenceTrajectory(
            [
                ArcPrimitive(
                    radius=R1,
                    arc_len=delta,
                    center_shift=initial_xsi[:2],
                    circle_rot=0,
                    resolution=res,
                    vel_norm=vel_norm,
                ),
                StraightLinePrimitive(
                    end=(10 * cos(angle1), 10 * sin(angle1)),
                    vel_norm=vel_norm,
                    resolution=res,
                ),
                ArcPrimitive(
                    radius=R2,
                    arc_len=angle2,
                    center_shift=VEC2_ZERO,
                    circle_rot=0,
                    resolution=res,
                    vel_norm=vel_norm,
                ),
            ]
        )

    def plot(self):
        p = np.array(self.path)
        if len(p) == 0:
            logger.info("Path is empty, nothing to show")
            exit()

        plt.axes().set_aspect("equal")
        plt.plot(p[:, 0], p[:, 1])
        plt.grid()
        plt.show()

        vs = np.array(self.vs)
        plt.plot(vs[:, 0])
        plt.plot(vs[:, 1], "--")
        plt.grid()
        plt.show()

        plt.plot(self.path_err)
        plt.grid()
        plt.show()

        exit()

    def draw(self, surface: Surface, dt: float):
        if np.linalg.norm(self.target.pos - self.conf.h(self.robot.state)) < 0.5:
            try:
                self.target = self.traj.sample()
                self.path.append(self.conf.h(self.robot.state))
                self.path_err.append(
                    np.min(
                        np.linalg.norm(self.traj.samples_pos - self.path[-1], axis=1)
                    )
                )
            except StopIteration:
                self.plot()

        v = self.reg.compute_control(self.robot.state, self.target, dt)
        self.vs.append(v)
        self.robot.step(v[0], v[1], dt)

        draw_points(
            surface, self.traj.samples_pos, self.traj_draw_params, self.screen_params
        )
        draw_points(surface, self.path, self.path_draw_params, self.screen_params)
        draw_points(
            surface, [self.target.pos], self.target_draw_params, self.screen_params
        )
        draw_ackermann(
            surface, self.robot, self.robot_draw_params, self.screen_params, draw_h=True
        )


class Teleop(TrajTracking):
    def __init__(self) -> None:
        self.speed_mod = 0
        self.steering_mod = 0

        L1 = np.array([[2, 0], [0, 1]])
        L2 = np.array([[1, 0], [0, 1]])

        conf = AckermannConfigForStaticFeedback.from_symbolic(
            0.35, 1, 0.4, 0.2, LambdifiedAckermannForStaticFeedback
        )
        self.reg = StaticFeedbackByStateController(conf, L1, L2)
        initial_state = AckermannState(
            np.array([0, 0, 0], dtype=float), -conf.alpha3s, 0, 0
        )
        self.robot = AckermannModel(conf, initial_state)

        self.robot_draw_params = DrawParams()
        self.target_draw_params = DrawParams(size=10, color=(255, 0, 0))
        self.screen_params = ScreenParams(1920, 1080)

    def handle_keydown(self, key: event.Event) -> None:
        match key:
            case pygame.K_w:
                self.speed_mod = 0.1
            case pygame.K_s:
                self.speed_mod = -0.1
            case pygame.K_d:
                self.steering_mod = 0.1
            case pygame.K_a:
                self.steering_mod = -0.1
            case pygame.K_SPACE:
                self.robot.state.eta = 0
                self.robot.state.zeta = 0

    def handle_keyup(self, key: event.Event) -> None:
        match key:
            case pygame.K_w | pygame.K_s:
                self.speed_mod = 0
            case pygame.K_d | pygame.K_a:
                self.steering_mod = 0

    def draw(self, surface: Surface, dt: float):
        self.robot.step(-self.speed_mod, -self.steering_mod, dt)
        draw_ackermann(
            surface, self.robot, self.robot_draw_params, self.screen_params, draw_h=True
        )
