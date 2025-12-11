from math import cos, pi, sin
from typing import Any, Generator

import numpy as np
import pygame
from matplotlib import pyplot as plt
from numpy.typing import NDArray
from pygame import Surface, event

from platra.constants import PI_DOUBLE, PI_DOUBLE_NEG
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
from platra.core.traj.primitives import ArcPrimitive, StraightLineTimedPrimitive
from platra.core.traj.stitcher import stitch
from platra.disp.draw import DrawParams, draw_pts
from platra.disp.robot import draw_ackermann
from platra.disp.screen import ScreenParams


class Lab2TrajTracking:
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
        time (float): The duration of the straight-line movement in seconds.
        radius_2 (float): The radius of the second circular motion.
        clockwise (bool): Direction for the recond circular motion.
    """

    def __init__(
        self,
        initial_xsi: NDArray = np.array([6, 7, -2 * pi / 3]),
        radius_1: float = 6,
        delta: float = -1 * pi,
        alpha: float = pi / 3,
        time: float = 1,
        radius_2: float = 10,
        clockwise: bool = True,
    ) -> None:
        self._traj_ended = False
        self.path = []
        self.vs = []
        self.path_err = []

        L1 = np.array([[20, 0], [0, 10]])
        L2 = np.array([[2, 0], [0, 2]])

        # self.conf = AckermannConfigForStaticFeedback.from_symbolic(
        #     0.35, 1, 0.4, 0.2, LambdifiedAckermannForStaticFeedback
        # )
        # self.reg = StaticFeedbackByStateController(self.conf, L1, L2)
        self.conf = AckermannConfigForDynamicFeedback.from_symbolic(
            0.4, 1, 2, 0.2, LambdifiedAckermannForDynamicFeedback
        )
        self.reg = DynamicFeedbackByStateController(self.conf)

        initial_state = RobotState(initial_xsi.copy(), -self.conf.alpha3s, 0, 0)
        self.robot = create_robot_model(self.conf, initial_state)

        self.traj_draw_params = DrawParams(size=2, color=(0, 0, 255))
        self.path_draw_params = DrawParams(size=2, color=(0, 0, 0))
        self.robot_draw_params = DrawParams()
        self.target_draw_params = DrawParams(size=4, color=(255, 0, 0))

        self.screen_params = ScreenParams(1920, 1080, 30, shift=(0, 500))

        self.target = initial_xsi[:2]
        self.target_prev = initial_xsi[:2]
        self.target_diff = np.zeros(2)
        self.target_diff_prev = np.zeros(2)
        self.target_ddiff = np.zeros(2)
        self.target_ddiff_prev = np.zeros(2)
        self.traj = self._traj_generator(
            initial_xsi,
            radius_1,
            delta,
            alpha,
            time,
            radius_2,
            clockwise,
        )
        self._complete_traj = np.array(
            list(
                self._traj_generator(
                    initial_xsi,
                    radius_1,
                    delta,
                    alpha,
                    time,
                    radius_2,
                    clockwise,
                )
            )
        )

    def _traj_generator(
        self, initial_xsi, R1, delta, alpha, time, R2, clockwise, samples=1000
    ) -> Generator[NDArray, Any, None]:
        angle1 = alpha + initial_xsi[2] + delta
        angle2 = PI_DOUBLE_NEG if clockwise else PI_DOUBLE

        primitives = []
        primitives.append(
            ArcPrimitive(
                radius=R1,
                arc_len=delta,
                shift=initial_xsi[:2] + np.array([-R1, 0]),
                angle=0,
                samples=samples,
            )
        )
        primitives.append(
            StraightLineTimedPrimitive(
                end=(10 * cos(angle1), 10 * sin(angle1)),
                time=time,
                samples=int(samples * 0.5),
            )
        )
        primitives.append(
            ArcPrimitive(
                radius=R2,
                arc_len=angle2,
                shift=np.array(
                    [
                        -R2 * cos(alpha + delta - pi / 5),
                        -R2 * sin(alpha + delta - pi / 5),
                    ]
                ),
                angle=pi + pi / 10,
                samples=int(samples * 2),
            )
        )

        gen = stitch(primitives)
        return gen

    def handle_keyup(self, key: event.Event) -> None:
        pass

    def handle_keydown(self, key: event.Event) -> None:
        pass

    def handle_mouse(self, surface: Surface) -> None:
        pass

    def draw(self, surface: Surface, dt: float):
        if np.linalg.norm(self.target - self.conf.h(self.robot.state)) < 0.5:
            try:
                self.target = next(self.traj)
                self.path.append(self.conf.h(self.robot.state))
                self.path_err.append(
                    np.min(np.linalg.norm(self._complete_traj - self.path[-1], axis=1))
                )

            except StopIteration:
                if not self._traj_ended:
                    self._traj_ended = True
                    print("Trajectory ended")
                p = np.array(self.path)
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

        self.target_diff = (self.target - self.target_prev) / dt
        self.target_ddiff = (self.target_diff - self.target_diff_prev) / dt
        v = self.reg.update_control(
            self.robot.state,
            self.target,
            self.target_diff,
            self.target_ddiff,
            dt,
        )
        self.target_prev = self.target
        self.target_diff_prev = self.target_diff
        self.target_ddiff_prev = self.target_ddiff
        self.vs.append(v)
        self.robot.step(v[0], v[1], dt)

        # Drawing
        draw_pts(
            surface, self._complete_traj, self.traj_draw_params, self.screen_params
        )
        draw_pts(surface, self.path, self.path_draw_params, self.screen_params)
        draw_pts(surface, [self.target], self.target_draw_params, self.screen_params)
        draw_ackermann(
            surface, self.robot, self.robot_draw_params, self.screen_params, draw_h=True
        )


class Lab2Teleop(Lab2TrajTracking):
    def __init__(self) -> None:
        self.speed_mod = 0
        self.steering_mod = 0

        L1 = np.array([[2, 0], [0, 1]])
        L2 = np.array([[1, 0], [0, 1]])

        conf = AckermannConfigForStaticFeedback.from_symbolic(
            0.35, 1, 0.4, 0.2, LambdifiedAckermannForStaticFeedback
        )
        self.reg = StaticFeedbackByStateController(conf, L1, L2)
        initial_state = RobotState(
            np.array([0, 0, 0], dtype=float), -conf.alpha3s, 0, 0
        )
        self.robot = create_robot_model(conf, initial_state)

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
