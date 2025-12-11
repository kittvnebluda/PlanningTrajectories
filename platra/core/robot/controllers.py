from math import cos, sin

import numpy as np

from platra.core.robot.base import RobotConfig, RobotState, RobotStateExt
from platra.core.robot.configs import AckermannConfigForDynamicFeedback


class StaticFeedbackByStateController:
    def __init__(
        self,
        conf: RobotConfig,
        lambda1: np.ndarray,
        lambda2: np.ndarray,
    ):
        self.conf = conf
        self.e_prev = np.array([0.0, 0.0], dtype=float)
        self.e_prev = np.array([0.0, 0.0], dtype=float)
        self.la_sum = lambda1 + lambda2
        self.la_mul = lambda1 @ lambda2

    def update_control(
        self, state: RobotState, z1d: np.ndarray, z1d_ddot: np.ndarray, dt: float
    ) -> np.ndarray:
        assert dt != 0

        h = self.conf.h(state)
        e = h - z1d
        e_dot = (e - self.e_prev) / dt
        self.e_prev = e

        K_inv = self.conf.k_inv(state)

        w = z1d_ddot - self.la_sum @ e_dot - self.la_mul @ e
        g = self.conf.g(state)

        v = K_inv @ (w - g)
        v = np.clip(v, [-6, -5], [6, 5])

        return v


class DynamicFeedbackByStateController:
    def __init__(
        self,
        conf: AckermannConfigForDynamicFeedback,
    ):
        self.conf = conf
        self.state = RobotStateExt()
        self.control = np.zeros(2)
        self.v1 = 0
        self.a2 = 1 / 3
        self.a1 = 3
        self.a0 = 1

    def _update_extended_state(self, s: RobotState, dt: float) -> None:
        self.state.x = s.x
        self.state.y = s.y
        self.state.nu = s.nu
        self.state.beta_s = s.beta_s
        self.state.eta = s.eta
        self.state.zeta = s.zeta

    def update_control(
        self,
        state: RobotState,
        z1d: np.ndarray,
        z1d_dot: np.ndarray,
        z1d_ddot: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        print(
            f"Eta: {state.eta:5.5f},\t V1: {self.v1:5.5f},\t U1: {self.control[0]:5.5f}, \t V2: {self.control[1]:5.5f}"
        )
        print(
            f"z1d: {z1d[0], z1d[1]}, z1d_dot: {z1d_dot[0], z1d_dot[1]}, z1d_ddot: {z1d_ddot[0], z1d_ddot[1]}"
        )
        if dt <= 1e-6:
            return np.zeros(2)

        self._update_extended_state(state, dt)

        if abs(state.eta) < 1e-3:
            return np.array([0.1, 0.0])

        h = self.conf.h(state)
        e = h - z1d
        e_dot = self.conf.h_dot(self.state) - z1d_dot
        e_ddot = self.conf.h_ddot(self.state) - z1d_ddot

        z1d_dddot = np.zeros(2)
        w = z1d_dddot - (self.a2 * e_ddot + self.a1 * e_dot + self.a0 * e)

        self.control = self.conf.a_inv(self.state) @ (w - self.conf.b(self.state))
        # self.control = np.clip(self.control, [-4, -4], [4, 4])

        self.v1 = self.control[0] * dt
        self.state.v1 = self.v1

        return np.array([self.v1, self.control[1]])
