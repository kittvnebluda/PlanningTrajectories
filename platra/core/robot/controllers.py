from math import pi

import numpy as np

from platra.utils import (
    fix_angle,
    rot_t,
)

from ..symbolic.mass_point import MassPointSymbolic, PolarImplicitCurve
from ..traj import TrajSample
from .ackermann import (
    AckermannConfig,
    AckermannState,
    AckermannStateExt,
)
from .configs import AckermannConfigForDynamicFeedback
from .mass_point import MassPointState
from .robot import MassPointConfig, RobotController


class StaticFeedbackByStateController(RobotController):
    def __init__(
        self,
        conf: AckermannConfig,
        lambda1: np.ndarray,
        lambda2: np.ndarray,
    ):
        self.conf = conf
        self.e_prev = np.array([0.0, 0.0], dtype=float)
        self.e_prev = np.array([0.0, 0.0], dtype=float)
        self.la_sum = lambda1 + lambda2
        self.la_mul = lambda1 @ lambda2

    def compute_control(
        self, state: AckermannState, target: TrajSample, dt: float
    ) -> np.ndarray:
        assert dt != 0

        z1d = target.pos
        z1d_ddot = target.acc

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


class DynamicFeedbackByStateController(RobotController):
    def __init__(
        self,
        conf: AckermannConfigForDynamicFeedback,
    ):
        self.conf = conf
        self.state = AckermannStateExt()
        self.control = np.zeros(2)
        self.v1 = 0
        self.a2 = 1 / 3
        self.a1 = 100
        self.a0 = 100

    def _update_extended_state(self, s: AckermannState, dt: float) -> None:
        self.state.x = s.x
        self.state.y = s.y
        self.state.nu = s.nu
        self.state.beta_s = s.beta_s
        self.state.eta = s.eta
        self.state.zeta = s.zeta

    def compute_control(
        self,
        state: AckermannState,
        target: TrajSample,
        dt: float,
    ) -> np.ndarray:
        z1d = target.pos
        z1d_dot = target.vel
        z1d_ddot = target.acc

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


class CoordinatedController(RobotController):
    def __init__(
        self,
        conf: MassPointConfig,
        mps: PolarImplicitCurve | MassPointSymbolic,
        s_dot_star,
        k_s,
        k_e1,
        k_e2,
        k_d1,
        k_d2,
        traj_id,
        fx_max,
        fy_max,
        m_max,
        reverse=False,
        delta_alpha=0.0,
    ) -> None:
        self.conf = conf
        self.mps = mps
        self.traj_id = traj_id
        self.fx_max = fx_max
        self.fy_max = fy_max
        self.m_max = m_max
        self.reverse = -1 if reverse else 1
        self.delta_alpha = delta_alpha
        self.s_dot_star = self.reverse * s_dot_star  # Желаемая касательная скорость
        self.m = self.conf.mass
        self.J = self.conf.inertia_moment

        if reverse:
            self.delta_alpha += pi

        # Коэффициенты регулятора
        self.k_s = k_s
        self.k_e1 = k_e1
        self.k_e2 = k_e2
        self.k_d1 = k_d1
        self.k_d2 = k_d2

        # Внутреннее состояние для отладки
        self.s = 0
        self.debug_info = {
            "s": 0.0,  # Касательная координата
            "e": 0.0,  # Ортогональное отклонение
            "phi": 0.0,  # Значение функции траектории
            "xsi": 0.0,  # Кривизна
            "xsi_dot": 0.0,
            "s_dot": 0.0,  # Касательная скорость
            "e_dot": 0.0,  # Производная отклонения
            "delta": 0.0,  # Угловое отклонение
            "alpha_star": 0.0,  # Угол базиса Френе
            "Fx": 0.0,  # Сила по x
            "Fy": 0.0,  # Сила по y
            "M": 0.0,  # Момент
        }

    def compute_control(
        self, state: MassPointState, target: TrajSample, dt: float
    ) -> np.ndarray:
        x = state.x
        y = state.y
        vx = state.vx
        vy = state.vy
        alpha = state.alpha
        omega = state.omega

        # Вычисляем текущие значения
        if isinstance(self.mps, PolarImplicitCurve):
            phi = self.mps.phi(x, y)
            alpha_star = fix_angle(self.mps.alpha(x, y))
            jacobi = rot_t(alpha_star)
            xi = self.mps.xi(x, y)
            xi_dot = self.mps.xi_dot(x, y)
        else:
            phi = self.mps.phi(self.traj_id, x, y)
            alpha_star = fix_angle(self.mps.alpha(self.traj_id, x, y))
            jacobi = rot_t(alpha_star)
            xi = self.mps.xi(self.traj_id, x, y)
            xi_dot = self.mps.xi_dot(self.traj_id, x, y)

        # Вычисляем текущее положение в координатах Фрине
        s_dot, e_dot = jacobi @ np.array([vx, vy])

        # Вычисляем угловое отклонение
        delta = fix_angle(alpha - alpha_star + self.delta_alpha)
        delta_dot = omega - xi * s_dot

        # Вычисляем виртуальные регуляторы
        u_s = -xi * s_dot * e_dot + self.k_s * (self.s_dot_star - s_dot)
        u_e = xi * s_dot**2 + self.k_e1 * e_dot - self.k_e2 * phi
        u_delta = (
            xi_dot * s_dot
            + (xi**2) * s_dot * e_dot
            - self.k_d1 * delta_dot
            - self.k_d2 * delta
        )

        F_vec = self.m * rot_t(self.delta_alpha) @ np.array([u_s, u_e])
        Fx, Fy = F_vec

        M = self.J * (xi * u_s + u_delta)

        F = np.array([Fx, Fy, M])
        F = np.clip(
            F,
            (-self.fx_max, -self.fy_max, -self.m_max),
            (self.fx_max, self.fy_max, self.m_max),
        )

        self.s += dt * s_dot

        # Обновляем внутреннее состояние для отладки
        self.debug_info.update(
            {
                "s": self.s,
                "e": phi,
                "phi": phi,
                "xsi": xi,
                "xsi_dot": xi_dot,
                "s_dot": s_dot,
                "e_dot": e_dot,
                "delta": delta,
                "alpha_star": alpha_star,
                "Fx": F[0],
                "Fy": F[1],
                "M": F[2],
            }
        )

        return F
