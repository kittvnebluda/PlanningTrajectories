from math import pi
from time import perf_counter
from typing import Callable

import numpy as np
import pygame
from core.robot.controllers import CoordinatedController, CoordinatedController3D
from core.robot.mass_point import (
    MassPointModel,
    MassPointState,
)
from core.robot.mass_point_3d import (
    MassPointModel3D,
    MassPointState3D,
)
from core.robot.robot import MassPointConfig
from core.symbolic.mass_point import (
    CURVES_LAB3,
    MassPointSymbolic,
    PolarImplicitCurve,
)
from core.traj import TrajParams, TrajSample, WaypointsTrajectory
from disp import DrawParams, ScreenParams, draw
from disp import robotviz as rv
from matplotlib import pyplot as plt
from numpy.typing import NDArray
from pygame import Surface
from sympy import symbols
from utils import fix_angle

from .labs import Laboratory

x = np.linspace(-10, 10, 500)
y = np.linspace(-10, 10, 500)
X, Y = np.meshgrid(x, y)


def implicit_curve_points(
    phi_func: Callable,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    n: int,
) -> list[NDArray]:
    x = np.linspace(*x_range, n)
    y = np.linspace(*y_range, n)
    X, Y = np.meshgrid(x, y)
    Z = phi_func(X, Y)

    fig, ax = plt.subplots()
    cs = ax.contour(X, Y, Z, levels=[0])
    plt.close(fig)

    curves = []
    for seg in cs.allsegs[0]:
        curves.append(seg)

    return curves


class TrajStabilization2D(Laboratory):
    def __init__(self) -> None:
        self.screen_params = ScreenParams(1920, 1080, 50, (-200, 200))

        self.v_dp = DrawParams(color=(255, 0, 150), size=4)
        self.path_dp = DrawParams(color=(0, 0, 0), size=4)
        self.traj_dp = DrawParams(color=(0, 0, 255), size=4)
        self.robot_dp = DrawParams(color=(255, 0, 0), size=1)

        self.mps = MassPointSymbolic(CURVES_LAB3)
        phi_curves = []
        for phi in self.mps.phi_funcs:
            curves = implicit_curve_points(phi, (-100, 100), (-100, 100), 1000)
            phi_curves.append(curves[0])

        self.trajs = [
            WaypointsTrajectory(phi_curve, TrajParams()) for phi_curve in phi_curves
        ]
        self.path = []
        self.deltas = []
        self.es = []
        self.fxs = []
        self.fys = []
        self.ms = []

        self.conf = MassPointConfig(mass=4.1, inertia_moment=2)
        self.initial_state = MassPointState(x=0, y=-2, alpha=3 * pi / 2)
        self.robot = MassPointModel(self.conf, self.initial_state)

        s_dot_star = 3

        self.ctrl0 = CoordinatedController(
            conf=self.conf,
            mps=self.mps,
            s_dot_star=s_dot_star,
            k_s=10,
            k_e1=50,
            k_e2=170,
            k_d1=40,
            k_d2=110,
            traj_id=0,
            fx_max=50,
            fy_max=600,
            m_max=600,
            reverse=True,
        )
        self.ctrl1 = CoordinatedController(
            conf=self.conf,
            mps=self.mps,
            s_dot_star=s_dot_star,
            k_s=10,
            k_e1=60,
            k_e2=170,
            k_d1=40,
            k_d2=110,
            traj_id=1,
            fx_max=50,
            fy_max=600,
            m_max=600,
        )
        self.ctrl2 = CoordinatedController(
            conf=self.conf,
            mps=self.mps,
            s_dot_star=s_dot_star,
            k_s=10,
            k_e1=60,
            k_e2=170,
            k_d1=40,
            k_d2=110,
            traj_id=2,
            fx_max=50,
            fy_max=600,
            m_max=600,
        )
        self.regs = [self.ctrl0, self.ctrl1, self.ctrl2]

        self.switch_pts = np.array(
            [[7.09482, -2.24976], [10.18416, 8.99717], [-7.52007586, 4.49763808]]
        )
        self.traj_id = 0

    @property
    def x(self):
        return self.robot.x

    @property
    def y(self):
        return self.robot.y

    @property
    def alpha(self):
        return self.robot.alpha

    def draw_debug_info(self, surface: Surface, debug_info) -> None:
        font = pygame.font.SysFont("Arial", 16)
        debug_texts = [
            f"x = {self.robot.state.x:.2f}",
            f"y = {self.robot.state.y:.2f}",
            f"alpha = {self.robot.state.alpha:.2f}",
            f"s = {debug_info['s']:.2f} (длина дуги)",
            f"e = {debug_info['e']:.2f} (отклонение)",
            f"s_dot = {debug_info['s_dot']:.2f} (касат. скорость)",
            f"e_dot = {debug_info['e_dot']:.2f} (производная отклонения)",
            f"xsi = {debug_info['xsi']:.4f} (кривизна)",
            f"xsi_dot = {debug_info['xsi_dot']:.4f}",
            f"delta = {debug_info['delta']:.2f} (угловое отклонение)",
            f"Fx = {debug_info['Fx']:.2f}",
            f"Fy = {debug_info['Fy']:.2f}",
            f"M = {debug_info['M']:.2f}",
        ]

        draw.frame(
            surface,
            self.screen_params,
            self.x,
            self.y,
            debug_info["alpha_star"],
            ("purple", "yellow"),
        )

        for i, text in enumerate(debug_texts):
            surface.blit(font.render(text, True, (0, 0, 0)), (10, 10 + i * 20))

    def plot_path(self):
        path = np.array(self.path)
        plt.figure()
        plt.axes().set_aspect("equal")
        plt.plot(path[:, 0], path[:, 1])
        plt.xlabel("x, m")
        plt.ylabel("y, m")
        plt.grid()
        plt.show()

    def plot_errors(self):
        n, m = 2, 1
        deltas = np.array(self.deltas)
        plt.figure()
        plt.subplot(n, m, 1)
        plt.plot(deltas)
        plt.xlabel("index")
        plt.ylabel("delta")
        plt.grid()

        plt.subplot(n, m, 2)
        plt.plot(self.es)
        plt.xlabel("index")
        plt.ylabel("e")
        plt.grid()

        plt.show()

    def plot_control(self):
        n, m = 3, 1
        plt.figure()

        plt.subplot(n, m, 1)
        plt.plot(self.fxs)
        plt.ylabel("Fx")
        plt.grid()

        plt.subplot(n, m, 2)
        plt.plot(self.fys)
        plt.ylabel("Fy")
        plt.grid()

        plt.subplot(n, m, 3)
        plt.plot(self.ms)
        plt.xlabel("index")
        plt.ylabel("M")
        plt.grid()

        plt.show()

    def draw(self, surface: Surface, dt: float) -> None:
        if np.linalg.norm(self.robot.pos - self.switch_pts[self.traj_id]) < 0.1:
            if self.traj_id == 2:
                self.plot_path()
                self.plot_errors()
                self.plot_control()
                exit()
            else:
                self.traj_id += 1
        tid = self.traj_id

        u = self.regs[tid].compute_control(self.robot.state, TrajSample(), dt)
        self.robot.step(u, dt)

        p = np.array((self.robot.x, self.robot.y))
        v = np.array([self.robot.vx, self.robot.vy])
        self.path.append(p)
        self.deltas.append(self.regs[tid].debug_info["delta"])
        self.es.append(self.regs[tid].debug_info["e"])
        self.fxs.append(self.regs[tid].debug_info["Fx"])
        self.fys.append(self.regs[tid].debug_info["Fy"])
        self.ms.append(self.regs[tid].debug_info["M"])

        draw.polyline(
            surface, self.trajs[tid].samples_pos, self.traj_dp, self.screen_params
        )
        draw.polyline(surface, self.path, self.path_dp, self.screen_params)
        rv.cursor(surface, self.robot.pose, self.robot_dp, self.screen_params)
        draw.vector(surface, p, v, self.v_dp, self.screen_params)

        self.draw_debug_info(surface, self.regs[tid].debug_info)


class TrajStabilization2DEuclidianSpiral(Laboratory):
    def __init__(self) -> None:
        self.screen_params = ScreenParams(1920, 1080, 30)

        self.v_dp = DrawParams(color=(255, 0, 150), size=4)
        self.path_dp = DrawParams(color=(0, 0, 0), size=4)
        self.traj_dp = DrawParams(color=(0, 0, 255), size=4)
        self.robot_dp = DrawParams(color=(255, 0, 0), size=1)

        self.path = []
        self.deltas = []
        self.es = []
        self.fxs = []
        self.fys = []
        self.ms = []

        r, theta = symbols("r theta", real=True)
        self.mpses = [PolarImplicitCurve(-r + theta + 2 * np.pi * k) for k in range(10)]

        n = 1000
        r = np.linspace(0, 50, n)
        theta = np.linspace(0, 50 * np.pi, n)
        R, T = np.meshgrid(r, theta)
        X, Y = R * np.cos(T), R * np.sin(T)
        Z = -R + T
        fig, ax = plt.subplots()
        cs = ax.contour(X, Y, Z, levels=[0])
        plt.close(fig)

        phi_curve = []
        for seg in cs.allsegs[0]:
            phi_curve.append(seg)
        phi_curve = np.array(phi_curve)[0]

        self.traj = WaypointsTrajectory(phi_curve, TrajParams())

        self.conf = MassPointConfig(mass=4.1, inertia_moment=2)
        self.initial_state = MassPointState(x=0.1, y=0.1, alpha=0)
        self.robot = MassPointModel(self.conf, self.initial_state)

        s_dot_star = 3

        self.reg = CoordinatedController(
            conf=self.conf,
            mps=self.mpses[0],
            s_dot_star=s_dot_star,
            k_s=30,
            k_e1=30,
            k_e2=30,
            k_d1=30,
            k_d2=30,
            traj_id=0,
            fx_max=300,
            fy_max=300,
            m_max=600,
            reverse=True,
        )

    def plot_path(self):
        path = np.array(self.path)
        plt.figure()
        plt.axes().set_aspect("equal")
        plt.plot(path[:, 0], path[:, 1])
        plt.xlabel("x, m")
        plt.ylabel("y, m")
        plt.grid()
        plt.show()

    def plot_errors(self):
        n, m = 2, 1
        deltas = np.array(self.deltas)
        plt.figure()
        plt.subplot(n, m, 1)
        plt.plot(deltas)
        plt.xlabel("index")
        plt.ylabel("delta")
        plt.grid()

        plt.subplot(n, m, 2)
        plt.plot(self.es)
        plt.xlabel("index")
        plt.ylabel("e")
        plt.grid()

        plt.show()

    def plot_control(self):
        n, m = 3, 1
        plt.figure()

        plt.subplot(n, m, 1)
        plt.plot(self.fxs)
        plt.ylabel("Fx")
        plt.grid()

        plt.subplot(n, m, 2)
        plt.plot(self.fys)
        plt.ylabel("Fy")
        plt.grid()

        plt.subplot(n, m, 3)
        plt.plot(self.ms)
        plt.xlabel("index")
        plt.ylabel("M")
        plt.grid()

        plt.show()

    def on_close(self) -> None:
        self.plot_path()
        self.plot_errors()
        self.plot_control()

    def select_best_spiral(self, x, y):
        vals = [abs(mps.phi(x, y)) for mps in self.mpses]
        return int(np.argmin(vals))

    def draw_debug_info(self, surface: Surface, debug_info) -> None:
        font = pygame.font.SysFont("Arial", 16)
        debug_texts = [
            f"x = {self.robot.state.x:.2f}",
            f"y = {self.robot.state.y:.2f}",
            f"alpha = {self.robot.state.alpha:.2f}",
            f"s = {debug_info['s']:.2f} (длина дуги)",
            f"e = {debug_info['e']:.2f} (отклонение)",
            f"s_dot = {debug_info['s_dot']:.2f} (касат. скорость)",
            f"e_dot = {debug_info['e_dot']:.2f} (производная отклонения)",
            f"xsi = {debug_info['xsi']:.4f} (кривизна)",
            f"xsi_dot = {debug_info['xsi_dot']:.4f}",
            f"delta = {debug_info['delta']:.2f} (угловое отклонение)",
            f"Fx = {debug_info['Fx']:.2f}",
            f"Fy = {debug_info['Fy']:.2f}",
            f"M = {debug_info['M']:.2f}",
        ]

        draw.frame(
            surface,
            self.screen_params,
            self.robot.x,
            self.robot.y,
            debug_info["alpha_star"],
            ("purple", "yellow"),
        )

        for i, text in enumerate(debug_texts):
            surface.blit(font.render(text, True, (0, 0, 0)), (10, 10 + i * 20))

    def draw(self, surface: Surface, dt: float) -> None:
        mps_id = self.select_best_spiral(self.robot.x, self.robot.y)

        self.reg.mps = self.mpses[mps_id]
        self.reg.delta_alpha = (
            self.mpses[mps_id].alpha(self.robot.x, self.robot.y) - pi / 4
        )
        u = self.reg.compute_control(self.robot.state, TrajSample(), dt)
        self.robot.step(u, dt)

        p = np.array((self.robot.x, self.robot.y))
        v = np.array([self.robot.vx, self.robot.vy])

        self.path.append(p)
        self.deltas.append(self.reg.debug_info["delta"])
        self.es.append(fix_angle(self.reg.debug_info["e"]))
        self.fxs.append(self.reg.debug_info["Fx"])
        self.fys.append(self.reg.debug_info["Fy"])
        self.ms.append(self.reg.debug_info["M"])

        draw.polyline(surface, self.traj.samples_pos, self.traj_dp, self.screen_params)
        draw.polyline(surface, self.path, self.path_dp, self.screen_params)
        rv.cursor(surface, self.robot.pose, self.robot_dp, self.screen_params)

        # self.draw_debug_info(surface, self.reg.debug_info)


class TrajStabilization3D(Laboratory):
    def __init__(self) -> None:
        self.screen_params = ScreenParams(1920, 1080, 50, (-200, 200))

        self.path = []
        self.debug = []

        self.initial_pos = np.array([0, -2, 10])

        self.conf = MassPointConfig(mass=4.1, inertia_moment=2)
        self.initial_state = MassPointState3D(pos=self.initial_pos)
        self.robot = MassPointModel3D(self.conf, self.initial_state)

        self.reg = CoordinatedController3D(
            conf=self.conf,
            s_dot_star=2.1,
            k_s=20,
            k_1e1=200,
            k_1e2=200,
            k_2e1=200,
            k_2e2=200,
            k_r=10,
            k_w=10,
            fx_max=2000,
            fy_max=2000,
            fz_max=2000,
            mc_max=1000,
        )

        self.start_time = perf_counter()

    def draw_debug_info(self, surface: Surface, debug_info) -> None:
        font = pygame.font.SysFont("Arial", 16)
        debug_texts = [
            f"Fc = {debug_info['Fc'].round(3)}",
            f"Mc = {debug_info['Mc'].round(3)}",
            f"e_w = {debug_info['e_w'].round(3)}",
            f"a_d = {debug_info['a_d'].round(3)}",
            f"omega = {debug_info['omega'].round(2)}",
            f"omega_star= {debug_info['omega_star'].round(2)}",
            f"omega_dot_star= {debug_info['omega_dot_star'].round(2)}",
            f"alpha = {debug_info['alpha'].round(2)}",
            f"alpha_star = {debug_info['alpha_star'].round(2)}",
            f"e_r = {debug_info['e_r'].round(3)}",
        ]

        for i, text in enumerate(debug_texts):
            surface.blit(font.render(text, True, (0, 0, 0)), (10, 10 + i * 20))

    def plot_path(self):
        path = np.array(self.path)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        ax.plot(path[:, 0], path[:, 1], path[:, 2])
        ax.scatter(
            self.initial_pos[0],
            self.initial_pos[1],
            self.initial_pos[2],
            color="red",
            s=100,
            label="Начало",
            marker="o",
            edgecolors="black",
            linewidths=0.5,
        )

        ax.set_xlabel("x, m")
        ax.set_ylabel("y, m")
        ax.set_zlabel("z, m")

        ax.grid()
        ax.set_box_aspect((1, 1, 1))

        plt.title("Трехмерная траектория")
        plt.show()

    def plot_errors(self):
        n, m = 2, 1
        plt.figure()

        deltas = np.array([self.debug[i]["delta"] for i in range(len(self.debug))])
        plt.subplot(n, m, 1)
        plt.plot(deltas[:, 0])
        plt.plot(deltas[:, 1])
        plt.plot(deltas[:, 2])
        plt.xlabel("index")
        plt.ylabel("delta")
        plt.grid()

        es = np.array([self.debug[i]["e"] for i in range(len(self.debug))])
        plt.subplot(n, m, 2)
        plt.plot(es[:, 0])
        plt.plot(es[:, 1])
        plt.xlabel("index")
        plt.ylabel("e")
        plt.grid()

        plt.show()

    def plot_control(self):
        n, m = 2, 1
        plt.figure()

        fcs = np.array([self.debug[i]["Fc"] for i in range(len(self.debug))])
        plt.subplot(n, m, 1)
        plt.plot(fcs[:, 0])
        plt.plot(fcs[:, 1])
        plt.plot(fcs[:, 2])
        plt.ylabel("Fc")
        plt.grid()

        mcs = np.array([self.debug[i]["Mc"] for i in range(len(self.debug))])
        plt.subplot(n, m, 2)
        plt.plot(mcs[:, 0])
        plt.plot(mcs[:, 1])
        plt.plot(mcs[:, 2])
        plt.xlabel("index")
        plt.ylabel("Mc")
        plt.grid()

        plt.show()

    def plot_angles(self):
        n, m = 8, 1
        plt.figure()

        omega = np.array([self.debug[i]["omega"] for i in range(len(self.debug))])
        plt.subplot(n, m, 1)
        plt.plot(omega[:, 0])
        plt.plot(omega[:, 1])
        plt.plot(omega[:, 2])
        plt.xlabel("index")
        plt.ylabel("omega")
        plt.grid()

        omega_star = np.array(
            [self.debug[i]["omega_star"] for i in range(len(self.debug))]
        )
        plt.subplot(n, m, 2)
        plt.plot(omega_star[:, 0])
        plt.plot(omega_star[:, 1])
        plt.plot(omega_star[:, 2])
        plt.xlabel("index")
        plt.ylabel("omega_star")
        plt.grid()

        omega_dot_star = np.array(
            [self.debug[i]["omega_dot_star"] for i in range(len(self.debug))]
        )
        plt.subplot(n, m, 3)
        plt.plot(omega_dot_star[:, 0])
        plt.plot(omega_dot_star[:, 1])
        plt.plot(omega_dot_star[:, 2])
        plt.xlabel("index")
        plt.ylabel("omega_dot_star")
        plt.grid()

        e_w = np.array([self.debug[i]["e_w"] for i in range(len(self.debug))])
        plt.subplot(n, m, 4)
        plt.plot(e_w[:, 0])
        plt.plot(e_w[:, 1])
        plt.plot(e_w[:, 2])
        plt.xlabel("index")
        plt.ylabel("e_w")
        plt.grid()

        a_d = np.array([self.debug[i]["a_d"] for i in range(len(self.debug))])
        plt.subplot(n, m, 5)
        plt.plot(a_d[:, 0])
        plt.plot(a_d[:, 1])
        plt.plot(a_d[:, 2])
        plt.xlabel("index")
        plt.ylabel("a_d")
        plt.grid()

        alpha = np.array([self.debug[i]["alpha"] for i in range(len(self.debug))])
        plt.subplot(n, m, 6)
        plt.plot(alpha[:, 0])
        plt.plot(alpha[:, 1])
        plt.plot(alpha[:, 2])
        plt.xlabel("index")
        plt.ylabel("alpha")
        plt.grid()

        alpha_star = np.array(
            [self.debug[i]["alpha_star"] for i in range(len(self.debug))]
        )
        plt.subplot(n, m, 7)
        plt.plot(alpha_star[:, 0])
        plt.plot(alpha_star[:, 1])
        plt.plot(alpha_star[:, 2])
        plt.xlabel("index")
        plt.ylabel("alpha_star")
        plt.grid()

        e_r = np.array([self.debug[i]["e_r"] for i in range(len(self.debug))])
        plt.subplot(n, m, 8)
        plt.plot(e_r[:, 0])
        plt.plot(e_r[:, 2])
        plt.plot(e_r[:, 1])
        plt.xlabel("index")
        plt.ylabel("e_r")
        plt.grid()

        plt.show()

    def draw(self, surface: Surface, dt: float) -> None:
        u = self.reg.compute_control(self.robot.state, TrajSample(), dt)
        self.robot.step(u, dt)

        p = self.robot.pos

        self.path.append(p)
        self.debug.append(self.reg.debug_info.copy())

        self.draw_debug_info(surface, self.reg.debug_info)

        if perf_counter() - self.start_time > 15:
            self.plot_path()
            self.plot_errors()
            self.plot_control()
            self.plot_angles()
            exit()
