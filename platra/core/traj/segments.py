from abc import ABC, abstractmethod
from time import perf_counter
from typing import Iterable

import numpy as np

from .offline import interpolate_c0
from .traj import TrajParams, TrajSample
from platra.utils import rot_t as get_rot_mat


class TrajectoryPrimitive(ABC, Iterable):
    @abstractmethod
    def __next__(self) -> TrajSample | None:
        pass


class ArcPrimitive(TrajectoryPrimitive):
    def __init__(
        self,
        radius: float,
        arc_len: float,
        center_shift: np.ndarray,
        circle_rot: float,
        resolution: float,
        vel_norm: float,
    ):
        theta_end = arc_len / radius
        theta = np.arange(0, theta_end, resolution)
        pos = radius * np.vstack([np.cos(theta), np.sin(theta)])
        rot = get_rot_mat(circle_rot)
        self.pos = rot @ pos + center_shift.reshape(2, 1)

        vel = np.vstack([-np.sin(theta), np.cos(theta)])
        vel = vel / np.linalg.norm(vel, axis=1) * vel_norm
        self.vel = rot @ vel

        acc_mag = vel_norm**2 / radius
        acc = np.vstack([-np.cos(theta), -np.sin(theta)])
        acc = acc / np.linalg.norm(acc, axis=1) * acc_mag
        self.acc = rot * acc

        self.jerk = np.zeros(2)

        self.last_index = len(self.pos) - 1
        self.i = -1

    def __next__(self):
        if self.i >= self.last_index:
            return None
        self.i += 1
        return TrajSample(
            self.pos[self.i], self.vel[self.i], self.acc[self.i], self.jerk
        )

    def __iter__(self):
        self.i = -1
        return self


class StraightLinePrimitive(TrajectoryPrimitive):
    def __init__(
        self, end: tuple[float | int, float | int], vel_norm: float, resolution: float
    ):
        self.start_time = perf_counter()

        self.pos = interpolate_c0(
            np.array([(0, 0), end]), TrajParams(resolution=resolution)
        )
        self.last_index = len(self.pos) - 1
        self.vel = np.array(end) / np.linalg.norm(end) * vel_norm
        self.acc = np.array([0, 0])
        self.jerk = np.array([0, 0])

        self.i = -1

    def __next__(self):
        if self.i >= self.last_index:
            return None
        self.i += 1
        return TrajSample(self.pos[self.i], self.vel, self.acc, self.jerk)

    def __iter__(self):
        self.i = -1
        return self
