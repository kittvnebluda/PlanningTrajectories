from abc import ABC, abstractmethod
from math import cos, sin
from time import perf_counter
from typing import Iterable

import numpy as np
from numpy.typing import NDArray


def get_rot_mat(ang: float) -> np.ndarray:
    return np.array([[cos(ang), -sin(ang)], [sin(ang), cos(ang)]])


class TrajectoryPrimitive(ABC, Iterable):
    @abstractmethod
    def __next__(self) -> NDArray | None:
        pass


class ArcPrimitive(TrajectoryPrimitive):
    def __init__(
        self,
        radius: float,
        arc_len: float,
        shift: np.ndarray,
        angle: float,
        samples: int = 200,
    ):
        self.r = radius
        self.angle = arc_len
        self.n = samples
        t = np.linspace(0, self.angle, self.n)
        self.x = self.r * np.cos(t)
        self.y = self.r * np.sin(t)
        self.pts = get_rot_mat(angle) @ np.vstack([self.x, self.y]) + shift.reshape(
            2, 1
        )
        self.i = -1

    def __next__(self) -> NDArray | None:
        if self.i >= len(self.x) - 1:
            return None
        self.i += 1
        return self.pts[:, self.i]

    def __iter__(self):
        return self


class StraightLineTimedPrimitive(TrajectoryPrimitive):
    def __init__(self, end=(1.0, 1.0), time=10, samples=200):
        self.end = end
        self.n = samples
        self.time = time

        self.start_time = perf_counter()

        self.x = np.linspace(0, self.end[0], self.n)
        self.y = np.linspace(0, self.end[1], self.n)

        self.i = -1

    def __next__(self) -> NDArray | None:
        if (
            self.i >= len(self.x) - 1
        ):  # perf_counter() - self.start_time >= self.time or
            return None
        self.i += 1
        return np.array([self.x[self.i], self.y[self.i]])

    def __iter__(self):
        return StraightLineTimedPrimitive(self.end, self.time, self.n)
