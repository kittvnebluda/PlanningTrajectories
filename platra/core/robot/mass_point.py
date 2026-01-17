from dataclasses import dataclass, field
from math import cos, sin
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from utils import fix_angle

from .robot import MassPointConfig


@dataclass
class MassPointState:
    x: float = field(default=0.0)
    y: float = field(default=0.0)
    vx: float = field(default=0.0)
    vy: float = field(default=0.0)
    alpha: float = field(default=0.0)  # Point orientation
    omega: float = field(default=0.0)  # Point angular velocity


class MassPointModel:
    def __init__(
        self, conf: MassPointConfig, initial_state: Optional[MassPointState] = None
    ):
        self.conf = conf
        if initial_state is None:
            self.state = MassPointState()
        else:
            self.state = initial_state

    def step(self, u: NDArray, dt: float) -> None:
        assert len(u) == 3, "Control must be vector 3 🫵"

        s = self.state
        m = self.conf.mass
        j = self.conf.inertia_moment

        s.omega += dt * u[2] / j
        s.alpha += dt * s.omega
        s.alpha = fix_angle(s.alpha)
        s.vy += dt * (u[0] * sin(s.alpha) - u[1] * cos(s.alpha)) / m
        s.vx += dt * (u[0] * cos(s.alpha) + u[1] * sin(s.alpha)) / m
        s.y += dt * s.vy
        s.x += dt * s.vx

    @property
    def x(self):
        return self.state.x

    @property
    def y(self):
        return self.state.y

    @property
    def vx(self):
        return self.state.vx

    @property
    def vy(self):
        return self.state.vy

    @property
    def alpha(self):
        return self.state.alpha

    @property
    def omega(self):
        return self.state.omega

    @property
    def pos(self):
        return np.array([self.x, self.y])

    @property
    def pose(self):
        return np.array([self.x, self.y, self.alpha])
