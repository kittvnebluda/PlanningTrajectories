from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from utils import orthonormalize, vee_op, wedge_op

from .robot import MassPointConfig


@dataclass
class MassPointState3D:
    pos: NDArray = field(default_factory=lambda: np.zeros(3))
    vel: NDArray = field(default_factory=lambda: np.zeros(3))
    omega: NDArray = field(default_factory=lambda: np.zeros(3))
    R: NDArray = field(default_factory=lambda: wedge_op(np.ones(3)))

    def alpha(self):
        return vee_op(self.R - self.R.T) * 0.5


class MassPointModel3D:
    def __init__(
        self, conf: MassPointConfig, initial_state: Optional[MassPointState3D] = None
    ):
        self.conf = conf
        if initial_state is None:
            self.state = MassPointState3D()
        else:
            self.state = initial_state
        self.omega_dot = np.zeros(3)

    def step(self, u: NDArray, dt: float) -> None:
        assert len(u) == 6, "Control must be vector 6 🫵"

        s = self.state
        m = self.conf.mass
        j = self.conf.inertia_moment
        A = np.array([[m, 0, 0], [0, m, 0], [0, 0, j]])

        s.vel = s.vel + dt * u[:3] / m
        s.pos = s.pos + dt * s.vel
        self.omega_dot = self.omega_dot + dt * np.linalg.inv(A) @ (
            u[3:] - np.cross(s.omega, A @ s.omega)
        )
        s.omega = s.omega + dt * self.omega_dot
        s.R = s.R + dt * wedge_op(s.omega) @ s.R
        s.R = orthonormalize(s.R)

    @property
    def pos(self):
        return self.state.pos

    @property
    def vel(self):
        return self.state.vel

    @property
    def alpha(self):
        return self.state.alpha

    @property
    def omega(self):
        return self.state.omega
