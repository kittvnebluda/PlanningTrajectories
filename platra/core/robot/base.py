from dataclasses import dataclass, field
from typing import Optional, Protocol

import numpy as np


@dataclass
class RobotState:
    xsi: np.ndarray = field(default_factory=lambda: np.zeros(3))  # [x, y, nu] - SO3
    beta_s: float = field(default=0.0)  # Steering wheel angle
    eta: float = field(default=0.0)  # xsi control
    zeta: float = field(default=0.0)  # beta_s control

    @property
    def x(self):
        return self.xsi[0]

    @x.setter
    def x(self, x: float):
        self.xsi[0] = x

    @property
    def y(self):
        return self.xsi[1]

    @y.setter
    def y(self, y: float):
        self.xsi[1] = y

    @property
    def nu(self):
        return self.xsi[2]

    @nu.setter
    def nu(self, nu: float):
        self.xsi[2] = nu


@dataclass
class RobotStateExt(RobotState):
    v1: float = field(default=0.0)
    v2: float = field(default=0.0)


class RobotConfig(Protocol):
    # R^T(nu)*Sigma(beta_s)
    def rts(self, state: RobotState) -> np.ndarray: ...
    # h(z)
    def h(self, state: RobotState) -> np.ndarray: ...


class RobotController(Protocol):
    def update_control(
        self, state: RobotState, target: np.ndarray, dt: float
    ) -> np.ndarray: ...


class RobotModel[T: RobotConfig]:
    def __init__(self, conf: T, initial_state: Optional[RobotState] = None):
        self.conf = conf
        if initial_state is None:
            self.state = RobotState(xsi=np.zeros(3), beta_s=0.0, eta=0.0, zeta=0.0)
        else:
            self.state = initial_state

    def step(self, v1: float, v2: float, dt: float) -> None:
        assert not np.isnan(v1)
        assert not np.isnan(v2)
        assert not np.isnan(dt)
        s = self.state
        s.xsi = s.xsi + dt * self.conf.rts(s) * s.eta
        s.beta_s += dt * s.zeta
        # clipped, s.beta_s = self.conf.clip_beta_s(s.beta_s)
        s.eta += dt * v1
        s.zeta += dt * v2
        # if clipped == 1 and v2 > 0 or clipped == 2 and v2 < 0:
        #     s.zeta = 0

    @property
    def x(self) -> float:
        return self.state.xsi[0]

    @property
    def y(self) -> float:
        return self.state.xsi[1]

    @property
    def nu(self) -> float:
        return self.state.xsi[2]

    @property
    def beta_s(self) -> float:
        return self.state.beta_s

    @property
    def eta(self) -> float:
        return self.state.eta

    @property
    def zeta(self) -> float:
        return self.state.zeta


class Robot:
    def __init__(
        self,
        conf: RobotConfig,
        controller: RobotController,
        initial_state: Optional[RobotState] = None,
    ):
        self.model = RobotModel(conf, initial_state)
        self.controller = controller
        self.state = self.model.state

    def step(self, target, dt):
        v1, v2 = self.controller.update_control(self.state, target, dt)
        self.model.step(v1, v2, dt)
