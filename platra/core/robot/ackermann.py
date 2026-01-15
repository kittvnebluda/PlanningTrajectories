from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from math import atan2, hypot, pi
from typing import Optional

from numba import njit
import numpy as np


@dataclass
class AckermannState:
    xsi: np.ndarray = field(default_factory=lambda: np.zeros(3))  # [x, y, nu]
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
class AckermannStateExt(AckermannState):
    v1: float = field(default=0.0)
    v2: float = field(default=0.0)


class AckermannConfig(ABC):
    """
    Configuration parameters for the Ackermann steering model.

    Attributes:
        Lf (float): Distance from the center of the robot's coordinate system to a fixed wheels
        Ls (float): Distance from the center of the robot's coordinate system to steering wheels in Y axis
        e (float): Distance from the center of the robot's coordinate system to a following point
        r (float): Wheel radius
    """

    def __init__(self, Lf: float, Ls: float, e: float, r: float) -> None:
        if e == 0:
            raise Exception("'e' cannot be a zero in AckermannConfig")

        self.Lf = Lf
        self.Ls = Ls
        self.Lf2_div_Ls = Lf * 2 / Ls
        self.ls = hypot(Lf, Ls)

        self.alpha1f: float = 0.0
        self.alpha2f: float = pi
        self.alpha3s = atan2(-Ls, Lf)
        self.alpha4s = atan2(-Ls, -Lf)

        self.e = e
        self.r = r

    # R^T(nu)*Sigma(beta_s)
    @abstractmethod
    def rts(self, state: AckermannState) -> np.ndarray: ...
    # h(z)
    @abstractmethod
    def h(self, state: AckermannState) -> np.ndarray: ...

    # # K^{-1}(z)
    # @abstractmethod
    # def k_inv(self, state: AckermannState) -> np.ndarray: ...
    # # g(z)
    # @abstractmethod
    # def g(self, state: AckermannState) -> np.ndarray: ...


class AckermannModel:
    def __init__(
        self, conf: AckermannConfig, initial_state: Optional[AckermannState] = None
    ):
        self.conf = conf
        if initial_state is None:
            self.state = AckermannState()
        else:
            self.state = initial_state

    def step(self, v1: float, v2: float, dt: float) -> None:
        assert not np.isnan(v1)
        assert not np.isnan(v2)
        assert not np.isnan(dt)
        s = self.state
        s.xsi = s.xsi + dt * self.conf.rts(s) * s.eta
        s.beta_s += dt * s.zeta
        s.eta += dt * v1
        s.zeta += dt * v2

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
