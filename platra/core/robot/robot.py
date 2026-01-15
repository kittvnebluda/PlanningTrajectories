from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from platra.core.traj import TrajSample


@dataclass
class MassPointConfig:
    mass: float
    inertia_moment: float


class RobotController(ABC):
    @abstractmethod
    def compute_control(self, state, target: TrajSample, dt: float) -> np.ndarray: ...


class Robot:
    def __init__(
        self,
        conf,
        controller: RobotController,
        initial_state=None,
    ):
        self.model = ...
        self.controller = controller
        self.state = ...

    def step(self, target, dt):
        v1, v2 = self.controller.compute_control(self.state, target, dt)
        ...
