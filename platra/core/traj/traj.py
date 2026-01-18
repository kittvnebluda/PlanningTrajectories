from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from functools import cached_property
from typing import NamedTuple, Sequence

import numpy as np
from numpy.typing import NDArray


@dataclass
class TrajSample:
    pos: NDArray = field(default_factory=lambda: np.zeros(2))
    vel: NDArray = field(default_factory=lambda: np.zeros(2))
    acc: NDArray = field(default_factory=lambda: np.zeros(2))
    jerk: NDArray = field(default_factory=lambda: np.zeros(2))


# TODO: How can we reset trajectory?
class Trajectory(ABC):
    @abstractmethod
    def sample(self) -> TrajSample: ...
    @cached_property
    @abstractmethod
    def samples(self) -> Sequence[TrajSample]: ...
    @cached_property
    @abstractmethod
    def samples_pos(self) -> Sequence[NDArray]: ...


class InterpType(Enum):
    C0 = auto()
    C1 = auto()
    C2 = auto()
    BSpline = auto()


class TrajParams(NamedTuple):
    resolution: float = 0.1
    smooth_radius: float = 0.5  # default corner radius for C1 smoothing
    curvature_gain: float = 0.3  # gain for C2 smoothing (controls k)
    bspline_degree: int = 3
    interp_type: InterpType = InterpType.C0
