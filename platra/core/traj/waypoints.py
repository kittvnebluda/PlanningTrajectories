from abc import ABC, abstractmethod
from functools import cached_property
from itertools import cycle
from typing import Sequence

from numpy.typing import NDArray

from .offline import (
    interpolate_bsplines,
    interpolate_c0,
    interpolate_c1,
    interpolate_c2,
)
from .traj import InterpType, Trajectory, TrajParams, TrajSample

INTERP_FUNCTIONS = {
    InterpType.C0: interpolate_c0,
    InterpType.C1: interpolate_c1,
    InterpType.C2: interpolate_c2,
    InterpType.BSpline: interpolate_bsplines,
}


class WaypointInterpolator(ABC):
    @abstractmethod
    def generate(self, waypoints: NDArray, params: TrajParams) -> NDArray: ...


class WaypointsTrajectory(Trajectory):
    def __init__(self, waypoints: NDArray, params: TrajParams) -> None:
        assert waypoints.shape[1] == 2, "Waypoints must be 2D"
        self._inter_type_cycle = cycle([t for t in InterpType])
        self.wps = waypoints
        self.params = params
        self.interp_type = params.interp_type
        self.reset_pts()

    def sample(self) -> TrajSample:
        if self._i >= self.length - 1:
            raise StopIteration("Trajectory exhausted")
        self._i += 1
        return TrajSample(pos=self._pts[self._i])

    @cached_property
    def samples(self) -> Sequence[TrajSample]:
        return [TrajSample(pt) for pt in self._pts]

    @cached_property
    def samples_pos(self) -> Sequence[NDArray]:
        return [t.pos for t in self.samples]

    def _invalidate_cache(self):
        self.__dict__.pop("samples", None)
        self.__dict__.pop("samples_pos", None)

    def reset_pts(self):
        self._pts = INTERP_FUNCTIONS[self.interp_type](self.wps, self.params)
        self._i = -1
        self._invalidate_cache()
        self.length = len(self._pts)

    def set_interp_type(self, new_type: InterpType) -> None:
        if new_type is self.interp_type:
            return
        self.interp_type = new_type
        self.reset_pts()

    def next_interp_type(self):
        self.set_interp_type(next(self._inter_type_cycle))

    @property
    def pt(self) -> NDArray:
        return self._pts[self._i]

    def set_waypoints(self, waypoints: NDArray) -> None:
        self.wps = waypoints
        self.reset_pts()

    @property
    def pts(self) -> NDArray:
        return self._pts
