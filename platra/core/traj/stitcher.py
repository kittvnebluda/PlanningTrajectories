from typing import Any, Generator, Iterable, Optional

import numpy as np
from numpy.typing import NDArray

from platra.core.traj.traj import Trajectory

from .segments import TrajectoryPrimitive
from .traj import TrajSample

VEC2_ZERO = np.zeros(2)


def stitch(
    primitives: Iterable[TrajectoryPrimitive],
    shift: Optional[np.ndarray] = None,
) -> Generator[TrajSample, Any, None]:
    if shift is None:
        shift = np.array([0.0, 0.0])
    else:
        assert len(shift) == 2

    for prim in primitives:
        it = iter(prim)
        pt = next(it)
        pt_last = pt.copy()

        while pt is not None:
            pt = shift + pt
            pt_last = pt.copy()
            yield pt
            pt = next(it)

        shift = pt_last


class SequenceTrajectory(Trajectory):
    def __init__(
        self, primitives: Iterable[TrajectoryPrimitive], shift: Optional[NDArray] = None
    ) -> None:
        self.gena = stitch(primitives, shift)
        self.last_gen = TrajSample()

    def sample(self) -> TrajSample:
        gen = next(self.gena)
        if gen is None:
            self.last_gen.vel = VEC2_ZERO
            self.last_gen.acc = VEC2_ZERO
            self.last_gen.jerk = VEC2_ZERO
            return self.last_gen
        self.last_gen = gen
        return gen

    def to_samples(self) -> Iterable[TrajSample]:
        return tuple(self.gena)
