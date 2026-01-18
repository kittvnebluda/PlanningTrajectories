from copy import deepcopy
from functools import cached_property
from typing import Any, Generator, Iterable, Optional, Sequence

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

    last_sample_pos = shift.copy()
    for prim in primitives:
        it = iter(prim)
        sample: TrajSample | None = next(it)

        while sample is not None:
            sample.pos = sample.pos + shift
            yield sample
            last_sample_pos = sample.pos.copy()
            sample = next(it)

        shift = last_sample_pos.copy()


class SequenceTrajectory(Trajectory):
    def __init__(
        self, primitives: Iterable[TrajectoryPrimitive], shift: Optional[NDArray] = None
    ) -> None:
        self._gen_factory = lambda: stitch(primitives, shift)  # ← important!
        self._gen = self._gen_factory()
        self.last_gen = TrajSample()

    def sample(self) -> TrajSample:
        gen = next(self._gen)
        if gen is None:
            self.last_gen.vel = VEC2_ZERO
            self.last_gen.acc = VEC2_ZERO
            self.last_gen.jerk = VEC2_ZERO
            return self.last_gen
        self.last_gen = gen
        return gen

    @cached_property
    def samples(self) -> Sequence[TrajSample]:
        return tuple(self._gen_factory())

    @cached_property
    def samples_pos(self) -> Sequence[NDArray]:
        return [s.pos for s in self.samples]
