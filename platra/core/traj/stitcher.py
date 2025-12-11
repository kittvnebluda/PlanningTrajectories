from typing import Any, Generator, Iterable, Optional

import numpy as np
from numpy.typing import NDArray

from .primitives import TrajectoryPrimitive


def stitch(
    primitives: Iterable[TrajectoryPrimitive],
    shift: Optional[np.ndarray] = None,
) -> Generator[NDArray, Any, None]:
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
