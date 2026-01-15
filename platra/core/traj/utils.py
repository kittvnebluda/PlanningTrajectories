import numpy as np
from numba import njit
from numpy.typing import NDArray

from platra.core.traj.traj import Trajectory


@njit
def compute_curvature(p_prev: NDArray, p_curr: NDArray, p_next: NDArray) -> float:
    """Compute discrete curvature for three points."""
    a = np.linalg.norm(p_curr - p_prev)
    b = np.linalg.norm(p_next - p_curr)
    c = np.linalg.norm(p_next - p_prev)

    if a * b * c == 0:
        return 0.0

    area = 0.5 * np.abs(
        (p_prev[0] - p_next[0]) * (p_curr[1] - p_prev[1])
        - (p_prev[0] - p_curr[0]) * (p_next[1] - p_prev[1])
    )
    return 4.0 * area / (a * b * c)


@njit
def iqr_mask(arr, k=1.5) -> NDArray:
    q1, q3 = np.percentile(arr, [25, 75])
    iqr = q3 - q1
    lower = q1 - k * iqr
    upper = q3 + k * iqr
    return (arr >= lower) & (arr <= upper)  # Boolean mask of outliers


def traj_curvature(traj: Trajectory) -> NDArray:
    poses = traj.samples_pos
    curv = []
    for i in range(len(poses) - 2):
        curv.append(compute_curvature(poses[i], poses[i + 1], poses[i + 2]))
    mask = iqr_mask(curv)
    return np.array(curv)[mask]


def traj_length(traj: Trajectory) -> np.float64:
    poses = traj.samples_pos
    length = np.float64(0)
    for i in range(len(poses) - 1):
        x, y = poses[i] - poses[i + 1]
        length += np.sqrt(x**2 + y**2)
    return length
