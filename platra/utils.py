from math import cos, pi, sin

import numpy as np
from numpy.typing import NDArray

from .constants import PI_DOUBLE, PI_NEG


def rot_t(ang: float) -> np.ndarray:
    return np.array([[cos(ang), sin(ang)], [-sin(ang), cos(ang)]])


def rot(ang: float) -> np.ndarray:
    return np.array([[cos(ang), -sin(ang)], [sin(ang), cos(ang)]])


def fix_angle(ang: float) -> float:
    while abs(ang) > pi:
        if ang < PI_NEG:
            ang += PI_DOUBLE
        if ang > pi:
            ang -= PI_DOUBLE
    return ang


def fix_angle_vec(ang: NDArray) -> NDArray:
    return np.array([fix_angle(a) for a in ang])


def rot_x(ang: float) -> np.ndarray:
    return np.array([[1, 0, 0], [0, cos(ang), -sin(ang)], [0, sin(ang), cos(ang)]])


def rot_y(ang: float) -> np.ndarray:
    return np.array([[cos(ang), 0, sin(ang)], [0, 1, 0], [-sin(ang), 0, cos(ang)]])


def rot_z(ang: float) -> np.ndarray:
    return np.array([[cos(ang), -sin(ang), 0], [sin(ang), cos(ang), 0], [0, 0, 1]])


def rot_3d(ang: NDArray) -> NDArray:
    return rot_z(ang[2]) @ rot_y(ang[1]) @ rot_x(ang[0])


def wedge_op(w):
    assert w.shape == (3,)
    return np.array(
        [
            [0, w[2], -w[1]],
            [-w[2], 0, w[0]],
            [w[1], -w[0], 0],
        ]
    )


def vee_op(S: NDArray):
    assert isinstance(S, np.ndarray) and S.shape == (3, 3), (
        "Матрица должна быть размером 3×3"
    )
    assert np.allclose(S, -S.T, atol=1e-8), "Матрица не является кососимметрической"

    x = S[1, 2]
    y = S[2, 0]
    z = S[0, 1]

    return np.array([x, y, z])


def orthonormalize(R):
    U, _, Vt = np.linalg.svd(R)
    return U @ Vt
