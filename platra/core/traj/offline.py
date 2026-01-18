from math import atan2, cos, isclose, pi, sin, sqrt

import numpy as np
from numba import njit
from numpy.typing import NDArray

from platra.typings import Number

from .traj import TrajParams

# ============================================================
#  Geometry helpers
# ============================================================


def compute_corner_radius(
    p1: NDArray[np.float64], p2: NDArray[np.float64], p3: NDArray[np.float64]
) -> tuple[float, float]:
    """
    Compute the maximum turning radius given 3 consecutive points.
    Return it and angle between points.
    """
    v1 = p1 - p2
    v2 = p3 - p2
    a, b = np.linalg.norm(v1), np.linalg.norm(v2)

    if a == 0 or b == 0:
        return 0, 0

    cos_angle = np.clip(np.dot(v1, v2) / (a * b), -1.0, 1.0)
    angle = np.arccos(cos_angle)

    if angle < 1e-6:
        return 0, 0

    if isclose(angle, pi, abs_tol=0.00001):
        return np.inf, pi

    bisector = v1 + v2
    bisector /= np.linalg.norm(bisector)

    r = a * abs(np.tan(angle / 2)) / 2
    return r, angle


@njit
def compute_min_k(r: Number) -> float:
    return 18 / (25 * r**2 * 5 ** (1 / 2)) * 1.5


@njit
def get_rot_mat(ang: float) -> np.ndarray:
    return np.array([[cos(ang), -sin(ang)], [sin(ang), cos(ang)]])


@njit
def bspline_basis(t: float, i: int, k: int, knots: NDArray[np.float64]) -> float:
    if k == 0:
        return 1.0 if knots[i] <= t < knots[i + 1] else 0.0

    left_num = (t - knots[i]) * bspline_basis(t, i, k - 1, knots)
    left_den = knots[i + k] - knots[i]
    left = left_num / left_den if left_den != 0 else 0

    right_num = (knots[i + k + 1] - t) * bspline_basis(t, i + 1, k - 1, knots)
    right_den = knots[i + k + 1] - knots[i + 1]
    right = right_num / right_den if right_den != 0 else 0

    return left + right


@njit
def bspline_basis_vectorized(
    ts: NDArray[np.float64], i: int, k: int, knots: NDArray[np.float64]
) -> NDArray[np.float64]:
    basis = []
    for t in ts:
        basis.append(bspline_basis(t, i, k, knots))
    return np.array(basis, dtype=np.float64)


# ============================================================
#  Trajectory interpolation
# ============================================================


def interpolate_c0(waypoints: np.ndarray, params: TrajParams) -> np.ndarray:
    """C0 continuity (piecewise linear)."""
    if len(waypoints) < 2:
        return waypoints
    pts = []
    for i in range(len(waypoints) - 1):
        p1, p2 = waypoints[i], waypoints[i + 1]
        segment = np.linspace(p1, p2, int(np.linalg.norm(p2 - p1) / params.resolution))
        if len(segment) == 0:
            continue
        pts.append(segment)
    return np.vstack(pts)


def interpolate_arc(
    center: np.ndarray,
    r: float,
    p_entry: np.ndarray,
    p_exit: np.ndarray,
    sign: int | np.ndarray,
    pts_num: int,
) -> np.ndarray:
    start_angle = np.arctan2(p_entry[1] - center[1], p_entry[0] - center[0])
    end_angle = np.arctan2(p_exit[1] - center[1], p_exit[0] - center[0])
    # Ensure direction of rotation
    if sign > 0 and end_angle < start_angle:
        end_angle += 2 * np.pi
    elif sign < 0 and end_angle > start_angle:
        end_angle -= 2 * np.pi

    angles = np.linspace(start_angle, end_angle, pts_num)
    return np.stack(
        [center[0] + r * np.cos(angles), center[1] + r * np.sin(angles)], axis=1
    )


def interpolate_c1(waypoints: np.ndarray, params: TrajParams) -> np.ndarray:
    """C1 continuity using circular arc blending at corners."""
    pts = []
    p_exit_last = waypoints[0]
    for i in range(1, len(waypoints) - 1):
        a, b, c = waypoints[i - 1], waypoints[i], waypoints[i + 1]
        r, corner_angle = compute_corner_radius(a, b, c)

        if not r:
            continue

        if isclose(corner_angle, pi):
            pts.extend(interpolate_c0(np.array([p_exit_last, b]), params))
            p_exit_last = b
            continue

        dir1 = b - a
        dir1 /= np.linalg.norm(dir1)
        dir2 = c - b
        dir2 /= np.linalg.norm(dir2)

        sign = int(np.sign(np.cross(dir1, dir2)))
        half_corner = corner_angle / 2
        r = min(r, params.smooth_radius)
        d = r / abs(np.tan(half_corner))
        dc = r / abs(np.sin(half_corner))
        phi1 = atan2(dir1[1], dir1[0])
        cc = get_rot_mat(pi - sign * half_corner + phi1) @ np.array([dc, 0]) + b

        p_entry = b - dir1 * d
        p_exit = b + dir2 * d

        n_points = max(8, int(corner_angle / params.resolution * r / 2))
        arc = interpolate_arc(cc, r, p_entry, p_exit, sign, n_points)

        pts.extend(interpolate_c0(np.array([p_exit_last, p_entry]), params))
        pts.extend(arc)
        p_exit_last = p_exit

    pts.extend(interpolate_c0(np.array([p_exit_last, waypoints[-1]]), params))
    return np.array(pts)


def interpolate_cubic_parabola(
    p_exit: np.ndarray,
    k: float,
    resolution: float,
    rotation: float = 0,
) -> np.ndarray:
    R = get_rot_mat(rotation)
    xL_end = abs((R.T @ p_exit)[0])
    xL = np.arange(0, xL_end, resolution)
    yL = k * (xL**3)
    pts_local = np.vstack((xL, yL))
    pts = R @ pts_local
    return pts.T


def interpolate_c2(waypoints: np.ndarray, params: TrajParams) -> np.ndarray:
    """C2 continuity by connecting straight lines, cubic parabolas and arcs."""
    pts = []
    p_exit_last = waypoints[0]
    for i in range(1, len(waypoints) - 1):
        a, b, c = waypoints[i - 1], waypoints[i], waypoints[i + 1]
        r, corner_angle = compute_corner_radius(a, b, c)

        if not r:
            continue

        if isclose(corner_angle, pi):
            pts.extend(interpolate_c0(np.array([p_exit_last, b]), params))
            p_exit_last = b
            continue

        half_corner_angle = corner_angle / 2
        r = min(r, params.smooth_radius)
        d = r / abs(np.tan(half_corner_angle))
        dc = r / abs(np.sin(half_corner_angle))
        k = compute_min_k(r)
        k = max(k, params.curvature_gain)

        dir1 = b - a
        dir1 /= np.linalg.norm(dir1)
        dir2 = c - b
        dir2 /= np.linalg.norm(dir2)
        sign = int(np.sign(np.cross(dir1, dir2)))
        phi1 = np.atan2(dir1[1], dir1[0])
        phi2 = np.atan2(dir2[1], dir2[0])

        parab1_entry = b - dir1 * d
        parab1_exit = (
            get_rot_mat(phi1)
            @ (np.array([1 / (6 * k * r), sign * k / (6 * k * r) ** 3]))
            + parab1_entry
        )
        parab2_exit = b + dir2 * d
        parab2_entry = (
            get_rot_mat(phi2 - pi)
            @ (np.array([1 / (6 * k * r), -sign * k / (6 * k * r) ** 3]))
            + parab2_exit
        )

        h = np.linalg.norm(parab2_entry - parab1_exit)
        dc1 = sqrt(r**2 - (h / 2) ** 2)
        dc2 = sqrt(np.linalg.norm(b - parab2_entry) ** 2 - (h / 2) ** 2)
        dc = dc1 + dc2
        cc = get_rot_mat(pi - sign * half_corner_angle + phi1) @ np.array([dc, 0]) + b

        line = interpolate_c0(np.array([p_exit_last, parab1_entry]), params)
        parab1 = (
            interpolate_cubic_parabola(
                parab1_exit - parab1_entry, sign * k, params.resolution, phi1
            )
            + parab1_entry
        )
        parab2 = (
            interpolate_cubic_parabola(
                parab2_exit - parab2_entry, -sign * k, params.resolution, phi2 - pi
            )[::-1]
            + parab2_exit
        )
        n_points = max(8, int(abs(corner_angle) / params.resolution * r / 2))
        arc = interpolate_arc(cc, r, parab1_exit, parab2_entry, sign, n_points)

        pts.extend(line)
        pts.extend(parab1)
        pts.extend(arc)
        pts.extend(parab2)
        p_exit_last = parab2_exit

    pts.extend(interpolate_c0(np.array([p_exit_last, waypoints[-1]]), params))
    return np.array(pts)


def interpolate_bsplines(waypoints: np.ndarray, params: TrajParams) -> np.ndarray:
    n = len(waypoints)
    if n == 2:
        print("Only two waypoints! Using C0 continuity trajectory")
        return interpolate_c0(waypoints, params)

    k = params.bspline_degree
    if n <= k:
        print(f"Not enough waypoints to use given B-spline curve degree ({k})!")
        k = n - 1
        print(f"Lowered curve degree to {k}")

    if not (2 <= k <= n + 1):
        print("Bad B spline curve order!")
        return np.array([])

    knots = np.concatenate(
        (
            np.zeros(k, dtype=np.float64),
            np.linspace(0, 1, n - k + 1, dtype=np.float64),
            np.ones(k, dtype=np.float64),
        )
    )
    t = np.arange(0, 1, params.resolution, dtype=np.float64)
    curve = np.zeros((len(t), waypoints.shape[1]), dtype=np.float64)
    for i in range(n):
        Ni = bspline_basis_vectorized(t, i, k, knots)[:, None]
        curve += Ni * waypoints[i]
    curve = np.vstack((curve, waypoints[-1]))
    return interpolate_c0(curve, params)
