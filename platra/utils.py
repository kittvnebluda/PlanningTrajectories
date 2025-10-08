from math import cos, isclose, pi, sin
from typing import Callable, Iterable

import numpy as np
from pygame import Surface, Vector2, draw

METERS2PX = 100
PX2METERS = 1 / METERS2PX
SCRN_WIDTH = 1280
SCRN_HEIGHT = 720
SCRN_HALF_WIDTH = int(SCRN_WIDTH / 2)
SCRN_HALF_HEIGHT = int(SCRN_HEIGHT / 2)
PI_NEG = -pi
PI_HALF = pi / 2
PI_DOUBLE = 2 * pi
PI_DOUBLE_NEG = 2 * PI_NEG

N = 500
cubic_parabola_xs = np.linspace(0, 10, N)


class Twist:
    def __init__(self, vx: float = 0, vy: float = 0, w: float = 0) -> None:
        self.v = Vector2(vx, vy)
        self.w = w


class Pose:
    def __init__(self, position: Vector2, orientation: float = 0) -> None:
        self.pos = position
        self.theta = orientation


class GameState:
    def __init__(self, pose: Pose, velocity: Twist) -> None:
        self.pose = pose
        self.vel = velocity
        self.dt = 0.0
        self.running = True


def cmdvel(state: GameState, twist: Twist):
    state.vel = twist
    vx, vy, w = twist.v.x, twist.v.y, twist.w
    dt = state.dt
    dx = vx * dt
    dy = vy * dt
    state.pose.theta += w * dt
    state.pose.pos.x += cos(state.pose.theta) * dx - sin(state.pose.theta) * dy
    state.pose.pos.y += sin(state.pose.theta) * dx + cos(state.pose.theta) * dy


def x2screen(x: float) -> int:
    return int(x * METERS2PX + SCRN_HALF_WIDTH)


def y2screen(y: float) -> int:
    return int(-y * METERS2PX + SCRN_HALF_HEIGHT)


def from_screen(x: int, y: int) -> Vector2:
    return Vector2(
        (x - SCRN_HALF_WIDTH) * PX2METERS, -(y - SCRN_HALF_HEIGHT) * PX2METERS
    )


def to_screen(x: float, y: float) -> tuple[int, int]:
    return x2screen(x), y2screen(y)


def vec2screen(v: Vector2) -> tuple[int, int]:
    return to_screen(v.x, v.y)


def fix_angle(x: float) -> float:
    if x > pi:
        return x + PI_DOUBLE_NEG
    if x < PI_NEG:
        return x + PI_DOUBLE
    return x


def draw_robot(surface: Surface, state, color="red", size=0.1) -> None:
    theta = state.pose.theta

    p1 = Vector2(size, 0).rotate_rad(theta)
    p2 = Vector2(-size / 4, size / 2).rotate_rad(theta)
    p3 = Vector2(-size / 4, -size / 2).rotate_rad(theta)

    pts = (p1 + state.pose.pos, p2 + state.pose.pos, p3 + state.pose.pos)
    pts_px = [to_screen(pt.x, pt.y) for pt in pts]

    draw.polygon(surface, color, pts_px)


def draw_pts(surface: Surface, pts: Iterable[Vector2], color="blue", size=5) -> None:
    for pt in pts:
        draw.circle(surface, color, to_screen(pt.x, pt.y), size)


def signed_dist_to_line(
    point: Vector2, line_point: Vector2, angle: float | int, bias: float | int
) -> float:
    """
    Calculates signed distance to a line containing point 'line_point'
    which is perpenducilar to a line with 'angle'
    """
    return (
        cos(angle) * (point.x - line_point.x)
        + sin(angle) * (point.y - line_point.y)
        + bias
    )


def draw_line(
    surface: Surface,
    a: Vector2,
    b: Vector2,
    color,
    size,
    dash_length,
    gap_length,
    dashed=False,
) -> None:
    if not dashed:
        draw.line(surface, color, vec2screen(a), vec2screen(b), size)
    v = b - a
    length = v.magnitude()
    if isclose(length, 0):
        return
    steps = int(length // (dash_length + gap_length))
    direction = v.normalize()

    start = a
    for _ in range(steps):
        end = start + direction * dash_length
        draw.line(
            surface,
            color,
            to_screen(start.x, start.y),
            to_screen(end.x, end.y),
            size,
        )
        start = end + direction * gap_length


def draw_arc(
    surface: Surface,
    circle_center: Vector2,
    turn_direction: int,
    angle_step: float | int,
    start_point: Vector2,
    q_function: Callable[[Vector2], float],
    color,
    size,
    dashed=False,
) -> Vector2:
    """Draws arc and returns arc end in global coordinates"""
    q = 0
    start = start_point.copy()
    end = start_point.copy()
    while q <= 0:
        end = (end - circle_center).rotate_rad(
            turn_direction * angle_step
        ) + circle_center
        draw.line(
            surface,
            color,
            vec2screen(start),
            vec2screen(end),
            size,
        )
        if dashed:
            end = (end - circle_center).rotate_rad(
                turn_direction * angle_step
            ) + circle_center
        q = q_function(end)
        start = end
    return end


def draw_cubic_parabola(
    surface: Surface,
    k: float | int,
    psi: float | int,
    shift: Vector2,
    sign: int,
    q: Callable[[Vector2], float],
    color=(200, 0, 255),
    width=2,
):
    yL = sign * k * (cubic_parabola_xs**3)
    R = np.array([[cos(psi), -sin(psi)], [sin(psi), cos(psi)]])
    pts_local = np.vstack((cubic_parabola_xs, yL))
    pts_global = R @ pts_local + np.array([[shift.x], [shift.y]])

    for i in range(N - 1):
        p2 = Vector2(pts_global[0, i + 1], pts_global[1, i + 1])
        if q(p2) > 0:
            break
        p1 = Vector2(pts_global[0, i], pts_global[1, i])
        draw.line(surface, color, vec2screen(p1), vec2screen(p2), width)


def draw_binary_map(surface):
    pass
