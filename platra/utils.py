from dataclasses import dataclass, field
from math import cos, pi, sin

import numpy as np
from pygame import Vector2

from .constants import (
    PI_DOUBLE,
    PI_DOUBLE_NEG,
    PI_NEG,
)
from .state import GameState, Twist


@dataclass(frozen=True)
class ScreenParams:
    width: int
    height: int
    meters2px: float = 100.0
    half_width: float = field(init=False)
    half_height: float = field(init=False)
    px2meters: float = field(init=False)

    def __post_init__(self):
        object.__setattr__(self, "half_width", self.width / 2)
        object.__setattr__(self, "half_height", self.height / 2)
        object.__setattr__(self, "px2meters", 1 / self.meters2px)


def cmdvel(state: GameState, twist: Twist):
    state.vel = twist
    vx, vy, w = twist.v.x, twist.v.y, twist.w
    dt = state.dt
    dx = vx * dt
    dy = vy * dt
    state.pose.theta += w * dt
    state.pose.pos.x += cos(state.pose.theta) * dx - sin(state.pose.theta) * dy
    state.pose.pos.y += sin(state.pose.theta) * dx + cos(state.pose.theta) * dy


def x2screen(x: float, params: ScreenParams) -> int:
    return int(x * params.meters2px + params.half_width)


def y2screen(y: float, params: ScreenParams) -> int:
    return int(-y * params.meters2px + params.half_height)


def from_screen(x: int, y: int, params: ScreenParams) -> tuple[float, float]:
    return (
        (x - params.half_width) * params.px2meters,
        -(y - params.half_height) * params.px2meters,
    )


def to_screen(x: float, y: float, params: ScreenParams) -> tuple[int, int]:
    return x2screen(x, params), y2screen(y, params)


def vec2screen(v: Vector2, params: ScreenParams) -> tuple[int, int]:
    return to_screen(v.x, v.y, params)


def fix_angle(x: float) -> float:
    if x > pi:
        return x + PI_DOUBLE_NEG
    if x < PI_NEG:
        return x + PI_DOUBLE
    return x


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


def np_array_to_vectors(array: np.ndarray) -> list[Vector2]:
    return [Vector2(x, y) for x, y in array]
