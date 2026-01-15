from dataclasses import dataclass, field

import numpy as np
from pygame import Vector2

from platra.types import Pixel


@dataclass(frozen=True)
class ScreenParams:
    width: int
    height: int
    meter2px: float = 100.0
    shift: Pixel = 0, 0
    half_width: float = field(init=False)
    half_height: float = field(init=False)
    px2meters: float = field(init=False)

    def __post_init__(self):
        object.__setattr__(self, "half_width", self.width / 2)
        object.__setattr__(self, "half_height", self.height / 2)
        object.__setattr__(self, "px2meters", 1 / self.meter2px)


def x2px(x: float, params: ScreenParams) -> int:
    return int(x * params.meter2px + params.half_width + params.shift[0])


def y2px(y: float, params: ScreenParams) -> int:
    return int(-y * params.meter2px + params.half_height + params.shift[1])


def from_screen(x: int, y: int, params: ScreenParams) -> tuple[float, float]:
    return (
        (x - params.half_width) * params.px2meters,
        -(y - params.half_height) * params.px2meters,
    )


def to_screen(x: float, y: float, params: ScreenParams) -> Pixel:
    return x2px(x, params), y2px(y, params)


def vec2screen(v: Vector2, params: ScreenParams) -> Pixel:
    return to_screen(v.x, v.y, params)


def np_vec2screen(v: np.ndarray, params: ScreenParams) -> Pixel:
    assert len(v) == 2, "Numpy array must be 2D to be converted to pixels"
    return to_screen(v[0], v[1], params)
