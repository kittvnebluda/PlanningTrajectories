from typing import Iterable, NamedTuple

import numpy as np
from pygame import Surface, Vector2, draw

from .utils import ScreenParams, to_screen


class DrawParams(NamedTuple):
    size: int = 2
    color: tuple[int, int, int] = (0, 0, 255)
    draw_waypoints: bool = False
    dashed: bool = False
    dash_len: float = 0.05
    gap_len: float = 0.05


def draw_robot(
    surface: Surface, state, draw_params: DrawParams, screen_params: ScreenParams
) -> None:
    theta = state.pose.theta

    p1 = Vector2(draw_params.size, 0).rotate_rad(theta)
    p2 = Vector2(-draw_params.size / 4, draw_params.size / 2).rotate_rad(theta)
    p3 = Vector2(-draw_params.size / 4, -draw_params.size / 2).rotate_rad(theta)

    pts = (p1 + state.pose.pos, p2 + state.pose.pos, p3 + state.pose.pos)
    pts_px = [to_screen(pt.x, pt.y, screen_params) for pt in pts]

    draw.polygon(surface, draw_params.color, pts_px)


def draw_pts(
    surface: Surface,
    pts: Iterable,
    draw_params: DrawParams,
    screen_params: ScreenParams,
) -> None:
    for pt in pts:
        draw.circle(
            surface,
            draw_params.color,
            to_screen(pt[0], pt[1], screen_params),
            draw_params.size,
        )


def draw_line(
    surface: Surface,
    pts: np.ndarray,
    draw_params: DrawParams,
    screen_params: ScreenParams,
) -> None:
    for i in range(len(pts) - 1):
        draw.line(
            surface,
            draw_params.color,
            to_screen(pts[i][0], pts[i][1], screen_params),
            to_screen(pts[i + 1][0], pts[i + 1][1], screen_params),
            draw_params.size,
        )
