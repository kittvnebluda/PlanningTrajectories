from typing import Iterable, NamedTuple, Sequence

import numpy as np
from numpy.typing import NDArray
from pygame import Surface, Vector2, draw

from .screen import ScreenParams, np_vec2screen, to_screen, vec2screen


class DrawParams(NamedTuple):
    size: int = 2
    color: tuple[int, int, int] = (0, 0, 255)
    draw_waypoints: bool = False
    dashed: bool = False
    dash_len: float = 0.05
    gap_len: float = 0.05


def draw_points(
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


def draw_polyline(
    surface: Surface,
    pts: Sequence,
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


def draw_vector(
    surface: Surface,
    origin: NDArray,
    vec: NDArray,
    draw_params: DrawParams,
    screen_params: ScreenParams,
) -> None:
    """
    Draws a 2D vector on a Pygame surface as a line with a triangular arrowhead.

    Args:
        surface (pygame.Surface): The Pygame surface onto which the vector will
            be drawn.
        origin (numpy.ndarray): A 2-element array [x, y] representing the
            starting point of the vector in world coordinates.
        vec (numpy.ndarray): A 2-element array [vx, vy] representing the vector
            components in world coordinates.
        draw_params (DrawParams): Rendering parameters.
        screen_params (ScreenParams): Parameters for mapping world coordinates.

    Returns:
        None
    """
    assert len(origin) == 2, "Origin must be 2D"
    assert len(vec) == 2, "Vector must be 2D"

    end = origin + vec

    norm = np.linalg.norm(vec)
    if norm == 0:
        return

    vec = vec / norm
    n = np.array([-vec[1], vec[0]])  # perpendicular

    head_len = norm / 20 * draw_params.size
    head_width = head_len

    base = end - head_len * vec
    tip = end
    left = base + 0.5 * head_width * n
    right = base - 0.5 * head_width * n

    draw.line(
        surface,
        draw_params.color,
        np_vec2screen(origin, screen_params),
        np_vec2screen(base, screen_params),
        draw_params.size,
    )

    pts = [
        to_screen(tip[0], tip[1], screen_params),
        to_screen(left[0], left[1], screen_params),
        to_screen(right[0], right[1], screen_params),
    ]

    draw.polygon(surface, draw_params.color, pts)


def draw_frame(
    surface: Surface,
    screen_params: ScreenParams,
    x: float,
    y: float,
    angle: float,
    colors=("red", "green"),
    size: int = 3,
) -> None:
    axis_len = 0.2 * size

    origin = Vector2(x, y)
    dir = Vector2(axis_len, 0).rotate_rad(angle)

    draw.line(
        surface,
        colors[0],
        vec2screen(origin, screen_params),
        vec2screen(origin + dir, screen_params),
        size,
    )
    draw.line(
        surface,
        colors[1],
        vec2screen(origin, screen_params),
        vec2screen(origin + dir.rotate(90), screen_params),
        size,
    )
