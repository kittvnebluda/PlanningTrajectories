from math import atan, cos, isclose, pi, sin, tan

import numpy as np
from constants import PI3DIV2
from core.robot.ackermann import AckermannConfig, AckermannModel, AckermannState
from pygame import Surface, Vector2, draw

from .draw import DrawParams
from .screen import ScreenParams, np_vec2screen, to_screen


def draw_cursor(
    surface: Surface,
    pose: np.ndarray,  # [x, y, theta]
    draw_params: DrawParams,
    screen_params: ScreenParams,
) -> None:
    origin = Vector2(pose[0], pose[1])
    pts = [
        Vector2(draw_params.size, 0),
        Vector2(-draw_params.size / 4, draw_params.size / 2),
        Vector2(0, 0),
        Vector2(-draw_params.size / 4, -draw_params.size / 2),
    ]
    pts = [origin + p.rotate_rad(pose[2]) for p in pts]
    pts_px = [to_screen(pt.x, pt.y, screen_params) for pt in pts]
    draw.polygon(surface, draw_params.color, pts_px)


def _draw_ackerman_body(
    surface: Surface,
    screen_params: ScreenParams,
    draw_params: DrawParams,
    robot: AckermannModel,
) -> None:
    c = robot.conf
    half_width = c.Lf
    axle_dist = c.Ls

    front_overhang = 0.9 * axle_dist
    rear_overhang = 0
    front_y = -axle_dist / 2.0 - front_overhang
    rear_y = axle_dist / 2.0 + rear_overhang

    # Немного скруглим бока / нос: используем 8 точек (обтекаемый восьмиугольник)
    body_local = [
        Vector2(+half_width * 0.85, front_y * 0.95),  # нос правый верх (вперед)
        Vector2(+half_width, front_y),  # нос правый
        Vector2(+half_width, 0.4 * axle_dist),  # правый бок ближе к переду
        Vector2(+half_width * 0.6, -0.1 * axle_dist),  # правый бок ближе к центру
        Vector2(+half_width * 0.6, rear_y),  # правый задний
        Vector2(-half_width * 0.6, rear_y),  # левый задний
        Vector2(-half_width * 0.6, -0.1 * axle_dist),  # левый центр
        Vector2(-half_width, 0.4 * axle_dist),  # левый бок ближе к переду
        Vector2(-half_width, front_y),  # нос левый
        Vector2(-half_width * 0.85, front_y * 0.95),  # нос левый верх
    ]

    # Поворот и перенос: локальная ось Y — вперёд, учёт угла robot.nu
    body = [v.rotate_rad(robot.nu) + Vector2(robot.x, robot.y) for v in body_local]

    body_px = [to_screen(p.x, p.y, screen_params) for p in body]
    draw.polygon(surface, draw_params.color, body_px)

    # Контур (обводка) чуть темнее
    draw.lines(surface, (10, 10, 10), True, body_px, max(1, draw_params.size // 2))

    # Нарисуем «капот» — небольшую центральную прямую впереди (визуальный акцент)
    hood_local_a = Vector2(0.0, front_y * 0.6)
    hood_local_b = Vector2(0.0, front_y)
    hood_a = hood_local_a.rotate_rad(robot.nu) + Vector2(robot.x, robot.y)
    hood_b = hood_local_b.rotate_rad(robot.nu) + Vector2(robot.x, robot.y)
    draw.line(
        surface,
        (200, 200, 200),
        to_screen(hood_a.x, hood_a.y, screen_params),
        to_screen(hood_b.x, hood_b.y, screen_params),
        max(1, draw_params.size // 2),
    )

    # Фары — расположим по бокам носа (по локальным координатам)
    left_light_local = Vector2(-half_width * 0.45, front_y * 0.95)
    right_light_local = Vector2(+half_width * 0.45, front_y * 0.95)
    left_light = left_light_local.rotate_rad(robot.nu) + Vector2(robot.x, robot.y)
    right_light = right_light_local.rotate_rad(robot.nu) + Vector2(robot.x, robot.y)
    draw.circle(
        surface,
        (255, 240, 200),
        to_screen(left_light.x, left_light.y, screen_params),
        max(2, draw_params.size),
    )
    draw.circle(
        surface,
        (255, 240, 200),
        to_screen(right_light.x, right_light.y, screen_params),
        max(2, draw_params.size),
    )

    # Маркеры осей колёс (чтобы видеть положение передней/задней оси)
    rear_axle_local = Vector2(0.0, -axle_dist / 2.0)
    front_axle_local = Vector2(0.0, +axle_dist / 2.0)
    rear_axle = rear_axle_local.rotate_rad(robot.nu) + Vector2(robot.x, robot.y)
    front_axle = front_axle_local.rotate_rad(robot.nu) + Vector2(robot.x, robot.y)
    draw.line(
        surface,
        (0, 0, 0),
        to_screen(rear_axle.x, rear_axle.y, screen_params),
        to_screen(rear_axle.x + 0.1, rear_axle.y, screen_params),
        1,
    )
    draw.line(
        surface,
        (0, 0, 0),
        to_screen(front_axle.x, front_axle.y, screen_params),
        to_screen(front_axle.x + 0.1, front_axle.y, screen_params),
        1,
    )


def _draw_wheel(
    surface: Surface,
    screen_params: ScreenParams,
    draw_params: DrawParams,
    state: AckermannState,
    beta: float,
    length: float,
    angle: float,
    radius: float,
) -> None:
    wr = radius
    cr = radius / 5
    thickness = wr / 2

    pts = [
        Vector2(thickness - cr, radius),
        Vector2(thickness, radius - cr),
        Vector2(thickness, -radius + cr),
        Vector2(thickness - cr, -radius),
        Vector2(-thickness + cr, -radius),
        Vector2(-thickness, -radius + cr),
        Vector2(-thickness, radius - cr),
        Vector2(-thickness + cr, radius),
    ]

    bias_local = Vector2(length * cos(angle), length * sin(angle))
    pts = [v.rotate_rad(angle + beta) + bias_local for v in pts]

    bias_global = Vector2(state.xsi[0], state.xsi[1])
    pts = [v.rotate_rad(state.xsi[2]) + bias_global for v in pts]

    pts_px = [to_screen(pt.x, pt.y, screen_params) for pt in pts]
    draw.polygon(surface, draw_params.color, pts_px)


def _draw_wheel_axes(
    surface: Surface,
    screen_params: ScreenParams,
    state: AckermannState,
    beta: float,
    length: float,
    angle: float,
) -> None:
    axis_pts = [Vector2(-10, 0), Vector2(10, 0)]

    bias_local = Vector2(length * cos(angle), length * sin(angle))
    axis_pts = [v.rotate_rad(angle + beta) + bias_local for v in axis_pts]

    bias_global = Vector2(state.xsi[0], state.xsi[1])
    axis_pts = [v.rotate_rad(state.xsi[2]) + bias_global for v in axis_pts]

    axis_pts_px = [to_screen(pt.x, pt.y, screen_params) for pt in axis_pts]
    draw.line(surface, "black", axis_pts_px[0], axis_pts_px[1])


def _calc_beta_4s(beta_3s: float, conf: AckermannConfig) -> float:
    if isclose(-conf.alpha3s, beta_3s):
        return pi - conf.alpha4s
    return (
        atan(-conf.Lf2_div_Ls - 1 / tan(beta_3s + conf.alpha3s))
        - PI3DIV2
        - conf.alpha4s
    )


def _draw_h(
    surface: Surface,
    screen_params: ScreenParams,
    robot: AckermannModel,
) -> None:
    s = robot.state
    c = robot.conf
    draw.circle(
        surface,
        "red",
        np_vec2screen(c.h(s), screen_params),
        5,
    )


def draw_ackermann(
    surface: Surface,
    robot: AckermannModel,
    draw_params: DrawParams,
    screen_params: ScreenParams,
    draw_wheel_axes: bool = False,
    draw_h: bool = False,
) -> None:
    c = robot.conf

    wheel_draw_params = DrawParams(color=(255, 0, 255))

    beta_4s = _calc_beta_4s(robot.state.beta_s, c)
    betas = (0, 0, robot.state.beta_s, beta_4s)
    lengths = (c.Lf, c.Lf, c.ls, c.ls)
    alphas = (c.alpha1f, c.alpha2f, c.alpha3s, c.alpha4s)

    _draw_ackerman_body(surface, screen_params, draw_params, robot)

    for beta, len, alpha in zip(betas, lengths, alphas):
        if draw_wheel_axes:
            _draw_wheel_axes(surface, screen_params, robot.state, beta, len, alpha)
        _draw_wheel(
            surface,
            screen_params,
            wheel_draw_params,
            robot.state,
            beta,
            len,
            alpha,
            c.r,
        )

    if draw_h:
        _draw_h(surface, screen_params, robot)
