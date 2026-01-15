from .draw import DrawParams, draw_points, draw_polyline, draw_vector, draw_frame
from .gridviz import draw_grid, fit_map_to_screen, fit_screen_to_map
from .robotviz import draw_ackermann, draw_cursor
from .screen import (
    ScreenParams,
    from_screen,
    np_vec2screen,
    to_screen,
    vec2screen,
)


class _DrawAPI:
    __slots__ = ()
    points = staticmethod(draw_points)
    polyline = staticmethod(draw_polyline)
    vector = staticmethod(draw_vector)
    frame = staticmethod(draw_frame)


class _GridVizAPI:
    __slots__ = ()
    draw_grid = staticmethod(draw_grid)
    fit_map_to_screen = staticmethod(fit_map_to_screen)
    fit_screen_to_map = staticmethod(fit_screen_to_map)


class _ScreenAPI:
    __slots__ = ()
    to = staticmethod(to_screen)
    from_px = staticmethod(from_screen)
    vec = staticmethod(vec2screen)
    np_vec = staticmethod(np_vec2screen)


class _RobotVizAPI:
    __slots__ = ()
    cursor = staticmethod(draw_cursor)
    ackermann = staticmethod(draw_ackermann)


draw = _DrawAPI()
gridviz = _GridVizAPI()
screen = _ScreenAPI()
robotviz = _RobotVizAPI()

__all__ = ["draw", "gridviz", "DrawParams", "screen", "ScreenParams", "robotviz"]
