from typing import Protocol

from pygame import Surface, event

from platra.disp.screen import ScreenParams


class Laboratory(Protocol):
    screen_params: ScreenParams

    def handle_keydown(self, key: event.Event) -> None: ...
    def handle_keyup(self, key: event.Event) -> None: ...
    def handle_mouse(self, surface: Surface) -> None: ...
    def draw(self, surface: Surface, dt: float) -> None: ...
