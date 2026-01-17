from abc import ABC, abstractmethod

from disp.screen import ScreenParams
from pygame import Surface, event


class Laboratory(ABC):
    screen_params: ScreenParams

    @abstractmethod
    def draw(self, surface: Surface, dt: float) -> None: ...

    def handle_keydown(self, key: event.Event) -> None:
        pass

    def handle_keyup(self, key: event.Event) -> None:
        pass

    def handle_mouse(self, surface: Surface) -> None:
        pass

    def on_close(self) -> None:
        pass
