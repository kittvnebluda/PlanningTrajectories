from argparse import ArgumentParser
from typing import Type

import pygame

from platra.labs.lab3 import TrajStabilization2DEuclidianSpiral

from .labs import (
    Laboratory,
    Parking,
    Teleop,
    TrajPlanning,
    TrajStabilization2D,
    TrajStabilization3D,
    TrajTracking,
)

lab2_choice: dict[str, Type[Laboratory]] = {
    "tracking": TrajTracking,
    "teleop": Teleop,
}

lab3_choice: dict[str, Type[Laboratory]] = {
    "2d": TrajStabilization2D,
    "2d_spiral": TrajStabilization2DEuclidianSpiral,
    "3d": TrajStabilization3D,
}

parser = ArgumentParser()
subparser = parser.add_subparsers()

lab1_parser = subparser.add_parser("lab1")
lab1_parser.set_defaults(factory=lambda args: TrajPlanning())

lab2_parser = subparser.add_parser("lab2")
lab2_parser.add_argument("task", type=str, choices=lab2_choice.keys())
lab2_parser.set_defaults(factory=lambda args: lab2_choice[args.task]())

lab3_parser = subparser.add_parser("lab3")
lab3_parser.add_argument("task", type=str, choices=lab3_choice.keys())
lab3_parser.set_defaults(factory=lambda args: lab3_choice[args.task]())

parking_parser = subparser.add_parser("parking")
parking_parser.set_defaults(factory=lambda args: Parking())

args = parser.parse_args()

lab: Laboratory = args.factory(args)

pygame.init()

screen = pygame.display.set_mode((lab.screen_params.width, lab.screen_params.height))
clock = pygame.time.Clock()
dt = clock.tick(120) / 1000

running = True
pause = False
try:
    while running:
        for event in pygame.event.get():
            match event.type:
                case pygame.QUIT:
                    running = False
                case pygame.KEYUP:
                    lab.handle_keyup(event.key)
                case pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        pause = not pause
                    lab.handle_keydown(event.key)

        lab.handle_mouse(screen)
        screen.fill((255, 255, 255))
        lab.draw(screen, dt if not pause else 0)
        pygame.display.flip()

        dt = clock.tick(500) / 1000
finally:
    lab.on_close()
