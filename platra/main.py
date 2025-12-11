from argparse import ArgumentParser
from typing import Type

import pygame
import numpy as np

from .labs.lab1_traj_planning import Lab1TrajPlanning
from .labs.lab2_traj_tracking import Lab2Teleop, Lab2TrajTracking
from .labs.protocols import Laboratory

labs: dict[str, Type[Laboratory]] = {"tracking": Lab2TrajTracking, "teleop": Lab2Teleop}

parser = ArgumentParser()
subparser = parser.add_subparsers()

lab1_parser = subparser.add_parser("lab1")
lab1_parser.set_defaults(factory=lambda args: Lab1TrajPlanning())

lab2_parser = subparser.add_parser("lab2")
lab2_parser.add_argument("task", type=str, choices=["tracking", "teleop"])
lab2_parser.set_defaults(factory=lambda args: labs[args.task]())

args = parser.parse_args()

lab: Laboratory = args.factory(args)

pygame.init()

screen = pygame.display.set_mode((lab.screen_params.width, lab.screen_params.height))
clock = pygame.time.Clock()
dt = clock.tick(120) / 1000

running = True
while running:
    for event in pygame.event.get():
        match event.type:
            case pygame.QUIT:
                running = False
            case pygame.KEYDOWN:
                lab.handle_keydown(event.key)
            case pygame.KEYUP:
                lab.handle_keyup(event.key)

    lab.handle_mouse(screen)
    screen.fill((255, 255, 255))
    lab.draw(screen, dt)
    pygame.display.flip()

    dt = clock.tick(120) / 1000
