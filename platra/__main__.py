from argparse import ArgumentParser
from typing import Type

import pygame

from .debug import test_hybrid_astar as tha
from .labs import (
    Laboratory,
    Teleop,
    TrajPlanning,
    TrajStabilization2D,
    TrajStabilization2DEuclidianSpiral,
    TrajTracking,
)

lab2_choice: dict[str, Type[Laboratory]] = {
    "tracking": TrajTracking,
    "teleop": Teleop,
}

lab3_choice: dict[str, Type[Laboratory]] = {
    "2d": TrajStabilization2D,
    "2d_spiral": TrajStabilization2DEuclidianSpiral,
}

parser = ArgumentParser()
subparser = parser.add_subparsers(dest="mode", required=True)

# ----------------- LABS -----------------
lab1_parser = subparser.add_parser("lab1")
lab1_parser.set_defaults(factory=lambda args: TrajPlanning())

lab2_parser = subparser.add_parser("lab2")
lab2_parser.add_argument("task", type=str, choices=lab2_choice.keys())
lab2_parser.set_defaults(factory=lambda args: lab2_choice[args.task]())

lab3_parser = subparser.add_parser("lab3")
lab3_parser.add_argument("task", type=str, choices=lab3_choice.keys())
lab3_parser.set_defaults(factory=lambda args: lab3_choice[args.task]())

# ----------------- DEBUG -----------------
debug_parser = subparser.add_parser("debug")
debug_subparser = debug_parser.add_subparsers(dest="debug_module", required=True)

parking_parser = debug_subparser.add_parser("parking")
parking_parser.add_argument(
    "task_id",
    type=int,
    choices=list(range(len(tha.tasks))),
    help="Index of the parking scenario to run",
)
parking_parser.set_defaults(factory="debug_parking")

args = parser.parse_args()

if args.mode == "debug":
    if args.debug_module == "parking":
        if not (0 <= args.task_id < len(tha.tasks)):
            raise Exception("Wrong task id")
        start = tha.tasks[args.task_id][0]
        goal = tha.tasks[args.task_id][1]
        tha.main(start, goal)

else:
    lab: Laboratory = args.factory(args)

    pygame.init()

    screen = pygame.display.set_mode(
        (lab.screen_params.width, lab.screen_params.height)
    )
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
