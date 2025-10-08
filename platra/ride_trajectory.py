from argparse import ArgumentParser
from math import atan2

import pygame
from pygame import Vector2, draw

from .trajectory import Trajectory, TrajectoryC0, TrajectoryC1, TrajectoryC2
from .utils import (
    SCRN_HEIGHT,
    SCRN_WIDTH,
    GameState,
    Pose,
    Twist,
    cmdvel,
    draw_pts,
    draw_robot,
    fix_angle,
    vec2screen,
)

from .trajectory_collection import TRAJECTORIES


def main():
    while state.running:
        # Pygame Events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                state.running = False

        screen.fill("white")  # Clearing screen

        # Stuffff
        path_record.append(state.pose.pos.copy())
        target = t.get_target(state.pose.pos)
        e = target - state.pose.pos
        sp = min((e * 10).magnitude(), 1)
        phi = atan2(e.y, e.x)
        theta_err = 10 * fix_angle(phi - state.pose.theta)
        cmdvel(state, Twist(sp, 0, theta_err))

        # Drawing
        t.draw_trajectory(screen, 2)
        draw_pts(screen, path_record, color="black", size=1)
        draw_robot(screen, state)
        draw.circle(screen, "purple", vec2screen(target), 3)

        # Display the game
        pygame.display.flip()

        # FPS and time delta
        state.dt = clock.tick(120) / 1000


path_record: list[Vector2] = []

parser = ArgumentParser()
subparsers = parser.add_subparsers(
    dest="continuity_class",
    required=True,
    help="Continuity class of a trajectory",
)
p0 = subparsers.add_parser("C0")
p0.set_defaults(factory=lambda w, r, args: TrajectoryC0(w))

p1 = subparsers.add_parser("C1")
p1.set_defaults(factory=lambda w, r, args: TrajectoryC1(w, r))

p2 = subparsers.add_parser("C2")
p2.add_argument("k", type=float, help="Cubic parabola coefficient")
p2.set_defaults(factory=lambda w, r, args: TrajectoryC2(w, r, args.k))

args = parser.parse_args()

pygame.init()
screen = pygame.display.set_mode((SCRN_WIDTH, SCRN_HEIGHT))
clock = pygame.time.Clock()

state = GameState(Pose(pygame.Vector2(-5, -3)), Twist())

t: Trajectory = args.factory(
    TRAJECTORIES["book_ex"].waypoints, TRAJECTORIES["book_ex"].corner_radii, args
)

try:
    main()
finally:
    pygame.quit()
