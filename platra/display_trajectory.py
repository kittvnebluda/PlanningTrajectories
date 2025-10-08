import pygame
from argparse import ArgumentParser
from platra.trajectory import Trajectory, TrajectoryC0, TrajectoryC1, TrajectoryC2
from platra.utils import SCRN_HEIGHT, SCRN_WIDTH
from .trajectory_collection import TRAJECTORIES


def test_draw(t: Trajectory):
    pygame.init()
    screen = pygame.display.set_mode((SCRN_WIDTH, SCRN_HEIGHT))
    clock = pygame.time.Clock()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        screen.fill((255, 255, 255))
        t.draw_trajectory(screen, draw_size=2)
        pygame.display.flip()
        clock.tick(10)

    pygame.quit()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("name", type=str, help="Trajectory name from the collection")
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

    t = args.factory(
        TRAJECTORIES[args.name].waypoints, TRAJECTORIES[args.name].corner_radii, args
    )
    test_draw(t)
