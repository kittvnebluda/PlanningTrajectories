from typing import List, Tuple

from pygame import Vector2


class TrajectoryData:
    def __init__(self, waypoints: List[Tuple[float, float]], corner_radii: List[float]):
        self.waypoints = [Vector2(x, y) for x, y in waypoints]
        self.corner_radii = list(corner_radii)


TRIANGLE = TrajectoryData(waypoints=[(-2, -2), (0.5, 0.5), (2, -2)], corner_radii=[1])
SIX = TrajectoryData(
    waypoints=[(-5, -2), (-0.5, 0.5), (2, -2), (3, -2), (3, 2), (-5, 3)],
    corner_radii=[2, 0.5, 0.2, 3],
)
CROSS_SHAPE = TrajectoryData(
    waypoints=[(0, 0), (1, 1), (0, 1), (1, 0)], corner_radii=[0.1, 0.1]
)
FIGURE_EIGHT = TrajectoryData(
    waypoints=[(0, 0), (1, 1), (-1, 1), (1, -1), (-1, -1), (0, 0)],
    corner_radii=[0.4, 0.4, 0.4, 0.4],
)
BOOK_EX = TrajectoryData(
    waypoints=[
        (-5, -3),
        (-4, -1),
        (-2, -1),
        (4, -3),
        (4, -1),
        (3, 3),
        (2, 3),
        (2, 2),
        (1, 2),
        (-5, 3),
    ],
    corner_radii=[1.2, 2, 1, 2, 0.5, 0.5, 0.5, 2],
)

TRAJ_WPS: dict[str, TrajectoryData] = {
    "triangle": TRIANGLE,
    "six": SIX,
    "cross": CROSS_SHAPE,
    "eight": FIGURE_EIGHT,
    "book_ex": BOOK_EX,
}
