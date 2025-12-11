from queue import PriorityQueue

from .map.grid import Grid
from ..types import Cell


def _heuristic(c1: Cell, c2: Cell):
    return ((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2) ** (1 / 2)


def _discover_map(map: Grid, start: Cell, goal: Cell) -> dict[Cell, Cell]:
    frontier = PriorityQueue()
    frontier.put((0, start))
    came_from = dict()
    cost_so_far = dict()
    came_from[start] = None
    cost_so_far[start] = 0

    while not frontier.empty():
        current = frontier.get()[1]

        if current == goal:
            return came_from

        neighbors = map.neighbors(current)
        for next in neighbors:
            new_cost = cost_so_far[current] + map.cost(current, next)
            if next not in cost_so_far or new_cost < cost_so_far[next]:
                frontier.put((new_cost + _heuristic(current, goal), next))
                came_from[next] = current
                cost_so_far[next] = new_cost

    return came_from


def astar(map: Grid, start: Cell, goal: Cell) -> list[Cell]:
    came_from = _discover_map(map, start, goal)
    path = []
    current = goal
    try:
        while current != start:
            path.append(current)
            current = came_from[current]
    except KeyError:
        print("Can't reach goal")
        return []
    path.append(start)
    path.reverse()
    return path
