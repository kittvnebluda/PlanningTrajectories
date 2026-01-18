from queue import PriorityQueue

from platra.typings import Cell

from ..map import Grid
from .heuristics import h_euclidian
from .neighbors import grid_neighbors_8


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

        neighbors = grid_neighbors_8(map, current)
        for next in neighbors:
            new_cost = cost_so_far[current] + map.cost(current, next)
            if next not in cost_so_far or new_cost < cost_so_far[next]:
                frontier.put((new_cost + h_euclidian(current, goal), next))
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
