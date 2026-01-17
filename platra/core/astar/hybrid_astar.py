import logging
import timeit
from dataclasses import dataclass
from queue import PriorityQueue
from typing import Optional, Self

import numpy as np
from numba import njit

from platra.core.astar.heuristics import h_euclidian
from platra.core.map import Grid
from platra.core.map import grid as gr
from platra.core.robot.ackermann import AckermannConfig, AckermannModel, AckermannState

STEER_SET = [-0.6, -0.1, 0.0, 0.1, 0.6]
SPEED_SET = [0.1, -0.1]

dt = 0.1
N = 20

DTHETA = 2 / 15
TIMEOUT = 120

logger = logging.getLogger(__name__)


@dataclass
class HybridNode:
    x: float
    y: float
    theta: float
    beta: float

    g: float
    h: float

    parent: Optional[Self]
    control: tuple  # (v, steer)

    def f(self):
        return self.g + self.h

    def key(self, grid_params):
        cell = gr.from_world((self.x, self.y), *grid_params)
        theta_bin = int((self.theta % (2 * np.pi)) / DTHETA)
        return (cell[0], cell[1], theta_bin)

    def __lt__(self, obj: Self) -> bool:
        if self.f() < obj.f():
            return True
        return False


def generate_primitives(conf):
    primitives = {}
    for v in SPEED_SET:
        for steer in STEER_SET:
            model = AckermannModel(conf)

            xs, ys, thetas = [], [], []
            cost = 0.0

            for _ in range(N):
                model.step(v, steer, dt)
                xs.append(model.x)
                ys.append(model.y)
                thetas.append(model.nu)
                cost += abs(v) * dt

            primitives[(v, steer)] = {
                "xs": np.array(xs),
                "ys": np.array(ys),
                "thetas": np.array(thetas),
                "cost": cost,
            }
    return primitives


@njit
def car_footprint_points(length: float, half_width: float):
    length *= 1.05
    half_width *= 1.05
    xs = np.linspace(-half_width, half_width, 5)
    ys = np.linspace(0, length, 3)
    return np.array([[x, y] for x in xs for y in ys])


def footprint_collision(x, y, theta, grid, grid_params, length, half_width):
    pts = car_footprint_points(length, half_width)

    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    for px, py in pts:
        wx = x + cos_t * px - sin_t * py
        wy = y + sin_t * px + cos_t * py

        cell = gr.from_world((wx, wy), *grid_params)
        if not grid.is_free(cell):
            return True

    return False


class HybridAStarPlanner:
    def __init__(self, conf: AckermannConfig) -> None:
        self.conf = conf
        self.closed_set = {}
        self.primitives = generate_primitives(conf)

    def heuristic(self, node: HybridNode, goal):
        return h_euclidian((node.x, node.y), (goal[0], goal[1]))

    def is_goal(self, node: HybridNode, goal):
        pos_ok = np.hypot(node.x - goal[0], node.y - goal[1]) < 0.2
        angle_ok = abs(
            (node.theta - goal[2] + np.pi) % (2 * np.pi) - np.pi
        ) < np.deg2rad(15)
        return pos_ok and angle_ok

    def simulate_motion(
        self,
        grid: Grid,
        grid_params,
        node: HybridNode,
        v,
        steer,
        goal,
    ) -> Optional[HybridNode]:
        model = AckermannModel(
            self.conf, AckermannState(np.array([node.x, node.y, node.theta]), node.beta)
        )

        cost = 0.0
        for _ in range(N):
            model.step(v, steer, dt)
            cost += abs(v) * dt
            if footprint_collision(
                model.x,
                model.y,
                model.nu,
                grid,
                grid_params,
                self.conf.Ls,
                self.conf.Lf,
            ):
                return None
        if v < 0:
            cost *= 5  # штраф за задний ход
        if steer * node.control[1] < 0:
            cost *= 2  # штраф за поворот

        new_node = HybridNode(
            x=model.x,
            y=model.y,
            theta=model.nu,
            beta=model.beta_s,
            g=node.g + cost,
            h=0.0,
            parent=node,
            control=(v, steer),
        )
        new_node.h = self.heuristic(new_node, goal)
        return new_node

    def reconstruct_path(self, node: HybridNode):
        path = [node]
        while 1:
            if node.parent is None:
                break
            node = node.parent
            path.append(node)

        path.reverse()
        return path

    def plan(self, start, goal, grid: Grid, grid_params):
        """
        start, goal: (x, y, theta)
        """
        start_node = HybridNode(
            x=start[0],
            y=start[1],
            theta=start[2],
            beta=0.0,
            g=0.0,
            h=0.0,
            parent=None,
            control=(0, 0),
        )
        start_node.h = self.heuristic(start_node, goal)

        open_set = PriorityQueue()
        open_set.put((start_node.f(), start_node))
        self.closed_set = dict()

        time_start = timeit.default_timer()
        while not open_set.empty() and (timeit.default_timer() - time_start) < TIMEOUT:
            _, current = open_set.get()
            key = current.key(grid_params)

            if key in self.closed_set:
                continue
            self.closed_set[key] = current

            if self.is_goal(current, goal):
                return self.reconstruct_path(current)

            for v in SPEED_SET:
                for steer in STEER_SET:
                    child = self.simulate_motion(
                        grid, grid_params, current, v, steer, goal
                    )
                    if child is None:
                        continue

                    child_key = child.key(grid_params)
                    if child_key in self.closed_set:
                        continue

                    open_set.put((child.f(), child))

        logger.info("Hybrid A*: path not found")
        return []
