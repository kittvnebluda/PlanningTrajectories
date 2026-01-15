from dataclasses import dataclass
from queue import PriorityQueue
from typing import Optional, Self

import numpy as np

from platra.core.astar.heuristics import h_euclidian
from platra.core.map import Grid
from platra.core.map import grid as gr
from platra.core.robot.ackermann import AckermannConfig, AckermannModel, AckermannState

STEER_SET = [-1, 0.0, +1]
SPEED_SET = [+0.1, -0.1]

dt = 0.1
N = 20

DTHETA = 2 / 10


@dataclass
class HybridNode:
    x: float
    y: float
    theta: float
    beta: float

    g: float
    h: float

    parent: Optional[Self]
    control: Optional[tuple]  # (v, steer)

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


class HybridAStarPlanner:
    def __init__(self, conf: AckermannConfig) -> None:
        self.conf = conf
        self.closed_set = {}
        self.steer_prev = 0
        self.primitives = generate_primitives(conf)

    def heuristic(self, node: HybridNode, goal):
        return h_euclidian((node.x, node.y), (goal[0], goal[1]))

    def is_goal(self, node: HybridNode, goal):
        pos_ok = np.hypot(node.x - goal[0], node.y - goal[1]) < 0.3
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

            cell = gr.from_world((model.x, model.y), *grid_params)
            if not grid.is_free(cell):
                return None
        if v < 0:
            cost *= 2  # штраф за задний ход
        if steer * self.steer_prev < 0:
            cost *= 2  # штраф за поворот
        self.steer_prev = steer

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
            control=None,
        )
        start_node.h = self.heuristic(start_node, goal)

        open_set = PriorityQueue()
        open_set.put((start_node.f(), start_node))
        self.closed_set = dict()

        while not open_set.empty():
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

        print("Hybrid A*: path not found")
        return []
