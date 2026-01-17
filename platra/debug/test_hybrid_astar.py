import logging
import timeit
from argparse import ArgumentParser
from math import pi

import matplotlib.pyplot as plt
import numpy as np
from core.astar.hybrid_astar import HybridAStarPlanner
from core.map import Grid
from core.robot.configs import AckermannConfigForStaticFeedback
from core.symbolic.ackermann import LambdifiedAckermannForStaticFeedback
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Polygon

DEMO_VID_FILEPATH = "demo_has.mp4"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def draw_ackermann(ax, x, y, theta, half_width, length, color, alpha):
    corners = np.array(
        [
            [half_width, length],
            [half_width, 0],
            [-half_width, 0],
            [-half_width, length],
        ]
    )

    R = np.array(
        [
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)],
        ]
    )
    corners = (R @ corners.T).T
    corners[:, 0] += x
    corners[:, 1] += y

    body = Polygon(corners, closed=True, color=color, alpha=alpha)
    ax.add_patch(body)

    theta += pi / 2
    fx = x + np.cos(theta) * half_width * 0.6
    fy = y + np.sin(theta) * half_width * 0.6
    ax.plot([x, fx], [y, fy], color="black", linewidth=2)


def main(start, goal):
    grid_map = Grid.from_name("parking_lot")

    scale = 0.2
    shift_x = 0.0
    shift_y = 0.0
    grid_params = (scale, shift_x, shift_y)

    start_time = timeit.default_timer()
    conf = AckermannConfigForStaticFeedback.from_symbolic(
        0.35, 1.3, 0.4, 0.2, LambdifiedAckermannForStaticFeedback
    )
    logger.info(
        f"Robot config initialized in {timeit.default_timer() - start_time:.2f} s."
    )

    planner = HybridAStarPlanner(conf)
    start_time = timeit.default_timer()
    path = planner.plan(start, goal, grid_map, grid_params)
    logger.info(f"Hybrid A* elapsed in {timeit.default_timer() - start_time:.2f} s.")

    xs = [n.x for n in path]
    ys = [n.y for n in path]
    thetas = [n.theta for n in path]
    betas = [n.beta for n in path]
    vs = [n.control[0] for n in path]
    steers = [n.control[1] for n in path]

    t = np.arange(len(path))

    fig, axs = plt.subplots(2, 3, figsize=(14, 8), sharex=True)

    # --- x ---
    axs[0, 0].plot(t, xs, label="x")
    axs[0, 0].axhline(goal[0], linestyle="--", label="Целевой")
    axs[0, 0].set_ylabel("x, м")
    axs[0, 0].set_title("Координата X")
    axs[0, 0].legend()
    axs[0, 0].grid(True)

    # --- y ---
    axs[0, 1].plot(t, ys, label="y")
    axs[0, 1].axhline(goal[1], linestyle="--", label="Целевой")
    axs[0, 1].set_ylabel("y, м")
    axs[0, 1].set_title("Координата Y")
    axs[0, 1].legend()
    axs[0, 1].grid(True)

    # --- theta ---
    axs[0, 2].plot(t, thetas, label=r"$\theta(t)$")
    axs[0, 2].axhline(goal[2], linestyle="--", label="Целевой")
    axs[0, 2].set_ylabel(r"$\theta$, рад")
    axs[0, 2].set_title("Ориентация робота")
    axs[0, 2].legend()
    axs[0, 2].grid(True)

    # --- beta ---
    axs[1, 0].plot(t, betas)
    axs[1, 0].set_ylabel(r"$\beta$, рад")
    axs[1, 0].set_title("Угол рулевого колеса")
    axs[1, 0].grid(True)

    # --- velocity ---
    axs[1, 1].step(t, vs, where="post")
    axs[1, 1].set_ylabel("v, м/с")
    axs[1, 1].set_title("Линейная скорость")
    axs[1, 1].grid(True)

    # --- steering ---
    axs[1, 2].step(t, steers, where="post")
    axs[1, 2].set_ylabel(r"$\zeta$")
    axs[1, 2].set_title("Скорость поворота колеса")
    axs[1, 2].grid(True)

    for ax in axs[1, :]:
        ax.set_xlabel("Step")

    fig.suptitle("Hybrid A*: состояния и управления", fontsize=14)
    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots()
    occ = grid_map.occupancy
    ax.imshow(
        occ,
        origin="upper",
        extent=[
            shift_x,
            shift_x + occ.shape[1] * scale,
            shift_y - occ.shape[0] * scale,
            shift_y,
        ],  # pyright: ignore[reportArgumentType]
    )
    if len(path) == 0:
        for node in planner.closed_set.values():
            draw_ackermann(
                ax, node.x, node.y, node.theta, conf.Lf, conf.Ls, "lightgray", alpha=0.1
            )
    for node in path[::10]:
        draw_ackermann(
            ax, node.x, node.y, node.theta, conf.Lf, conf.Ls, "red", alpha=0.4
        )
    ax.plot(xs, ys, "b-", label="Hybrid A* Path")
    ax.plot(start[0], start[1], "go", label="Start")
    ax.plot(goal[0], goal[1], "bx", label="Goal")
    ax.set_aspect("equal")
    ax.grid()
    ax.legend()
    ax.set_title("Hybrid A* Ackermann Parking")
    plt.show()

    fig, ax = plt.subplots()
    ax.imshow(
        occ,
        origin="upper",
        extent=[
            shift_x,
            shift_x + occ.shape[1] * scale,
            shift_y - occ.shape[0] * scale,
            shift_y,
        ],  # pyright: ignore[reportArgumentType]
    )
    ax.set_aspect("equal")
    ax.grid()

    robot_patches = []

    def update(i):
        for p in robot_patches:
            p.remove()
        robot_patches.clear()

        node = path[i]
        ax.plot(start[0], start[1], "go", label="Start")
        ax.plot(goal[0], goal[1], "bx", label="Goal")
        draw_ackermann(ax, node.x, node.y, node.theta, conf.Lf, conf.Ls, "red", 0.8)

        robot_patches.extend(ax.patches)
        return robot_patches

    ani = FuncAnimation(
        fig,
        update,
        frames=len(path),
        interval=100,
        repeat=False,
    )
    plt.show()

    # logger.info(f"Saving vid to {DEMO_VID_FILEPATH}")
    # ani.save(DEMO_VID_FILEPATH)


tasks = [
    ((1, -1.5, 0.0), (3, -3.5, -np.pi / 2)),
    ((10, -9, -np.pi / 2), (2, -9, np.pi / 2)),
]
