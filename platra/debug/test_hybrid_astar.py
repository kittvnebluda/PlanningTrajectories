import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Polygon

from platra.core.astar.hybrid_astar import HybridAStarPlanner
from platra.core.map import Grid
from platra.core.robot.configs import AckermannConfigForStaticFeedback
from platra.core.symbolic.ackermann import LambdifiedAckermannForStaticFeedback

DEMO_VID_FILEPATH = "demo_has.mp4"


def draw_ackermann(ax, x, y, theta, L, W, color="blue", alpha=0.8):
    corners = np.array([[L, W], [L, -W], [-L, -W], [-L, W]])

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

    fx = x + np.cos(theta) * L * 0.6
    fy = y + np.sin(theta) * L * 0.6
    ax.plot([x, fx], [y, fy], color="black", linewidth=2)


def main():
    grid_map = Grid.from_name("parking_lot")

    scale = 0.2
    shift_x = 0.0
    shift_y = 0.0
    grid_params = (scale, shift_x, shift_y)

    start = (0.6, -1.5, 0.0)
    goal = (3.5, -4, np.pi / 2)

    conf = AckermannConfigForStaticFeedback.from_symbolic(
        0.35, 0.9, 0.4, 0.2, LambdifiedAckermannForStaticFeedback
    )
    planner = HybridAStarPlanner(conf)
    path = planner.plan(start, goal, grid_map, grid_params)

    xs = [n.x for n in path]
    ys = [n.y for n in path]

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

    if path is None:
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
        draw_ackermann(ax, node.x, node.y, node.theta, conf.Lf, conf.Ls, "red")

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

    print(f"Saving to {DEMO_VID_FILEPATH}")
    ani.save(DEMO_VID_FILEPATH)


if __name__ == "__main__":
    main()
