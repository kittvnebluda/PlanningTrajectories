import numpy as np
from matplotlib import pyplot as plt

from ..trajectory import (
    interpolate_bsplines,
    interpolate_c0,
    interpolate_c1,
    interpolate_c2,
    interpolate_cubic_parabola,
)
from ..trajectory import TrajParams
from ..trajectories import TRAJECTORIES


def show_trajectory(interp_func, pts):
    wps = np.array(pts, dtype=np.float64)
    params = TrajParams(
        resolution=0.01, smooth_radius=1, curvature_gain=0.1, bspline_degree=2
    )
    traj = interp_func(wps, params)
    plt.scatter(traj[:, 0], traj[:, 1], s=2)
    plt.scatter(wps[:, 0], wps[:, 1], c="red", s=4)
    plt.show()


def show_cubic_parabola(p_entry, p_exit):
    pts = interpolate_cubic_parabola(p_exit - p_entry, 1, 0.1, 0) + p_entry
    plt.scatter(pts[:, 0], pts[:, 1])
    plt.show()


interp_funcs = {
    "c0": interpolate_c0,
    "c1": interpolate_c1,
    "c2": interpolate_c2,
    "bspline": interpolate_bsplines,
}
if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    p1 = subparsers.add_parser("traj", help="Visualize trajectory")
    p1.add_argument("continuity_class", choices=interp_funcs.keys())
    p1.add_argument("pts", choices=TRAJECTORIES.keys())
    p1.set_defaults(
        factory=lambda args: show_trajectory(
            interp_funcs[args.continuity_class], TRAJECTORIES[args.pts].waypoints
        )
    )

    p2 = subparsers.add_parser("cubic_parabola", help="Visualize cubic parabola")
    p2.add_argument("p1x", type=float)
    p2.add_argument("p1y", type=float)
    p2.add_argument("p2x", type=float)
    p2.add_argument("p2y", type=float)
    p2.set_defaults(
        factory=lambda args: show_cubic_parabola(
            np.array([args.p1x, args.p1y]),
            np.array([args.p2x, args.p2y]),
        )
    )

    args = parser.parse_args()
    args.factory(args)
