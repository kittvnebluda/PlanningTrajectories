from abc import ABC, abstractmethod
from functools import partial
from math import acos, atan2, cos, pi, sin, sqrt, tan, isclose
from typing import NamedTuple

from pygame import SRCALPHA, Color, Surface, Vector2

from platra.exceptions import BadCubicParabolaParameter
from platra.types import Number

from .utils import (
    draw_arc,
    draw_cubic_parabola,
    draw_line,
    draw_pts,
    signed_dist_to_line,
)

CORNER_GEN_K = 0.5


def get_max_corner_radius(v1_len: Number, sigma_half: Number) -> float:
    r = v1_len * abs(tan(sigma_half))
    return r / 2 * 0.9


def get_min_k(r: Number) -> float:
    return 18 / (25 * r**2 * 5 ** (1 / 2)) * 1.5


def check_corner_radius(radius: Number, k: Number) -> None:
    radius_min = 3 * sqrt(2) / (5 * sqrt(k) * 5 ** (1 / 4))
    if radius < radius_min:
        raise BadCubicParabolaParameter(k, radius, radius_min)


def cubic_parab_pt(t: float, k: float, angle: float, bias: Vector2) -> Vector2:
    return Vector2(t, k * t**3).rotate_rad(angle) + bias


class DrawOpts(NamedTuple):
    size: int = 2
    draw_waypoints: bool = False
    dashed: bool = False
    dash_len: float = 0.05
    gap_len: float = 0.05


class Trajectory(ABC):
    def __init__(self, waypoints: list[Vector2]):
        self._waypoint_id: int = 0
        self._waypoints: list[Vector2] = waypoints

    @abstractmethod
    def get_target(self, pos: Vector2) -> Vector2:
        pass

    @abstractmethod
    def draw_trajectory(self, surface: Surface, opts: DrawOpts) -> None:
        pass


class TrajectoryC0(Trajectory):
    def __init__(self, waypoints: list[Vector2]):
        super().__init__(waypoints)
        self._L = 0.1
        self._target: Vector2 = Vector2(0, 0)

    @property
    def waypoints(self) -> list[Vector2]:
        return self._waypoints

    @waypoints.setter
    def waypoints(self, new_waypoints: list[Vector2]):
        self._waypoints = new_waypoints
        self._waypoint_id = 0

    def get_target(self, pos: Vector2) -> Vector2:
        if len(self._waypoints) > self._waypoint_id + 1:
            a = self._waypoints[self._waypoint_id]
            b = self._waypoints[self._waypoint_id + 1]
            v = b - a
            projection = (pos - a).project(v) + a
            self._target = projection + v.normalize() * self._L
            phi = atan2(v.y, v.x)
            q = signed_dist_to_line(self._target, a, phi, -v.magnitude())
            if q > 0:
                self._target = b
                self._waypoint_id += 1

        return self._target

    def draw_trajectory(self, surface: Surface, opts: DrawOpts = DrawOpts()):
        """
        Draw the trajectory on a given Pygame surface.

        Args:
            surface (pygame.Surface):
                The surface to draw the trajectory on.
            opts (DrawOpts):
                Options for drawing.
        """
        if opts.draw_waypoints:
            draw_pts(surface, self._waypoints, size=opts.size)
        for i in range(len(self._waypoints) - 1):
            a = self._waypoints[i]
            b = self._waypoints[i + 1]
            draw_line(
                surface,
                a,
                b,
                "blue",
                opts.size,
                opts.dash_len,
                opts.gap_len,
                opts.dashed,
            )


class TrajectoryC1(Trajectory):
    def __init__(self, waypoints: list[Vector2]):
        super().__init__(waypoints)
        self._dist_to_virtual_target = 0.1
        self._virtual_target: Vector2 = Vector2(0, 0)
        self._trajectory_surface: Surface | None = None

    @property
    def waypoints(self) -> list[Vector2]:
        return self._waypoints

    @waypoints.setter
    def waypoints(self, new_waypoints: list[Vector2]):
        self._waypoints = new_waypoints
        self._waypoint_id = 0
        self._trajectory_surface = None

    def get_target(self, pos: Vector2) -> Vector2:
        waypoints_num = len(self._waypoints)
        if waypoints_num > self._waypoint_id + 2:
            a = self._waypoints[self._waypoint_id]
            b = self._waypoints[self._waypoint_id + 1]
            c = self._waypoints[self._waypoint_id + 2]
            v1 = b - a
            v2 = c - b
            projection = (pos - a).project(v1) + a
            self._virtual_target = (
                projection + v1.normalize() * self._dist_to_virtual_target
            )

            # Handling straight line
            phi1 = atan2(v1.y, v1.x)
            phi2 = atan2(v2.y, v2.x)
            phi_div = phi2 - phi1
            # Angle between v1 and v2
            sigma = pi - phi_div if phi_div > 0 else -pi - phi_div
            sigma_half = sigma / 2

            R = get_max_corner_radius(v1.magnitude(), sigma_half)
            d = abs(R / tan(sigma_half))
            q = signed_dist_to_line(self._virtual_target, a, phi1, d - v1.magnitude())
            if q <= 0:
                return self._virtual_target

            # Handling arc
            dc = abs(R / sin(sigma_half))
            cc = Vector2(dc, 0).rotate_rad(phi1 + pi - sigma_half) + b  # Circle center
            sign = 1 if sigma > 0 else -1
            theta = atan2(pos.y - cc.y, pos.x - cc.x)
            arc_step = self._dist_to_virtual_target / R

            theta_target = theta + sign * arc_step
            self._virtual_target = Vector2(
                cc.x + R * cos(theta_target),
                cc.y + R * sin(theta_target),
            )
            q = signed_dist_to_line(self._virtual_target, b, phi2, -d)
            if q > 0:
                self._waypoint_id += 1
        elif waypoints_num == self._waypoint_id + 2:
            a = self._waypoints[self._waypoint_id]
            b = self._waypoints[self._waypoint_id + 1]
            v = b - a
            projection = (pos - a).project(v) + a
            self._virtual_target = (
                projection + v.normalize() * self._dist_to_virtual_target
            )
            phi = atan2(v.y, v.x)
            q = signed_dist_to_line(self._virtual_target, a, phi, -v.magnitude())
            if q > 0:
                self._virtual_target = b
                self._waypoint_id += 1

        return self._virtual_target

    def _build_trajectory(self, surface_size, color, opts: DrawOpts) -> Surface:
        surface = Surface(surface_size, SRCALPHA)
        if opts.draw_waypoints:
            draw_pts(surface, self._waypoints, color=color, size=opts.size)
        if len(self._waypoints) < 3:
            return surface

        line_displacement = Vector2(0, 0)
        for i in range(len(self._waypoints) - 2):
            a = self._waypoints[i]
            b = self._waypoints[i + 1]
            c = self._waypoints[i + 2]
            v1 = b - a
            v2 = c - b
            dir = v1.normalize()
            phi1 = atan2(v1.y, v1.x)
            phi2 = atan2(v2.y, v2.x)
            phi_div = phi2 - phi1
            sigma = pi - phi_div if phi_div > 0 else -pi - phi_div
            sigma_half = sigma / 2
            sign = 1 if sigma > 0 else -1  # Direction of the turn
            R = get_max_corner_radius(v1.magnitude(), sigma_half)
            d = 0 if isclose(abs(sigma), pi) else abs(R / tan(sigma_half))

            # Handling straight line
            v1_end = b - dir * d  # Vector from point A to the start of the arc
            draw_line(
                surface,
                a + line_displacement,
                v1_end,
                color,
                opts.size,
                opts.dash_len,
                opts.gap_len,
                opts.dashed,
            )

            # Skipping arc if 180 degree angle
            if not d:
                line_displacement = Vector2(0, 0)
                continue

            # Handling arc
            dc = abs(R / sin(sigma_half))
            cc = Vector2(dc, 0).rotate_rad(phi1 + pi - sigma_half) + b  # Circle center
            angle_step = opts.dash_len / R
            q = partial(signed_dist_to_line, line_point=b, angle=phi2, bias=-d)
            arc_end = draw_arc(
                surface,
                cc,
                sign,
                angle_step,
                v1_end,
                q,
                Color(0, 200, 255),
                2,
                opts.dashed,
            )
            line_displacement = arc_end - b

        # Last straight line
        a = self._waypoints[-2]
        b = self._waypoints[-1]
        draw_line(
            surface,
            a + line_displacement,
            b,
            color,
            opts.size,
            opts.dash_len,
            opts.gap_len,
            opts.dashed,
        )

        return surface

    def draw_trajectory(self, surface: Surface, opts: DrawOpts = DrawOpts()) -> None:
        """
        Draw the trajectory on a given Pygame surface.

        Args:
            surface (pygame.Surface):
                The surface to draw the trajectory on.
            opts (DrawOpts):
                Options for drawing.
        """
        if self._trajectory_surface is None:
            self._trajectory_surface = self._build_trajectory(
                surface.get_size(),
                "blue",
                opts,
            )
        surface.blit(self._trajectory_surface, (0, 0))


class TrajectoryC2(Trajectory):
    def __init__(
        self,
        waypoints: list[Vector2],
    ):
        super().__init__(waypoints)
        self._dist_to_virtual_target = 0.1
        self._virtual_target: Vector2 = Vector2(0, 0)
        self._trajectory_surface: Surface | None = None
        self._t = 0.0
        self.cubic_parab_coef = 10.0

    @property
    def waypoints(self) -> list[Vector2]:
        return self._waypoints

    @waypoints.setter
    def waypoints(self, new_waypoints: list[Vector2]):
        self._waypoints = new_waypoints
        self._waypoint_id = 0
        self._trajectory_surface = None

    def get_target(self, pos: Vector2) -> Vector2:
        waypoints_num = len(self._waypoints)
        if waypoints_num > self._waypoint_id + 2:
            a = self._waypoints[self._waypoint_id]
            b = self._waypoints[self._waypoint_id + 1]
            c = self._waypoints[self._waypoint_id + 2]
            v1 = b - a
            v2 = c - b

            # Handling straight line
            phi1 = atan2(v1.y, v1.x)
            phi2 = atan2(v2.y, v2.x)
            phi_div = phi2 - phi1
            # Angle between v1 and v2
            sigma = pi - phi_div if phi_div > 0 else -pi - phi_div
            sigma_half = sigma / 2
            sign = 1 if sigma > 0 else -1

            R = get_max_corner_radius(v1.magnitude(), sigma_half)
            k = get_min_k(R)
            d = abs(R / tan(sigma_half))
            q = signed_dist_to_line(pos, a, phi1, d - v1.magnitude())
            if q <= 0:
                projection = (pos - a).project(v1) + a
                self._virtual_target = (
                    projection + v1.normalize() * self._dist_to_virtual_target
                )
                return self._virtual_target

            # Cubic parabola 1
            v1_end = b - v1.normalize() * d
            t1 = (
                Vector2(1 / (6 * k * R), sign * k / (6 * k * R) ** 3).rotate_rad(phi1)
                + v1_end
            )  # Cubic parabola 1 end
            t2 = (
                Vector2(1 / (6 * k * R), -sign * k / (6 * k * R) ** 3).rotate_rad(
                    phi2 - pi
                )
                + b
                + v2.normalize() * d
            )  # Cubic parabola 2 start
            h = (t2 - t1).magnitude()
            beta = acos(1 - h**2 / (2 * R**2))
            # FIX: Those are not correct alpha1, alpha2
            alpha1 = phi1
            alpha2 = phi2

            q = signed_dist_to_line(self._virtual_target, t1, alpha1, 0)
            if q <= 0:
                next_t = self._t + 0.01
                next_virtual_target = cubic_parab_pt(next_t, k, phi1, v1_end)
                targets_dist = (next_virtual_target - self._virtual_target).magnitude()
                if targets_dist < self._dist_to_virtual_target:
                    self._t = next_t

                self._virtual_target = cubic_parab_pt(self._t, k, phi1, v1_end)
                return self._virtual_target

            # Handling arc
            q = signed_dist_to_line(self._virtual_target, t2, alpha2, 0)
            if q <= 0:
                dc = abs(R / sin(sigma_half))
                cc = (
                    Vector2(dc, 0).rotate_rad(phi1 + pi - sigma_half) + b
                )  # Circle center
                theta = atan2(pos.y - cc.y, pos.x - cc.x)
                arc_step = self._dist_to_virtual_target / R

                theta_target = theta + sign * arc_step
                self._virtual_target = Vector2(
                    cc.x + R * cos(theta_target),
                    cc.y + R * sin(theta_target),
                )
                return self._virtual_target

            # Cubic parabola 2
            v2_start = b + v2.normalize() * d
            ang = phi2 - pi
            q = signed_dist_to_line(self._virtual_target, b, phi2, -d)
            if q <= 0:
                next_t = self._t - 0.01
                next_virtual_target = cubic_parab_pt(next_t, k, ang, v2_start)
                targets_dist = (next_virtual_target - self._virtual_target).magnitude()
                if targets_dist < self._dist_to_virtual_target:
                    self._t = next_t

                self._virtual_target = cubic_parab_pt(self._t, k, ang, v2_start)
                return self._virtual_target

            self._t = 0.0
            self._waypoint_id += 1

        elif waypoints_num == self._waypoint_id + 2:
            a = self._waypoints[self._waypoint_id]
            b = self._waypoints[self._waypoint_id + 1]
            v = b - a
            projection = (pos - a).project(v) + a
            self._virtual_target = (
                projection + v.normalize() * self._dist_to_virtual_target
            )
            phi = atan2(v.y, v.x)
            q = signed_dist_to_line(self._virtual_target, a, phi, -v.magnitude())
            if q > 0:
                self._virtual_target = b
                self._waypoint_id += 1

        return self._virtual_target

    def _build_trajectory(self, surface_size, color, opts: DrawOpts) -> Surface:
        surface = Surface(surface_size, SRCALPHA)
        if opts.draw_waypoints:
            draw_pts(surface, self._waypoints, color=color, size=opts.size)
        if len(self._waypoints) < 3:
            return surface

        line_displacement = Vector2(0, 0)
        for i in range(len(self._waypoints) - 2):
            a = self._waypoints[i]
            b = self._waypoints[i + 1]
            c = self._waypoints[i + 2]
            v1 = b - a
            v2 = c - b
            dir1 = v1.normalize()
            dir2 = v2.normalize()
            phi1 = atan2(v1.y, v1.x)
            phi2 = atan2(v2.y, v2.x)
            phi_div = phi2 - phi1
            sigma = pi - phi_div if phi_div > 0 else -pi - phi_div
            sigma_half = sigma / 2
            sign = 1 if sigma > 0 else -1  # Direction of the turn
            R = get_max_corner_radius(v1.magnitude(), sigma_half)
            min_k = get_min_k(R)
            k = self.cubic_parab_coef if self.cubic_parab_coef > min_k else min_k
            d = 0 if isclose(abs(sigma), pi) else abs(R / tan(sigma_half))

            # Handling straight line
            line_end = b - dir1 * d
            line_start = a + line_displacement
            if (line_end - line_start).magnitude() > 0:
                draw_line(
                    surface,
                    line_start,
                    line_end,
                    color,
                    opts.size,
                    opts.dash_len,
                    opts.gap_len,
                    opts.dashed,
                )

            if not d:
                line_displacement = Vector2(0, 0)
                continue

            # Cubic parabola 1
            t1 = (
                Vector2(1 / (6 * k * R), sign * k / (6 * k * R) ** 3).rotate_rad(phi1)
                + line_end
            )  # Cubic parabola 1 end
            t2 = (
                Vector2(1 / (6 * k * R), -sign * k / (6 * k * R) ** 3).rotate_rad(
                    phi2 - pi
                )
                + b
                + dir2 * d
            )  # Cubic parabola 2 start
            h = (t2 - t1).magnitude()
            beta = acos(1 - h**2 / (2 * R**2))
            # FIX: Those are not correct alpha1, alpha2
            alpha1 = phi1
            alpha2 = phi2

            q = partial(signed_dist_to_line, line_point=t1, angle=alpha1, bias=0)
            draw_cubic_parabola(
                surface,
                k,
                phi1,
                line_end,
                sign,
                q,
            )

            # Handling arc
            dc1 = sqrt(R**2 - (h / 2) ** 2)
            dc2 = sqrt((b - t2).magnitude_squared() - (h / 2) ** 2)
            dc = dc1 + dc2
            cc = Vector2(dc, 0).rotate_rad(phi1 + pi - sigma_half) + b  # Circle center
            angle_step = opts.dash_len / R
            q = partial(signed_dist_to_line, line_point=t2, angle=alpha2, bias=0)
            draw_arc(
                surface,
                cc,
                sign,
                angle_step,
                t1,
                q,
                Color(0, 200, 255),
                2,
                dashed=opts.dashed,
            )

            line_displacement = dir2 * d

            # Cubic parabola 2
            q = partial(signed_dist_to_line, line_point=t2, angle=alpha2 + pi, bias=0)
            draw_cubic_parabola(
                surface,
                k,
                phi2 - pi,
                b + line_displacement,
                -sign,
                q,
            )

        # Last straight line
        a = self._waypoints[-2]
        b = self._waypoints[-1]
        draw_line(
            surface,
            a + line_displacement,
            b,
            color,
            opts.size,
            opts.dash_len,
            opts.gap_len,
            opts.dashed,
        )

        return surface

    def draw_trajectory(self, surface: Surface, opts: DrawOpts = DrawOpts()) -> None:
        """
        Draw the trajectory on a given Pygame surface.

        Args:
            surface (pygame.Surface):
                The surface to draw the trajectory on.
            opts (DrawOpts):
                Options for drawing.
        """
        if self._trajectory_surface is None:
            self._trajectory_surface = self._build_trajectory(
                surface.get_size(), "blue", opts
            )
        surface.blit(self._trajectory_surface, (0, 0))
