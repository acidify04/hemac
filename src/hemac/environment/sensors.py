"""Sensor modules."""

import abc
from dataclasses import dataclass
from numpy import pi, tan, arctan2, cos, sin, sqrt
from pygame.draw import line, arc, circle
from pygame import Rect
from shapely.geometry.polygon import Polygon
from shapely.geometry import Point


class Sensor:
    """Instanciate sensor."""

    @abc.abstractmethod
    def draw_sensor(self, screen):
        """Draw sensor on a screen."""
        raise NotImplementedError("Function not implemented.")

    @abc.abstractmethod
    def update_poly_points(self, pos: tuple[float], theta: float, distance_from_ground: float) -> tuple:
        """Return polygon points of the sensor.

        Args:
        ----
            pos (tuple[float]): Current position.
            theta (float): Current angle vs origin.
            distance_from_ground (float): Distance from ground.

        Returns:
        -------
            tuple: Polygon points.

        """
        raise NotImplementedError("Function not implemented.")

    @abc.abstractmethod
    def is_point_detected(self, target_pos: tuple) -> bool:
        """Check if point is detected.

        Args:
        ----
            target_pos (tuple): Target position.

        Returns:
        -------
            bool: is point detected.

        """
        raise NotImplementedError("Function not implemented.")


@dataclass
class DownwardFacingCamera(Sensor):
    """Instanciate top camera."""

    hfov: float
    vfov: float
    depth_precision: int = 20
    angular_precision: float = 6
    vertice_pos: tuple | None = None

    def width(self, distance_from_ground):
        """Return width of camera area."""
        return tan(self.vfov / 2) * distance_from_ground * 2

    def height(self, distance_from_ground):
        """Return height of camera area."""
        return tan(self.hfov / 2) * distance_from_ground * 2

    def is_point_detected(self, target_pos: tuple) -> bool:
        """Check if point is detected.

        Args:
        ----
            target_pos (tuple): Target position.

        Returns:
        -------
            bool: is point detected.

        """
        polygon = Polygon(
            [
                (self.vertice_pos[0][0], self.vertice_pos[0][1]),
                (self.vertice_pos[1][0], self.vertice_pos[1][1]),
                (self.vertice_pos[2][0], self.vertice_pos[2][1]),
                (self.vertice_pos[3][0], self.vertice_pos[3][1]),
            ]
        )
        if polygon.contains(Point(target_pos[0], target_pos[1])):
            return True
        else:
            return False

    def draw_sensor(self, screen):
        """Draw sensor on a screen."""
        line(screen, (222, 200, 20), self.vertice_pos[0], self.vertice_pos[1])
        line(screen, (222, 200, 20), self.vertice_pos[1], self.vertice_pos[2])
        line(screen, (222, 200, 20), self.vertice_pos[2], self.vertice_pos[3])
        line(screen, (222, 200, 20), self.vertice_pos[3], self.vertice_pos[0])

    def update_poly_points(self, pos: tuple[float], theta: float, distance_from_ground: float) -> tuple:
        """Return polygon points of the sensor.

        Args:
        ----
            pos (tuple[float]): Current position.
            theta (float): Current angle vs origin.
            distance_from_ground (float): Distance from ground.

        Returns:
        -------
            tuple: Polygon points.

        """
        theta = -theta
        width = self.width(distance_from_ground)
        height = self.height(distance_from_ground)
        alpha = arctan2(height / 2, width / 2)
        beta = pi / 2 - alpha
        e = ((height / 2) ** 2 + (width / 2) ** 2) ** 0.5
        a = (cos(alpha + theta) * e, sin(alpha + theta) * e)
        b = (cos(beta + theta + pi / 2) * e, sin(beta + theta + pi / 2) * e)
        c = (cos(alpha + theta + pi) * e, sin(alpha + theta + pi) * e)
        d = (cos(beta + theta + 3 * pi / 2) * e, sin(beta + theta + 3 * pi / 2) * e)

        rounding = 6
        a = (round(a[0] + pos[0], rounding), round(a[1] + pos[1], rounding))
        b = (round(b[0] + pos[0], rounding), round(b[1] + pos[1], rounding))
        c = (round(c[0] + pos[0], rounding), round(c[1] + pos[1], rounding))
        d = (round(d[0] + pos[0], rounding), round(d[1] + pos[1], rounding))

        self.vertice_pos = (a, b, c, d)

        return self.vertice_pos


@dataclass
class ForwardFacingCamera(Sensor):
    """Instanciate Front Camera."""

    hfov: float = pi / 5
    sensing_range: int = 250
    depth_precision: int = 20
    angular_precision: float = 6
    pos: tuple | None = None
    theta: float | None = None
    vertices_pos: tuple | None = None

    def is_point_detected(self, target_pos):
        """Check if point is detected.

        Args:
        ----
            target_pos (tuple): Target position.

        Returns:
        -------
            bool: is point detected.

        """
        try:
            goal_x, goal_y = target_pos[0], target_pos[1]
            dist_to_goal = dist(self.pos[0], self.pos[1], goal_x, goal_y)
            if dist_to_goal < self.sensing_range:
                vector = [
                    goal_x - self.pos[0],
                    -goal_y + self.pos[1],
                ]  # y coordinates are inverted
                angle = arctan2(vector[1], vector[0])
                relative_angle = abs(angle - self.theta) % (2 * pi)
                relative_angle = (relative_angle + pi) % (2 * pi) - pi  # convert from [0, 2pi] to [-pi, pi]
                if abs(relative_angle) < self.hfov / 2:
                    return True
        except Exception as e:
            print(e)
        return False

    def update_poly_points(self, pos: tuple[float], theta: float, distance_from_ground: float = None) -> tuple:
        """Return polygon points of the sensor.

        Args:
        ----
            pos (tuple[float]): Current position.
            theta (float): Current angle vs origin.
            distance_from_ground (float): Distance from ground.

        Returns:
        -------
            tuple: Polygon points.

        """
        self.pos = pos
        self.theta = theta

        left_dx, left_dy = (
            self.sensing_range * cos(theta + self.hfov),
            self.sensing_range * sin(theta + self.hfov),
        )
        right_dx, right_dy = (
            self.sensing_range * cos(theta - self.hfov),
            self.sensing_range * sin(theta - self.hfov),
        )

        self.vertices_pos = (
            (pos[0], pos[1]),
            (pos[0] + left_dx, pos[1] - left_dy),
            (pos[0] + right_dx, pos[1] - right_dy),
        )

    def draw_sensor(self, screen):
        """Draw sensor on a screen."""
        line(screen, (222, 200, 20), self.vertices_pos[0], self.vertices_pos[1])
        line(screen, (222, 200, 20), self.vertices_pos[0], self.vertices_pos[2])

        self._draw_arc(
            screen,
            (222, 200, 0),
            self.vertices_pos[0],
            self.sensing_range,
            self.theta - self.hfov,
            self.theta + self.hfov,
        )

    def _draw_arc(self, surface, color, center, radius, start_angle, end_angle):
        """Draw arc."""
        rect = Rect(0, 0, radius * 2, radius * 2)
        rect.center = center
        arc(surface, color, rect, start_angle, end_angle)


@dataclass
class RoundCamera(Sensor):
    """Instanciate Front Camera."""

    sensing_range: int = 75
    depth_precision: int = 20
    angular_precision: int = 10
    pos: tuple | None = None
    theta: float | None = None
    vertices_pos: tuple | None = None

    def is_point_detected(self, target_pos):
        """Check if point is detected.

        Args:
        ----
            target_pos (tuple): Target position.

        Returns:
        -------
            bool: is point detected.

        """
        if (self.pos[0] - target_pos[0]) ** 2 + (self.pos[1] - target_pos[1]) ** 2 < self.sensing_range**2:
            return True
        else:
            return False

    def update_poly_points(self, pos: tuple[float], theta: float = None, distance_from_ground: float = None) -> tuple:
        """Return polygon points of the sensor.

        Args:
        ----
            pos (tuple[float]): Current position.
            theta (float): Current angle vs origin.
            distance_from_ground (float): Distance from ground.

        Returns:
        -------
            tuple: Polygon points.

        """
        self.pos = pos

    def draw_sensor(self, screen):
        """Draw sensor on a screen."""
        circle(screen, (0, 255, 0), (self.pos[0], self.pos[1]), self.sensing_range, width=1)


def dist(x1, y1, x2, y2):
    """Return distance between two points."""
    return sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
