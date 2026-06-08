"""potential fields implementation as a baseline for the quadcopters strategy."""
import numpy as np


def potential_fields(robot_position, goal_position, obstacles_rect):
    """Compute the velocity command for the robot using the potential fields method.

    Args.
    ----------
    robot_position : numpy.ndarray. The current position of the robot (2D vector).
    goal_position : numpy.ndarray. The goal position (2D vector).
    obstacles_rect : list of numpy.ndarray. A list of positions of obstacles (each a 2D vector).

    Returns
    -------
    numpy.ndarray: The velocity command (2D vector).

    """
    # Potential fields parameters
    robot_radius = 20
    attractive_coefficient = 3
    repulsive_coefficient = 20.0
    repulsive_threshold = 30.0

    def closest_point_in_rect(rect, point):
        """Find the distance to the closest point in a rectangle to a given point.

        Args.
        ----------
        rect (pygame.Rect): The rectangle.
        point (tuple): The point (x, y).

        Returns
        -------
        tuple: The closest point (x, y) in the rectangle to the given point,
        or the closest point on the boundary of the rect if the point is inside it.

        """
        left = rect.left
        right = rect.right
        top = rect.top
        bottom = rect.bottom

        if rect.collidepoint(point[0], point[1]):
            # Point is inside the rectangle
            px = point[0]
            py = point[1]
            distance_left = (px - left, (left, py))
            distance_right = (right - px, (right, py))
            distance_top = (py - top, (px, top))
            distance_bottom = (bottom - py, (px, bottom))
            # Get the minimum distance and corresponding point
            min_distance, closest_point = min(
                distance_left, distance_right, distance_top, distance_bottom, key=lambda x: x[0]
            )
            to_closest_point = np.array([closest_point[0] - point[0], closest_point[1] - point[1]])
        else:
            closest_x = max(rect.left, min(point[0], rect.right))
            closest_y = max(rect.top, min(point[1], rect.bottom))

            to_closest_point = np.array([closest_x - point[0], closest_y - point[1]])

        return to_closest_point

    # Calculate attractive force
    def attractive_force(position, goal):
        return np.clip([attractive_coefficient * goal[0], attractive_coefficient * goal[1]], -10, 10, dtype=np.float32)

    # Calculate repulsive force #TODO implement relative measurements in the environment!
    def repulsive_force(position, obstacle):
        to_closest_point = closest_point_in_rect(obstacle, position)
        distance = max(np.linalg.norm(to_closest_point) - robot_radius, 1.0)
        # print(f"distance to closest point: {distance}")
        if distance < repulsive_threshold:
            return np.array(
                repulsive_coefficient * (1 / distance - 1 / repulsive_threshold) * (to_closest_point / distance),
                dtype=np.float32,
            )
        else:
            return np.array([0.0, 0.0], dtype=np.float32)

    # Calculate total force
    force = attractive_force(robot_position, goal_position)
    # print(f"attractive force: {force}")
    for obstacle in obstacles_rect:
        force -= repulsive_force(robot_position, obstacle)

    action = np.clip(np.array(force, dtype=np.float32), -10, 10)
    return action
