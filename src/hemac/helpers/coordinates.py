"""Coordinate conversion module."""
import geopy
from geopy import distance
from pymap3d import geodetic2enu, enu2geodetic


def flat2lla(flat_earth_pos: list, lat_long_origin: list) -> list:
    """ENU to LLA conversion.

    Args:
    ----
        flat_earth_pos (list): Flat earth pos (ENU)
        lat_long_origin (list): LL at origin

    Returns:
    -------
        list: LLA coordinates

    """
    return list(enu2geodetic(flat_earth_pos[0], flat_earth_pos[1], flat_earth_pos[2],
                             lat_long_origin[0], lat_long_origin[1], 0))


def lla2flat(lla: list, lat_long_origin: list) -> list:
    """LLA to ENU conversion.

    Args:
    ----
        lla (list): LLA to convert.
        lat_long_origin (list): LLA origin

    Returns:
    -------
        list: ENU Coordinates.

    """
    return list(geodetic2enu(lla[0], lla[1], lla[2], lat_long_origin[0], lat_long_origin[1], 0))


def geodesic_distance_in_m_between_two_positions(lat_1: float, lon_1: float, alt_1: float, lat_2: float,
                                                 lon_2: float, alt_2: float) -> float:
    """Get the distance between two waypoints (LLA).

    - First step, compute horizontal distance between two waypoints (lat, long) to (lat, long).
        We get the geodesic distance.
    - Second step, compute the absolute value of the distance between altitude of waypoints.
    - These two values represent the cathetus of the rectangle.
      And we use the Pythagorean theorem to compute the hypotenuse (our distance)
    """
    distance_horizontal = distance.distance(geopy.Point(lat_1, lon_1), geopy.Point(lat_2, lon_2)).meters

    return (distance_horizontal ** 2 + (alt_1 - alt_2) ** 2) ** 0.5
