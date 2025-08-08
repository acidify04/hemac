"""HeMARL module."""

from dataclasses import dataclass, asdict


@dataclass
class DroneConfig:
    """Drone configuration class."""

    drone_max_charge: int = 9999
    drone_max_speed: int = 14
    drone_max_thrust: int = 5
    drone_altitude: int = 30
    drone_dimension: list = None
    drone_ui_dimension: int = 40
    drones_starting_pos: list = None
    starting_pos_coordinates_type: str = "geo"  # could be geo or cardinal
    position_origin: dict = None  # ex: {latitude: 45.67410702057657, longitude: -73.14298937620299}

    dict = asdict


@dataclass
class GeofenceConfig:
    """Geofence configuration class."""

    position_origin: dict = None  # ex: {latitude: 45.67410702057657, longitude: -73.14298937620299}
    coordinates_type: str = "geo"  # could be geo or cardinal
    area: list = None

    dict = asdict


@dataclass
class PatrolConfig:
    """Patrol configuration class."""

    position_origin: dict = None  # ex: {latitude: 45.67410702057657, longitude: -73.14298937620299}
    coordinates_type: str = "geo"  # could be geo or cardinal
    area: list = None
    benchmark: bool = True

    dict = asdict


@dataclass
class PoiConfig:
    """POI configuration class."""

    starting_pos: list = None  # ex: [45.67410702057657, -73.14298937620299] for lat, lon
    starting_pos_coordinates_type: str = "geo"  # could be geo or cardinal
    position_origin: dict = None  # ex: {latitude: 45.67410702057657, longitude: -73.14298937620299}
    spawn_mode: str = "fixed"  # could be fixed or random
    dimension: list = None  # [10, 12] width x height
    speed: int = 10
    variable_speed: bool = True
    draw_expected_position: bool = False
    draw_uncertainty: bool = True
    waypoints: list = None  # ex: [[45.67, -73.14], [[45.68, -73.15]] for lat, lon
    waypoints_coordinates_type: str = "geo"  # could be geo or cardinal

    dict = asdict


@dataclass
class HeMACConfig:
    """HeMAC configuration class."""

    time_factor: int = 1
    area_size: tuple = (1000, 1000)
    max_cycles: int = 300
    render_mode: str | None = None
    render_ratio: int = 1
    render_fps: int = 15
    observer_speed: int = 18
    n_observers: int = 0
    n_drones: int = 2
    min_obstacles: int = 5
    max_obstacles: int = 6
    known_goals: bool = False
    drone_config: DroneConfig = None
    geofence_config: GeofenceConfig = None
    patrol_config: PatrolConfig = None
    poi_config: list[PoiConfig] = None
    drone_sensor: dict | None = None
    observer_sensor: dict | None = None

    dict = asdict
