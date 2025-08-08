"""Coordinates conversion test module."""
import geopy.distance
from hemac.helpers.coordinates import lla2flat, flat2lla


class TestCoordinates:
    """TestCoordinates."""

    def test_lla2flat(self):
        """Test lla2flat."""
        origin = [45.67999193490854, -73.13984955665096, 20.0]
        flat_xyz = lla2flat(origin, origin)

        assert round(flat_xyz[0], 2) == 0
        assert round(flat_xyz[1], 2) == 0
        assert round(flat_xyz[2], 2) == 20

        destination = [45.68024641565777, -73.14021260084162, 20.0]
        flat_xyz = lla2flat(origin, destination)
        distance = (flat_xyz[0] ** 2 + flat_xyz[1] ** 2) ** 0.5
        assert distance < 42 and distance > 38

    def test_flat2lla(self):
        """Test flat2lla."""
        origin = [45.67999193490854, 73.13984955665096, 20.0]

        expected_distance = 40
        x_y_offset = expected_distance / (2 ** 0.5)

        flat_destination = [x_y_offset, x_y_offset, 20.0]

        lla_destination = flat2lla(flat_destination, origin)
        lla_destination[2] = round(lla_destination[2], 3)

        assert round(geopy.distance.geodesic(origin, lla_destination).m, 1) == expected_distance


if __name__ == "__main__":
    TestCoordinates().test_flat2lla()
