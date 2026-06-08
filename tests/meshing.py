"""meshing tests."""
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point
import time

start = time.perf_counter()
# Define a simple polygon
polygon = Polygon(
    [[100, 100], [200, 100], [200, 800], [900, 800], [900, 900], [100, 900]]
)  # Create a grid of points within the bounding box of the polygon


x_min, y_min, x_max, y_max = polygon.bounds
x, y = np.meshgrid(
    np.arange(x_min, x_max, 1), np.arange(y_min, y_max, 1)
)  # Flatten the grid points and filter those within the polygon
points = np.vstack((x.flatten(), y.flatten())).T

inside_points = np.array([point for point in points if polygon.contains(Point(point))])  # Create a simple zigzag path

print(f"method 1 found inside_points in {(time.perf_counter() - start)} s")
start = time.perf_counter()

explored_grid = np.ones((round(x_max - x_min), round(y_max - y_min)))
patrol_area_indices = [[], []]  # row indices [0] and column indices [1]
for x in range(explored_grid.shape[0]):
    for y in range(explored_grid.shape[1]):
        if polygon.contains(Point(x, y)):
            patrol_area_indices[0].append(x)
            patrol_area_indices[1].append(y)
print(f"method 2 found inside_points in {(time.perf_counter() - start)} s")

path = sorted(inside_points, key=lambda p: (p[1], p[0]))  # Plot the polygon and path

x_path = [p[0] for p in path]
y_path = [p[1] for p in path]

x, y = polygon.exterior.xy

plt.plot(x, y, "b")  # Plot the coverage path


plt.plot(x_path, y_path, "ro-")
plt.show()
