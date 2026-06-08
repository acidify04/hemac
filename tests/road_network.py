import pygame
import sys

# Initialize Pygame
pygame.init()

# Screen settings
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Simple Road Network")

# Colors
WHITE = (255, 255, 255)
GRAY = (50, 50, 50)
YELLOW = (255, 204, 0)
RED = (200, 0, 0)

# Road network data
nodes = {
    1: (100, 300),
    2: (300, 300),
    3: (500, 200),
    4: (500, 400),
    5: (700, 300)
}

# Each edge connects two nodes
edges = [
    (1, 2),
    (2, 3),
    (2, 4),
    (3, 5),
    (4, 5)
]

# Road properties
ROAD_WIDTH = 20

def draw_road(start, end):
    # Draw the road as a thick gray line
    pygame.draw.line(screen, GRAY, start, end, ROAD_WIDTH)

    # Add dashed center line (yellow)
    dash_length = 15
    total_length = ((end[0] - start[0])**2 + (end[1] - start[1])**2)**0.5
    num_dashes = int(total_length // (dash_length * 2))

    for i in range(num_dashes):
        # Calculate dash positions
        t_start = i * 2 * dash_length / total_length
        t_end = (i * 2 + 1) * dash_length / total_length

        dash_start = (
            start[0] + (end[0] - start[0]) * t_start,
            start[1] + (end[1] - start[1]) * t_start
        )
        dash_end = (
            start[0] + (end[0] - start[0]) * t_end,
            start[1] + (end[1] - start[1]) * t_end
        )

        pygame.draw.line(screen, YELLOW, dash_start, dash_end, 2)

def draw_nodes():
    for pos in nodes.values():
        pygame.draw.circle(screen, RED, pos, 6)

# Main loop
clock = pygame.time.Clock()
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    screen.fill(WHITE)

    # Draw roads
    for start_id, end_id in edges:
        start = nodes[start_id]
        end = nodes[end_id]
        draw_road(start, end)

    # Draw intersections (nodes)
    draw_nodes()

    pygame.display.flip()
    clock.tick(60)
