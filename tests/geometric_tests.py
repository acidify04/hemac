import pygame
import numpy as np

# Define screen size
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
BAR_WIDTH = 50
BAR_HEIGHT = SCREEN_HEIGHT // 2  # Bar height is half the screen height

# Function to map a probability to a color
def get_heatmap_color(prob, max_prob):
    """Map a probability (0 to max_prob) to an RGBA color."""
    green = int(prob / max_prob * 255)  # Green increases with probability
    red = int((1 - prob) * 255)  # Red decreases with probability
    return (red, green, 0, 100)  # Adjusted alpha for transparency in the bar

def draw_color_scale_bar(screen, max_prob):
    """Draw a vertical color scale bar representing the probability range."""
    bar_surface = pygame.Surface((BAR_WIDTH, BAR_HEIGHT), pygame.SRCALPHA)
    for i in range(BAR_HEIGHT):
        prob = (BAR_HEIGHT - i) / BAR_HEIGHT * max_prob  # Scale probability
        color = get_heatmap_color(prob, max_prob)
        pygame.draw.line(bar_surface, color, (0, i), (BAR_WIDTH, i))

    screen.blit(bar_surface, (SCREEN_WIDTH // 2 - BAR_WIDTH // 2, SCREEN_HEIGHT // 2 - BAR_HEIGHT // 2))

def draw_scale_values(screen, max_prob):
    """Draw the probability values along the color scale bar."""
    font = pygame.font.Font(None, 24)
    increments = 10  # Number of values to display
    for i in range(increments + 1):
        prob = i / increments * max_prob
        text = font.render(f'{prob:.2f}', True, (255, 255, 255))
        text_rect = text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + BAR_HEIGHT // 2 - (i * BAR_HEIGHT // increments)))
        screen.blit(text, text_rect)

# Pygame setup
pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption('Heatmap with Color Scale Bar')
clock = pygame.time.Clock()

# Main loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Fill background with dark green
    screen.fill((10, 50, 20))

    # Draw the color scale bar
    draw_color_scale_bar(screen, max_prob=0.1)

    # Draw probability scale values
    # draw_scale_values(screen, max_prob=1.0)

    # Update the display
    pygame.display.flip()
    clock.tick(30)

pygame.quit()
