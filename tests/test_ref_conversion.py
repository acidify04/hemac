""""Test the conversion from world red (ENU) to pyagme ref."""
import pygame


# Set up display
window_width, window_height = 800, 600

# Define the game area (Pygame.Rect)
game_area = pygame.Rect(0, 0, window_width, window_height)


# Define the function
def world_ref_to_game_ref(world_vec, game_area: pygame.Rect):
    """Convert a vector (pos, vel or accel), from the world referential (ENU) to Pygame display referential."""
    height = game_area.height
    game_ref = [world_vec[0], height - world_vec[1]]
    return game_ref


# Example vectors to test
world_vectors = [
    [100, 100],  # Bottom left in world coordinates
    [400, 300],  # Center in world coordinates
    [700, 500],  # Near top right in world coordinates
    [200, 450],  # Somewhere in the world coordinates
]

# Convert world vectors to game reference
game_vectors = [world_ref_to_game_ref(vec, game_area) for vec in world_vectors]
assert game_vectors[0][1] == window_height - world_vectors[0][1]
