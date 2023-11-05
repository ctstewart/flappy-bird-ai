import pygame
import random

# Initialize Pygame
pygame.init()

# Game Variables
screen_width = 400
screen_height = 600
bird_x = 50
bird_y = 300
bird_radius = 20
bird_movement = 0
gravity = 0.25
jump_height = 5
pipe_width = 70
pipe_gap = 200
pipe_frequency = 1500  # milliseconds
last_pipe = pygame.time.get_ticks() - pipe_frequency
score = 0
pipes = []
passed_pipes = []

# Set up the game window
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Flappy Bird")

# Main Game Loop
running = True
while running:
    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                bird_movement = -jump_height

    # Bird movement
    bird_movement += gravity
    bird_y += bird_movement

    # Create new pipes
    time_now = pygame.time.get_ticks()
    if time_now - last_pipe > pipe_frequency:
        pipe_height_random = random.randint(-100, 100)
        top_pipe_height = pipe_height_random + pipe_gap // 2
        bottom_pipe_height = screen_height - top_pipe_height - pipe_gap
        new_pipe_top = pygame.Rect(screen_width, 0, pipe_width, top_pipe_height)
        new_pipe_bottom = pygame.Rect(screen_width, screen_height - bottom_pipe_height, pipe_width, bottom_pipe_height)
        pipes.append((new_pipe_top, new_pipe_bottom))
        last_pipe = time_now

    # Move pipes and check for score
    for pipe_pair in pipes:
        pipe_top, pipe_bottom = pipe_pair
        pipe_top.x -= 5
        pipe_bottom.x -= 5

        # Check if the bird has passed the pipe
        if bird_x > pipe_top.right and pipe_pair not in passed_pipes:
            score += 1
            passed_pipes.append(pipe_pair)

    # Remove off-screen pipes
    pipes = [pipe_pair for pipe_pair in pipes if pipe_pair[0].right > 0]

    # Collision detection
    bird_rect = pygame.Rect(bird_x - bird_radius, bird_y - bird_radius, bird_radius * 2, bird_radius * 2)
    for pipe_top, pipe_bottom in pipes:
        if bird_rect.colliderect(pipe_top) or bird_rect.colliderect(pipe_bottom):
            running = False  # End the game

    if bird_y > screen_height - bird_radius or bird_y < bird_radius:
        running = False  # Bird hits the ground or goes off screen

    # Draw everything
    screen.fill((0, 0, 0))  # Fill screen with black
    for pipe_top, pipe_bottom in pipes:
        pygame.draw.rect(screen, (0, 128, 0), pipe_top)  # Draw top pipe
        pygame.draw.rect(screen, (0, 128, 0), pipe_bottom)  # Draw bottom pipe
    pygame.draw.circle(screen, (0, 0, 255), (bird_x, bird_y), bird_radius)  # Draw bird

    # Display score
    font = pygame.font.SysFont(None, 36)
    score_surface = font.render(f'Score: {score}', True, (255, 255, 255))
    screen.blit(score_surface, (10, 10))

    # Update the display
    pygame.display.update()

    # Cap the frame rate
    pygame.time.Clock().tick(30)

pygame.quit()
