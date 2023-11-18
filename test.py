import os
import pygame
import random
import numpy as np
# import tensorflow as tf
from tensorflow.keras.models import load_model


def choose_action(state, model):
    state = np.array(state).reshape(1, -1)
    q_values = model.predict(state, verbose=0)
    return np.argmax(q_values[0])


def flappy_bird(model):
    # Initialize Pygame
    pygame.init()

    # Game Variables
    screen_width = 720
    screen_height = 480
    frame = 0
    bird_x = 50
    bird_y = 300
    bird_radius = 20
    bird_movement = 0
    gravity = 0.25
    jump_height = 5
    pipe_width = 70
    pipe_gap = 200
    # pipe_frequency = 1500  # milliseconds
    pipe_frequency = 45 # frames
    # last_pipe = pygame.time.get_ticks() - pipe_frequency
    last_pipe = frame - pipe_frequency
    score = 0
    pipes = []
    passed_pipes = []
    action = 0
    # State:
    # bird's y position,
    # bird's y velocity,
    # top pipe's bottom y position,
    # bottom pipe's top y position,
    # top pipe's left x position - bird's x position,
    # next top pipe's bottom y position,
    # next bottom pipe's top y position,
    # next top pipe's left x position - bird's x position
    state = (bird_y, bird_movement, 0, 0, 0, 0, 0, 0)
    next_state = (bird_y, bird_movement, 0, 0, 0, 0, 0, 0)
    global high_score

    # Set up the game window
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Flappy Bird")

    # Main Game Loop
    running = True
    done = False
    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        state = next_state
        # Get agent decision
        # Choose action based on ε-greedy strategy
        action = choose_action(state, model)

        # If the agent says flap
        if action == 1:
            bird_movement = -jump_height

        # Bird movement
        bird_movement += gravity
        bird_y += bird_movement

        # Create new pipes
        # time_now = pygame.time.get_ticks()
        if frame - last_pipe >= pipe_frequency:
            pipe_height_random = random.randint(-100, 100)
            top_pipe_height = pipe_height_random + pipe_gap // 2
            bottom_pipe_height = screen_height - top_pipe_height - pipe_gap
            new_pipe_top = pygame.Rect(screen_width, 0, pipe_width, top_pipe_height)
            new_pipe_bottom = pygame.Rect(screen_width, screen_height - bottom_pipe_height, pipe_width, bottom_pipe_height)
            pipes.append((new_pipe_top, new_pipe_bottom))
            last_pipe = frame

        # Move pipes and check for score
        for pipe_pair in pipes:
            pipe_top, pipe_bottom = pipe_pair
            pipe_top.x -= 5
            pipe_bottom.x -= 5

            # Check if the bird has passed the pipe
            if bird_x > pipe_top.right and pipe_pair not in passed_pipes:
                score += 1
                if score > high_score:
                    high_score = score
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

        # Get the two sets of pipes that are closest to the bird but not in passed pipes
        closest_pipes = []
        for pipe_pair in pipes:
            if pipe_pair not in passed_pipes:
                if not closest_pipes:
                    closest_pipes.append(pipe_pair)
                elif pipe_pair[0].right > closest_pipes[-1][0].right:
                    closest_pipes.append(pipe_pair)
                if len(closest_pipes) == 2:
                    break
        next_pipe_top, next_pipe_bottom = closest_pipes[0]
        if len(pipes) > 1:
            next_next_pipe_top, next_next_pipe_bottom = closest_pipes[1]
        else:
            next_next_pipe_top, next_next_pipe_bottom = None, None

        next_bird_y = bird_y + bird_movement

        if next_next_pipe_top is not None and next_next_pipe_bottom is not None:
            next_state = (bird_y, next_bird_y, next_pipe_top.bottom, next_pipe_bottom.top, next_pipe_top.left - bird_x, next_next_pipe_top.bottom, next_next_pipe_bottom.top, next_next_pipe_top.left - bird_x)
        else:
            next_state = (bird_y, next_bird_y, next_pipe_top.bottom, next_pipe_bottom.top, next_pipe_top.left - bird_x, 0, 0, 0)
            
        # Draw everything
        screen.fill((0, 0, 0))  # Fill screen with black
        for pipe_top, pipe_bottom in pipes:
            pygame.draw.rect(screen, (0, 128, 0), pipe_top)  # Draw top pipe
            # Draw bottom pipe
            pygame.draw.rect(screen, (0, 128, 0), pipe_bottom)
        pygame.draw.circle(screen, (0, 0, 255), (bird_x, bird_y), bird_radius)  # Draw bird

        # Display score
        font = pygame.font.SysFont(None, 36)
        score_surface = font.render(f'Score: {score}', True, (255, 255, 255))
        screen.blit(score_surface, (10, 40))
        # Display high score
        high_score_surface = font.render(f'High Score: {high_score}', True, (255, 255, 255))
        screen.blit(high_score_surface, (10, 10))

        # Update the display
        pygame.display.update()

        # Cap the frame rate
        # pygame.time.Clock().tick(30)

        frame += 1

    pygame.quit()


model_episode = 749
model_dir = 'model_checkpoints'
model_path = os.path.join(model_dir, f"model_episode_{749}.keras")
model = load_model(model_path)

global high_score
high_score = 0

while True:
    flappy_bird(model)