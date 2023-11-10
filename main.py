import pygame
import random
import numpy as np
from time import sleep
# import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, InputLayer
import matplotlib.pyplot as plt



def train_model(model, replay_buffer, batch_size, discount_factor):
    # Sample a batch of experiences from the replay buffer
    minibatch = replay_buffer.sample(batch_size)

    # Separate the data into states, actions, rewards, etc.
    states, actions, rewards, next_states, dones = map(np.array, zip(*minibatch))

    # Predict Q-values for starting states
    predicted_q_values = model.predict(states, verbose=0)

    # Predict Q-values for next states
    predicted_next_q_values = model.predict(next_states, verbose=0)

    # Initialize target q-values as the predicted q-values
    target_q_values = predicted_q_values.copy()

    # Update the Q-values for actions taken
    for i in range(batch_size):
        if dones[i]:
            target_q_values[i, actions[i]] = rewards[i]
        else:
            target_q_values[i, actions[i]] = rewards[i] + discount_factor * np.max(predicted_next_q_values[i])

    # Train the model
    model.fit(states, target_q_values, epochs=1, verbose=0)


def choose_action(state, model, epsilon, action_size):
    should_make_rand_decision = np.random.rand()
    if should_make_rand_decision <= epsilon:
        # Exploration: choose a random action
        # decision = random.randrange(action_size)
        decision = np.random.choice([0, 1], p=[0.9, 0.1])
        # print(f"random action: {decision}")
        return decision
    else:
        # print("model action")
        state = np.array(state).reshape(1, -1)
        # print(state)
        q_values = model.predict(state, verbose=0)
        # print(q_values)
        # Exploitation: choose the best action based on model
        # print(np.argmax(q_values[0]))
        # print(f"agent action: {np.argmax(q_values[0])}")
        if np.argmax(q_values[0]) == 1:
            global agent_jumps
            agent_jumps += 1
        return np.argmax(q_values[0])


def create_dqn_model(state_shape, action_size=2):
    model = Sequential([
        InputLayer(input_shape=state_shape),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(action_size, activation='linear')  # Output layer
    ])
    model.compile(optimizer='adam', loss='mse')
    return model


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.total_steps = 0

    def add(self, state, action, reward, next_state, done):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, min(len(self.buffer), batch_size))

    def step(self):
        self.total_steps += 1

    def __len__(self):
        return len(self.buffer)


def flappy_bird(model, epsilon, replay_buffer, training_start, training_interval, batch_size, discount_factor):
    # Initialize Pygame
    pygame.init()

    # Game Variables
    screen_width = 720
    screen_height = 480
    framerate = 30
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
    total_reward = 0
    reward = 0
    state = (bird_y, bird_movement, 0, 0, 0)
    next_state = (bird_y, bird_movement, 0, 0, 0)

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
            # if event.type == pygame.KEYDOWN:
            #     if event.key == pygame.K_SPACE:
            #         bird_movement = -jump_height

        if len(replay_buffer) > training_start and replay_buffer.total_steps % training_interval == 0:
            print("Training model")
            train_model(model, replay_buffer, batch_size, discount_factor)

        state = next_state
        # Get agent decision
        # Choose action based on ε-greedy strategy
        action = choose_action(state, model, epsilon, 2)

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
            new_pipe_top = pygame.Rect(
                screen_width, 0, pipe_width, top_pipe_height)
            new_pipe_bottom = pygame.Rect(
                screen_width, screen_height - bottom_pipe_height, pipe_width, bottom_pipe_height)
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
                reward += 1000
                passed_pipes.append(pipe_pair)

        # Remove off-screen pipes
        pipes = [pipe_pair for pipe_pair in pipes if pipe_pair[0].right > 0]

        # Collision detection
        bird_rect = pygame.Rect(bird_x - bird_radius, bird_y -
                                bird_radius, bird_radius * 2, bird_radius * 2)
        for pipe_top, pipe_bottom in pipes:
            if bird_rect.colliderect(pipe_top) or bird_rect.colliderect(pipe_bottom):
                running = False  # End the game

        if bird_y > screen_height - bird_radius or bird_y < bird_radius:
            running = False  # Bird hits the ground or goes off screen

        # if running:
        #     reward += 1
        # else:
            # reward -= 1000

        if not running:
            reward -= 1000


        # Implement this function to define the state
        pipe_top, pipe_bottom = pipes[0]
        next_bird_y = bird_y + bird_movement
        # next_state = (bird_y, bird_movement, pipe_top.bottom,
        #               pipe_bottom.top, pipe_top.left - bird_x)

        # print(pipe_top.bottom, pipe_top.top, pipe_bottom.top, pipe_bottom.bottom)
        # print(bird_y)
        if bird_y > pipe_top.bottom and bird_y < pipe_bottom.top:
            # print(bird_y, pipe_top.bottom, pipe_bottom.top)
            # print("bird in pipe")
            reward += 1
        # else:
        #     reward -= 0.01

        next_state = (bird_y, next_bird_y, pipe_top.bottom, pipe_bottom.top, pipe_top.left - bird_x)

        if not running:
            done = True
            global reward_final
            reward_final = total_reward
            
        total_reward += reward

        replay_buffer.add(state, action, reward, next_state, done)

        reward = 0

        replay_buffer.step()

        # Draw everything
        screen.fill((0, 0, 0))  # Fill screen with black
        for pipe_top, pipe_bottom in pipes:
            pygame.draw.rect(screen, (0, 128, 0), pipe_top)  # Draw top pipe
            # Draw bottom pipe
            pygame.draw.rect(screen, (0, 128, 0), pipe_bottom)
        pygame.draw.circle(screen, (0, 0, 255), (bird_x, bird_y),
                           bird_radius)  # Draw bird

        # Display score
        font = pygame.font.SysFont(None, 36)
        score_surface = font.render(f'Score: {score}', True, (255, 255, 255))
        screen.blit(score_surface, (10, 10))

        # Update the display
        pygame.display.update()

        # Cap the frame rate
        # pygame.time.Clock().tick(30)

        frame += 1

    pygame.quit()


model = create_dqn_model((5,))
starting_episode = 0
# model = load_model(f"model_{starting_episode}.keras")
total_episodes = 1000
# epsilon = 1.0  # Starting epsilon value
epsilon = 1.0
# epsilon_min = 0.01  # Minimum epsilon value
epsilon_min = 0.1  # Minimum epsilon value
# epsilon_decay = 0.995  # Factor to decrease epsilon
epsilon_decay = 0.995
training_start = 1000  # Start training after 1,000 steps
training_interval = 200  # Train every 200 steps
batch_size = 32
replay_buffer = ReplayBuffer(5000)
discount_factor = 0.99
global agent_jumps
agent_jumps = 0
global reward_final
reward_final = 0
x_axis = []
y_axis = []

for episode in range(starting_episode, total_episodes):
    print("Epsilon: ", epsilon)
    # Decay epsilon
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay
    print("episode ", episode, " started")
    flappy_bird(model, epsilon, replay_buffer, training_start, training_interval, batch_size, discount_factor)
    x_axis.append(agent_jumps)
    y_axis.append(reward_final)
    if episode % 200 == 0:
        model.save(f"model_{episode}.keras")
    print("agent jumps this episode: ", agent_jumps)
    print()
    agent_jumps = 0
    reward_final = 0


# Example data
# x = np.linspace(0, 10, 100)  # x-axis data
# y = np.sin(x)               # y-axis data

# Generating a color gradient from red to blue
colors = np.linspace(0, 1, len(x_axis))
scatter = plt.scatter(x_axis, y_axis, c=colors, cmap='coolwarm')

# Adding colorbar for reference
plt.colorbar(scatter)

# Labeling the axes
plt.xlabel('X-axis')
plt.ylabel('Y-axis')

# Title of the plot
plt.title('Scatter Plot with Color Gradient from Red to Blue')

# Show the plot
plt.show()