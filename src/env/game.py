import pygame
import sys
import random
import math
import numpy as np
import time

# Game Constants
RENDER = True
RESOLUTION = WIDTH, HEIGHT = 360, 640
TITLE = "Doodle Jump"
TIME_SPEED = 1

pygame.init()
gravity = 0.15
background_color = (250, 248, 239)
x_scale = WIDTH / 360
y_scale = HEIGHT / 640

# Initialize Pygame display if rendering
if RENDER:
    pygame.font.init()
    screen = pygame.display.set_mode(RESOLUTION)
    pygame.display.set_caption(TITLE)

gravity *= y_scale

# Define the Player class


class Player():
    jump_force = 8 * y_scale
    max_x_speed = 16 * x_scale
    x_acceleration = 0.15 * x_scale
    color = (255, 255, 0)
    color2 = (0, 255, 255)
    height = 32 * y_scale
    width = 32 * y_scale

    def __init__(self):
        self.y = HEIGHT - self.height
        self.x = (WIDTH - self.width) // 2
        self.y_speed = -self.jump_force
        self.x_speed = 0
        self.direction = 0
        self.moving_direction = 0
        self.score = 0

    def move(self, left_key_pressed, right_key_pressed, time_scale):
        self.y_speed += gravity * time_scale
        self.y += self.y_speed * time_scale

        if left_key_pressed == right_key_pressed:
            self.x_speed = (max(0, abs(
                self.x_speed) - (self.x_acceleration / 2) * time_scale)) * self.moving_direction
            self.direction = 0
        elif left_key_pressed:
            self.x_speed = max(-self.max_x_speed, self.x_speed -
                               self.x_acceleration * time_scale)
            self.direction = -1
            self.moving_direction = -1
        elif right_key_pressed:
            self.x_speed = min(self.max_x_speed, self.x_speed +
                               self.x_acceleration * time_scale)
            self.direction = 1
            self.moving_direction = 1

        self.x += self.x_speed * time_scale

        if self.x + self.width + 20 < 0:
            self.x = WIDTH
        if self.x > WIDTH:
            self.x = -20 - self.width

    def jump(self):
        self.y_speed = -self.jump_force

    def high_jump(self):
        self.y_speed = -self.jump_force * 2

    def draw(self, screen):
        pygame.draw.rect(screen, self.color,
                         (self.x, self.y, self.width, self.height))
        pygame.draw.rect(screen, (0, 0, 0),
                         (self.x, self.y, self.width, self.height), 1)
        if self.direction <= 0:
            pygame.draw.rect(screen, self.color2, (self.x + 6 * y_scale,
                             self.y + 6 * y_scale, 7 * y_scale, 7 * y_scale))
            pygame.draw.rect(screen, (0, 0, 0), (self.x + 6 * y_scale,
                             self.y + 6 * y_scale, 7 * y_scale, 7 * y_scale), 1)
        if self.direction >= 0:
            pygame.draw.rect(screen, self.color2, (self.x + self.width -
                             13 * y_scale, self.y + 6 * y_scale, 7 * y_scale, 7 * y_scale))
            pygame.draw.rect(screen, (0, 0, 0), (self.x + self.width - 13 *
                             y_scale, self.y + 6 * y_scale, 7 * y_scale, 7 * y_scale), 1)
        if self.direction == 1:
            pygame.draw.rect(screen, self.color2, (self.x + self.width -
                             15 * y_scale, self.y + 18 * y_scale, 15 * y_scale, 7 * y_scale))
            pygame.draw.rect(screen, (0, 0, 0), (self.x + self.width - 15 *
                             y_scale, self.y + 18 * y_scale, 15 * y_scale, 7 * y_scale), 1)
        elif self.direction == -1:
            pygame.draw.rect(screen, self.color2, (self.x,
                             self.y + 18 * y_scale, 15 * y_scale, 7 * y_scale))
            pygame.draw.rect(screen, (0, 0, 0), (self.x, self.y +
                             18 * y_scale, 15 * y_scale, 7 * y_scale), 1)
        else:
            pygame.draw.rect(screen, self.color2, (self.x + 4 * y_scale,
                             self.y + 18 * y_scale, 24 * y_scale, 7 * y_scale))
            pygame.draw.rect(screen, (0, 0, 0), (self.x + 4 * y_scale,
                             self.y + 18 * y_scale, 24 * y_scale, 7 * y_scale), 1)

# Define the Platform class


class Platform():
    width = 64 * y_scale
    height = 16 * y_scale

    def __init__(self, y, score):
        self.x = random.randint(0, int(WIDTH - self.width))
        self.y = y
        # Platform types and probabilities
        if score < 500:
            self.type = 0
        elif score < 1500:
            self.type = random.choice([0] * 6 + [1] * 2)
        elif score < 2500:
            self.type = random.choice([0] * 4 + [1] * 4)
        elif score < 3500:
            self.type = random.choice([0] * 3 + [1] * 4 + [2])
        elif score < 5000:
            self.type = random.choice([0] * 2 + [1] * 3 + [2, 3])
        else:
            self.type = random.choice([1] * 5 + [2, 3, 3])

        if self.type == 0:
            self.color = (63, 255, 63)
            self.direction = 0
            self.alpha = -1
            self.have_spring = random.choice([False] * 15 + [True])
        elif self.type == 1:
            self.color = (127, 191, 255)
            self.direction = random.choice([-1, 1]) * y_scale
            self.have_spring = random.choice([False] * 15 + [True])
            self.alpha = -1
        elif self.type == 2:
            self.color = (191, 0, 0)
            self.direction = 0
            self.have_spring = False
            self.alpha = 0
        else:
            self.color = background_color
            self.direction = 0
            self.have_spring = False
            self.alpha = -1

    def move(self, time_scale):
        self.x += self.direction * time_scale
        if self.x < 0:
            self.direction *= -1
            self.x = 0
        if self.x + self.width > WIDTH:
            self.direction *= -1
            self.x = WIDTH - self.width

    def draw(self, screen, time_scale):
        pygame.draw.rect(screen, self.color,
                         (self.x, self.y, self.width, self.height))
        pygame.draw.rect(screen, (0, 0, 0),
                         (self.x, self.y, self.width, self.height), 1)
        if self.alpha > 0:
            self.alpha += 16 * time_scale
            s = pygame.Surface((self.width, self.height))
            s.set_alpha(self.alpha)
            s.fill(background_color)
            screen.blit(s, (self.x, self.y))

# Define the Spring class


class Spring():
    width = 16 * y_scale
    height = 8 * y_scale
    color = (127, 127, 127)

    def __init__(self, platform):
        self.x = platform.x + int(platform.width // 2) - int(self.width // 2) + random.randint(
            -int(platform.width // 2) + int(self.width // 2), int(platform.width // 2) - int(self.width // 2))

        self.y = platform.y - self.height
        self.direction = platform.direction
        self.left_edge = platform.x
        self.right_edge = platform.x + platform.width - self.width

    def move(self, time_scale):
        self.x += self.direction * time_scale
        if self.x < self.left_edge:
            self.direction *= -1
            self.x = self.left_edge
        if self.x > self.right_edge:
            self.direction *= -1
            self.x = self.right_edge

    def draw(self, screen):
        pygame.draw.rect(screen, self.color,
                         (self.x, self.y, self.width, self.height))
        pygame.draw.rect(screen, (0, 0, 0),
                         (self.x, self.y, self.width, self.height), 1)

# Define the DoodleJumpEnv class


class DoodleJumpEnv:
    def __init__(self):
        self.RENDER = RENDER
        if self.RENDER:
            self.screen = screen
        self.gravity = gravity
        self.WIDTH = WIDTH
        self.HEIGHT = HEIGHT
        self.y_scale = y_scale
        self.background_color = background_color
        self.reset_game()

    def reset_game(self):
        self.player = Player()
        self.platforms = [Platform(self.HEIGHT - 1, 0)]
        self.platforms[0].x = 0
        self.platforms[0].width = self.WIDTH
        self.platforms[0].color = (0, 0, 0)
        self.springs = []
        self.time_scale = 1
        self.prev_time = pygame.time.get_ticks()
        self.high_score = 0

    def frame_step(self, action):
        reward = 0
        terminal = False

        # Convert action to game inputs
        left_key_pressed, right_key_pressed = self.process_action(action)

        # Move the player
        self.player.move(left_key_pressed, right_key_pressed, self.time_scale)

        # Camera movement
        movement = 0
        if self.player.y < self.HEIGHT // 2 - self.player.height:
            movement = (self.HEIGHT // 2 - self.player.height - self.player.y)
            self.player.y = self.HEIGHT // 2 - self.player.height
            self.player.score += movement / 4 / self.y_scale
            if movement > 0:
                reward += 1  # Reward for moving up

        # Update game objects
        self.update_game(movement)

        # Generate new platforms and springs
        self.new_platforms()

        # Check for game over
        if self.is_game_over():
            reward = -100
            terminal = True
            self.reset_game()
        else:
            reward += 0.1  # Small reward for staying alive

        # Render the game
        if self.RENDER:
            self.render_game()

        # Get observation
        observation = self.get_observation()

        # Update time scale
        current_time = pygame.time.get_ticks()
        self.time_scale = (current_time - self.prev_time) / 10 * TIME_SPEED
        self.prev_time = current_time

        return observation, reward, terminal

    def process_action(self, action):
        # Action mapping: 0 = do nothing, 1 = move left, 2 = move right
        left_key_pressed = False
        right_key_pressed = False

        if action == 1:
            left_key_pressed = True
        elif action == 2:
            right_key_pressed = True

        return left_key_pressed, right_key_pressed

    def update_game(self, movement):
        i = 0
        while i < len(self.platforms):
            platform = self.platforms[i]
            platform.y += movement
            platform.move(self.time_scale)
            # Collision detection
            if self.player.y_speed >= 0 and self.player.x < platform.x + Platform.width and self.player.x + Player.width > platform.x and self.player.y + Player.height <= platform.y + self.time_scale * self.player.y_speed and self.player.y + Player.height >= platform.y:
                if platform.type != 2:
                    self.player.y = platform.y - Player.height
                    self.player.jump()
                    if platform.type == 3:
                        del self.platforms[i]
                        i -= 1
                else:
                    platform.alpha = max(1, platform.alpha)
                    if platform.alpha >= 255:
                        del self.platforms[i]
                        i -= 1
            i += 1

        for spring in self.springs:
            spring.y += movement
            spring.move(self.time_scale)
            if self.player.y_speed >= 0 and self.player.x < spring.x + Spring.width and self.player.x + Player.width > spring.x and self.player.y + Player.height >= spring.y and self.player.y <= spring.y + Spring.height:
                self.player.high_jump()

    def new_platforms(self):
        player = self.player
        if player.score < 500:
            gap_lower_bound, gap_upper_bound = 24, 48
        elif player.score < 1500:
            gap_lower_bound, gap_upper_bound = 26, 52
        elif player.score < 2500:
            gap_lower_bound, gap_upper_bound = 28, 56
        elif player.score < 3500:
            gap_lower_bound, gap_upper_bound = 30, 60
        elif player.score < 5000:
            gap_lower_bound, gap_upper_bound = 32, 64
        else:
            gap_lower_bound, gap_upper_bound = 34, 68

        # Delete platforms below the screen
        self.platforms = [p for p in self.platforms if p.y <= self.HEIGHT]
        # Delete springs below the screen
        self.springs = [s for s in self.springs if s.y <= self.HEIGHT]

        # Generate new platforms and springs
        while self.platforms[-1].y + Platform.height >= 0:
            gap = random.randint(
                gap_lower_bound, gap_upper_bound) * self.y_scale
            platform = Platform(self.platforms[-1].y - gap, player.score)
            # Avoid 3 fake platforms in a row
            if not (platform.type == 2 and self.platforms[-1].type == 2 and self.platforms[-2].type == 2):
                self.platforms.append(platform)
            if platform.have_spring:
                self.springs.append(Spring(platform))

    def is_game_over(self):
        if self.player.score == 0 and self.player.y + Player.height > self.HEIGHT - 2:
            self.player.y = self.HEIGHT - 2 - Player.height
            self.player.jump()
            return False
        elif self.player.y > self.HEIGHT:
            return True
        return False

    def get_observation(self):
        # Return the game screen as an observation
        if self.RENDER:
            observation = pygame.surfarray.array3d(
                pygame.display.get_surface())
            observation = np.transpose(observation, (1, 0, 2))
            return observation
        else:
            # Return a simplified observation if rendering is disabled
            return np.array([self.player.x, self.player.y, self.player.x_speed, self.player.y_speed])

    def render_game(self):
        self.screen.fill(self.background_color)

        for platform in self.platforms:
            platform.draw(self.screen, self.time_scale)

        for spring in self.springs:
            spring.draw(self.screen)

        self.player.draw(self.screen)

        # Display score
        font = pygame.font.SysFont("Comic Sans MS", int(24 * y_scale))
        text = font.render("Score:", True, (0, 0, 0))
        text2 = font.render(str(int(self.player.score)), True, (0, 0, 0))
        text3 = font.render("Best:", True, (0, 0, 0))
        text4 = font.render(str(int(self.high_score)), True, (0, 0, 0))
        text_width = max(text3.get_width(), text4.get_width())
        self.screen.blit(text, (10 * y_scale, 0))
        self.screen.blit(text2, (10 * y_scale, 24 * y_scale))
        self.screen.blit(text3, (self.WIDTH - text_width - 10 * y_scale, 0))
        self.screen.blit(text4, (self.WIDTH - text_width -
                         10 * y_scale, 24 * y_scale))
        pygame.display.update()

    def close(self):
        pygame.quit()
        sys.exit()


env = DoodleJumpEnv()

# Number of episodes to run
num_episodes = 5

for episode in range(num_episodes):
    env.reset_game()
    total_reward = 0
    terminal = False
    step = 0

    print(f"Starting episode {episode + 1}")

    while not terminal:
        # Select a random action
        action = 1
        # Take a step in the environment
        observation, reward, terminal = env.frame_step(action)
        total_reward += reward
        step += 1

        # Optional: Add a delay to slow down the game
        # time.sleep(0.01)

    print(f"Episode {episode + 1} ended with total reward: {total_reward}")

env.close()
