from rewards import calculate_reward
import time
import random
import os
import numpy as np
import pygame
from pygame.locals import *
import sys
import math
# sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')

path = './game/'
sys.path.append(path)


class DoodleJump:
    def __init__(self, server=False, reward_type=1, FPS=None, render_skip=0):
        self.inter_platform_distance = 80
        self.second_platform_prob = 850

        if server:
            os.environ['SDL_VIDEODRIVER'] = 'dummy'

        pygame.font.init()
        self.screen_width = 800
        self.screen_height = 800
        self.max_x_speed = 10
        self.max_y_speed = 35
        self.max_score = 100000
        self.FPSCLOCK = pygame.time.Clock()
        self.FPS = FPS if FPS else 0  # 0 for unlimited FPS in Pygame
        self.render_skip = render_skip  # Number of frames to skip
        self.frame_count = 0
        self.fps_counter = 0
        self.last_time = time.time()

        self.reward_type = reward_type
        self.screen = pygame.display.set_mode((800, 800))

        # --------------------------
        # self.green = pygame.image.load("assets/green.png").convert_alpha()
        # self.font = pygame.font.SysFont("Arial", 25)
        # self.blue = pygame.image.load("assets/blue.png").convert_alpha()
        # self.red = pygame.image.load("assets/red.png").convert_alpha()
        # self.red_1 = pygame.image.load("assets/red_1.png").convert_alpha()
        # self.playerRight = pygame.image.load(
        #     "assets/right.png").convert_alpha()
        # self.playerRight_1 = pygame.image.load(
        #     "assets/right_1.png").convert_alpha()
        # # print(self.playerRight.get_width())
        # self.playerLeft = pygame.image.load(
        #     "assets/left.png").convert_alpha()
        # self.playerLeft_1 = pygame.image.load(
        #     "assets/left_1.png").convert_alpha()
        # self.playerdead = pygame.image.load(
        #     "assets/playerdead.png").convert_alpha()
        # self.spring = pygame.image.load(
        #     "assets/spring.png").convert_alpha()
        # self.spring_1 = pygame.image.load(
        #     "assets/spring_1.png").convert_alpha()
        # self.monster = pygame.image.load(
        #     "assets/monster1.png").convert_alpha()
        # self.monsterdead = pygame.image.load(
        #     "assets/monsterdead.png").convert_alpha()

        # ------------------------------------------
        self.green = pygame.image.load(path+"assets/green.png").convert_alpha()
        self.font = pygame.font.SysFont("Arial", 25)
        self.blue = pygame.image.load(path+"assets/blue.png").convert_alpha()
        self.red = pygame.image.load(path+"assets/red.png").convert_alpha()
        self.red_1 = pygame.image.load(path+"assets/red_1.png").convert_alpha()
        self.playerRight = pygame.image.load(
            path+"assets/right.png").convert_alpha()
        self.playerRight_1 = pygame.image.load(
            path+"assets/right_1.png").convert_alpha()
        # print(self.playerRight.get_width())
        self.playerLeft = pygame.image.load(
            path+"assets/left.png").convert_alpha()
        self.playerLeft_1 = pygame.image.load(
            path+"assets/left_1.png").convert_alpha()
        self.playerdead = pygame.image.load(
            path+"assets/playerdead.png").convert_alpha()
        self.spring = pygame.image.load(
            path+"assets/spring.png").convert_alpha()
        self.spring_1 = pygame.image.load(
            path+"assets/spring_1.png").convert_alpha()
        self.monster = pygame.image.load(
            path+"assets/monster1.png").convert_alpha()
        self.monsterdead = pygame.image.load(
            path+"assets/monsterdead.png").convert_alpha()
        self.score = 0
        self.direction = 0
        self.playerx = 400
        self.playery = 450
        self.platforms = [[400, 500, 0, 0]]
        self.springs = []
        self.monsters = []
        self.cameray = 0
        self.jump = 0
        self.gravity = 0
        self.xmovement = 0
        self.die = 0
        self.timer = None
        self.generatePlatforms()

    def handlePlayerAction(self, actions=None):
        """
        Handles player and agent actions, updates physics, movement, camera, and rendering.

        :param actions: Optional list of actions for agents. If None, keyboard input is used.
        """
        # 1. Handle Physics (Gravity and Jumping)
        if not self.jump:
            self.playery += self.gravity
            self.gravity += 1
        else:
            self.playery -= self.jump
            self.jump -= 1
            if self.jump < 0:
                self.jump = 0  # Prevent negative jump values

        # 2. Handle Movement Inputs
        if actions is not None:
            # Agent actions
            # Assuming actions = [action_left, action_idle, action_right]
            if actions[2]:  # Move Right
                if self.xmovement < 10:
                    self.xmovement += 1
                self.direction = 0
            elif actions[0]:  # Move Left
                if self.xmovement > -10:
                    self.xmovement -= 1
                self.direction = 1
            else:  # Idle
                if self.xmovement > 0:
                    self.xmovement -= 1
                elif self.xmovement < 0:
                    self.xmovement += 1
        else:
            # Player keyboard inputs
            keys = pygame.key.get_pressed()
            if keys[K_RIGHT]:
                if self.xmovement < 10:
                    self.xmovement += 1
                self.direction = 0
            elif keys[K_LEFT]:
                if self.xmovement > -10:
                    self.xmovement -= 1
                self.direction = 1
            else:
                if self.xmovement > 0:
                    self.xmovement -= 1
                elif self.xmovement < 0:
                    self.xmovement += 1

        # 3. Handle Position Wrapping
        if self.playerx > 850:
            self.playerx = -50
        elif self.playerx < -50:
            self.playerx = 850

        self.playerx += self.xmovement

        # 4. Adjust Camera
        if self.playery - self.cameray <= 200:
            self.cameray -= 10

        # 5. Render Player Sprite
        # Calculate player's y-position relative to camera
        player_y = self.playery - self.cameray

        if not self.direction:
            if self.jump:
                self.screen.blit(self.playerRight_1, (self.playerx, player_y))
            else:
                self.screen.blit(self.playerRight, (self.playerx, player_y))
        else:
            if self.jump:
                self.screen.blit(self.playerLeft_1, (self.playerx, player_y))
            else:
                self.screen.blit(self.playerLeft, (self.playerx, player_y))

    def updatePlatforms(self):
        for p in self.platforms:
            rect = pygame.Rect(
                p[0], p[1], self.green.get_width() - 10, self.green.get_height())
            player = pygame.Rect(self.playerx, self.playery, self.playerRight.get_width(
            ) - 10, self.playerRight.get_height())

            if rect.colliderect(player) and player.bottom < rect.bottom and self.gravity and self.playery < (p[1] - self.cameray):
                if p[2] != 2:
                    self.jump = 20
                    self.gravity = 0
                else:
                    if p[-1] != 1:
                        self.jump = 20 # jump even when you hit red broken platform
                        p[-1] = 1
                    else:
                        self.jump = 0

            # moving blue platform left and right
            if p[2] == 1:
                if p[-1] == 1:
                    p[0] += 5
                    if p[0] > 550:
                        p[-1] = 0
                else:
                    p[0] -= 5
                    if p[0] <= 0:
                        p[-1] = 1

    def drawPlatforms(self):
        sc = False
        sp = False
        mon = False

        if self.platforms:
            first_platform_position = self.platforms[0][1] - self.cameray
            if first_platform_position > 800:
                # Build a new platform at the top, above the highest existing one
                x_primary = random.randint(50, 650)
                new_platform1 = self.platformScore(
                    x_primary, self.platforms[-1][1] -
                    self.inter_platform_distance
                )
                self.platforms.append(new_platform1)

                # Possibly spawn a second platform at the same vertical level
                chance = random.randint(0, 1000)
                if chance <= self.second_platform_prob:
                    x_secondary = x_primary
                    while abs(x_primary - x_secondary) < 150:
                        x_secondary = random.randint(50, 650)
                    new_platform2 = self.platformScore(
                        x_secondary, self.platforms[-2][1] -
                        self.inter_platform_distance
                    )
                    self.platforms.append(new_platform2)

                # Optionally add a spring or monster on the newly created platform
                latest_coords = self.platforms[-1]
                roll = random.randint(0, 1000)
                if roll > 900 and latest_coords[2] == 0:
                    # If the platform is green, place a spring on it
                    self.springs.append(
                        [latest_coords[0], latest_coords[1] - 25, 0])
                elif roll > 860 and latest_coords[2] == 0 and self.score > 25_000:
                    # Higher-level play: place a monster on it
                    self.monsters.append(
                        [latest_coords[0], latest_coords[1] - 50, 0])

                # Remove old (lowest) platform(s)
                removed_platform = self.platforms.pop(0)
                # If the next platform has the same y-level, remove that too
                if self.platforms and self.platforms[0][1] == removed_platform[1]:
                    self.platforms.pop(0)

                # Increase the score each time you effectively pass a platform
                self.score += 100
                sc = True

        for platform_info in self.platforms:
            x_pos, y_pos, p_type, p_flag = platform_info
            adjusted_y = y_pos - self.cameray

            if p_type == 0:  # green
                self.screen.blit(self.green, (x_pos, adjusted_y))
            elif p_type == 1:  # blue
                self.screen.blit(self.blue, (x_pos, adjusted_y))
            elif p_type == 2:  # red (may be broken)
                if not p_flag:
                    self.screen.blit(self.red, (x_pos, adjusted_y))
                else:
                    self.screen.blit(self.red_1, (x_pos, adjusted_y))

        for spring_info in self.springs:
            sx, sy, spring_state = spring_info
            sy_adjusted = sy - self.cameray

            if spring_state:
                self.screen.blit(self.spring_1, (sx, sy_adjusted))
            else:
                self.screen.blit(self.spring, (sx, sy_adjusted))

            # Collision detection with player
            if pygame.Rect(
                sx, sy, self.spring.get_width(), self.spring.get_height()
            ).colliderect(
                pygame.Rect(
                    self.playerx, self.playery,
                    self.playerRight.get_width(),
                    self.playerRight.get_height()
                )
            ):
                self.jump = 35
                self.cameray -= 40
                sp = True

        for m_info in self.monsters:
            mx, my, _ = m_info
            my_adjusted = my - self.cameray

            self.screen.blit(self.monster, (mx, my_adjusted))
            # If player collides with a monster, doodler dies
            if pygame.Rect(
                mx, my, self.monster.get_width(), self.monster.get_height()
            ).colliderect(
                pygame.Rect(
                    self.playerx, self.playery,
                    self.playerRight.get_width(),
                    self.playerRight.get_height()
                )
            ):
                self.screen.blit(self.monsterdead, (mx, my_adjusted))
                self.die = 1
                mon = True

        return sc, sp, mon

    def platformScore(self, x, vertical):
        match self.score:
            case score if score < 10_000:
                # Only green platforms at low scores
                return [x, vertical, 0, 0]

            case score if 10_000 <= score < 25_000:
                # Mostly green, sometimes blue
                kind_rand = random.randint(0, 1000)
                kind = 0 if kind_rand < 850 else 1
                return [x, vertical, kind, 0]

            case score if score >= 25_000:
                # Mix of green, blue, and red
                kind_rand = random.randint(0, 1000)
                if kind_rand < 800:
                    kind = 0
                elif kind_rand < 900:
                    kind = 1
                else:
                    kind = 2
                return [x, vertical, kind, 0]

            case _:
                # Optional: Handle unexpected score ranges
                raise ValueError(f"Unhandled score range: {self.score}")

    def generatePlatforms(self):

        vertical_position = 800
        while vertical_position > -300:
            # First platform at this y-level
            x_main = random.randint(50, 650)
            plat1 = self.platformScore(x_main, vertical_position)
            self.platforms.append(plat1)

            # Possibly spawn a second platform
            chance = random.randint(0, 1000)
            if chance <= self.second_platform_prob:
                x_alt = x_main
                while abs(x_main - x_alt) < 150:
                    x_alt = random.randint(50, 650)
                plat2 = self.platformScore(x_alt, vertical_position)
                self.platforms.append(plat2)

            vertical_position -= self.inter_platform_distance

    def drawGrid(self):
        for x in range(80):
            pygame.draw.line(self.screen, (222, 222, 222),
                             (x * 12, 0), (x * 12, 800))
            pygame.draw.line(self.screen, (222, 222, 222),
                             (0, x * 12), (800, x * 12))

    def getFeatures(self, max_platforms=5, max_monsters=1):
        """
        Returns a single 1D NumPy array encoding:
        1) Agent's normalized state
        2) Up to 10 closest platforms (relative coords, type in one-hot, broken-flag, has_spring).
        3) Up to 3 closest monsters (optional).
        """

        # -----------------------
        # 1) AGENT (DOODLER) STATE
        # -----------------------
        player_x_norm = self.playerx / float(self.screen_width)
        player_y_screen = self.playery - self.cameray
        player_y_norm = player_y_screen / float(self.screen_height)

        # Normalize velocities
        x_velocity_norm = self.xmovement / 10.0   # range ~[-1..+1]
        jump_norm = self.jump / 35.0        # range ~[0..1]
        # range ~[0..1], sometimes larger
        gravity_norm = self.gravity / 35.0

        agent_features = [
            player_x_norm,
            player_y_norm,
            x_velocity_norm,
            jump_norm,
            gravity_norm
        ]

        # ------------------------------------------------
        # 2) SELECT THE 10 CLOSEST PLATFORMS BY DISTANCE
        # ------------------------------------------------
        platforms_data = []
        for p in self.platforms:
            # p = [x, y, type, broken_flag]
            px = p[0]
            py = p[1] - self.cameray

            dx = px - self.playerx
            if dx > self.screen_width / 2:
                dx -= self.screen_width
            elif dx < -self.screen_width / 2:
                dx += self.screen_width

            dy = py - player_y_screen

            dist = math.sqrt(dx*dx + dy*dy)
            platforms_data.append((dist, px, py, p[2], p[3]))
        platforms_data.sort(key=lambda x: x[0])
        nearest_platforms = platforms_data[:max_platforms]

        # Build the sub-features for each of the 10 platforms
        platform_features = []
        for i in range(max_platforms):
            if i < len(nearest_platforms):
                (dist, px, py, p_type, p_broken) = nearest_platforms[i]

                # Recompute dx, dy with wrapping
                dx = px - self.playerx
                if dx > self.screen_width / 2:
                    dx -= self.screen_width
                elif dx < -self.screen_width / 2:
                    dx += self.screen_width
                dy = py - player_y_screen

                rel_x = dx / float(self.screen_width)
                rel_y = dy / float(self.screen_height)

                # 0=Green, 1=Blue, 2=Red
                type_vec = [0.0, 0.0, 0.0]
                if p_type in (0, 1, 2):
                    type_vec[p_type] = 1.0

                # broken_flag only if platform is Red
                broken_flag = float(p_broken) if p_type == 2 else 0.0

                # ------------------------------
                # has_spring = 1.0 if a spring is "on" this platform
                # ------------------------------
                has_spring = 0.0
                SPRING_THRESHOLD = 30.0
                for s in self.springs:
                    # s might be [sx, sy, spring_state]
                    sx = s[0]
                    sy = s[1] - self.cameray
                    # Compare to (px, py)
                    # Possibly also use wrapping on sx if needed
                    dx_s = sx - px
                    if dx_s > self.screen_width / 2:
                        dx_s -= self.screen_width
                    elif dx_s < -self.screen_width / 2:
                        dx_s += self.screen_width
                    dy_s = sy - py
                    spring_dist = math.sqrt(dx_s*dx_s + dy_s*dy_s)
                    if spring_dist < SPRING_THRESHOLD:
                        has_spring = 1.0
                        break

                # Combine into one sub-vector for this platform
                platform_features += [
                    rel_x,
                    rel_y,
                    type_vec[0],  # is_green
                    type_vec[1],  # is_blue
                    type_vec[2],  # is_red
                    broken_flag,
                    has_spring
                ]
            else:
                # If we don't have enough platforms, pad with zeros
                platform_features += [0.0]*7

        # --------------------------
        # 3) NEAREST MONSTERS (OPTIONAL)
        # --------------------------

        monsters_data = []
        for m in self.monsters:
            mx = m[0]
            my = m[1] - self.cameray
            # Optional: m[2] might be monster state or type
            m_state = m[2]

            dx = mx - self.playerx
            # wrap horizontally if needed
            if dx > self.screen_width / 2:
                dx -= self.screen_width
            elif dx < -self.screen_width / 2:
                dx += self.screen_width

            dy = my - player_y_screen
            dist = math.sqrt(dx*dx + dy*dy)
            monsters_data.append((dist, mx, my, m_state))

        # Sort and pick closest 3
        monsters_data.sort(key=lambda x: x[0])
        nearest_monsters = monsters_data[:max_monsters]

        monster_features = []
        for i in range(max_monsters):
            if i < len(nearest_monsters):
                dist, mx, my, m_state = nearest_monsters[i]
                dx = mx - self.playerx
                if dx > self.screen_width / 2:
                    dx -= self.screen_width
                elif dx < -self.screen_width / 2:
                    dx += self.screen_width
                dy = my - player_y_screen

                rel_mx = dx / float(self.screen_width)
                rel_my = dy / float(self.screen_height)
                # Just store (rel_mx, rel_my, m_state) for each monster
                monster_features += [rel_mx, rel_my, float(m_state)]
            else:
                monster_features += [0.0, 0.0, 0.0]

        # --------------------------
        # 4) CONCATENATE INTO A SINGLE FEATURE VECTOR
        # --------------------------
        features = agent_features + platform_features + monster_features
        features_array = np.array(features, dtype=np.float32)
        return features_array

    def getPixelFrame(self):
        data = pygame.surfarray.array3d(pygame.display.get_surface())
        return data

    def agentPlay(self, actions):
        """
        Executes a single step in the game based on the provided actions.

        :param actions: A list of actions for the agent, typically [action_left, action_idle, action_right].
        :return: A tuple containing the reward, terminal flag, and the current score.
        """
        # Initialize variables
        terminal = False
        reward = calculate_reward(self.reward_type, "ALIVE")
        return_score = self.score
        last_cameray = self.cameray

        pygame.display.flip()
        self.screen.fill((255, 255, 255))
        self.FPSCLOCK.tick(self.FPS)

        for event in pygame.event.get():
            if event.type == QUIT:
                sys.exit()

        if self.is_terminal_state():
            return self.handle_terminal_state("DEAD")

        self.drawGrid()
        score_inc, spring_touch, monster_touch = self.drawPlatforms()

        reward, terminal, return_score = self.update_reward_and_timer(
            score_inc, spring_touch, monster_touch, last_cameray
        )
        if terminal:
            return reward, terminal, return_score

        self.handlePlayerAction(actions)
        self.updatePlatforms()
        score_surface = self.font.render(str(self.score), True, (0, 0, 0))
        self.screen.blit(score_surface, (25, 25))
        pygame.display.flip()

        return reward, terminal, return_score

    def is_terminal_state(self):
        return self.die == 1 or (self.playery - self.cameray > 900)

    def update_reward_and_timer(self, score_inc, spring_touch, monster_touch, last_cameray):
        terminal = False
        reward = calculate_reward(self.reward_type, "ALIVE")
        return_score = self.score

        if score_inc:
            # Player has scored by passing a platform
            reward = calculate_reward(
                self.reward_type, "SCORED", spring_touch, monster_touch, self.score
            )
            self.timer = time.time()
        elif last_cameray == self.cameray:
            # Player might be stuck (camera hasn't moved)
            if self.timer is None:
                self.timer = time.time()
            else:
                elapsed_time = time.time() - self.timer
                if elapsed_time > 10:
                    # Player has been stuck for more than 10 seconds
                    return self.handle_terminal_state("STUCK")

        return reward, terminal, return_score

    def handle_terminal_state(self, reason):
        return_score = self.gameReboot()
        terminal = True
        reward = calculate_reward(self.reward_type, reason)

        if reason == "DEAD":
            print("terminated: Agent Died")
        elif reason == "STUCK":
            print("terminated: Agent stuck")

        return reward, terminal, return_score

    def gameReboot(self):

        old_score = self.score
        self.cameray = 0
        self.score = 0
        self.die = 0
        self.springs = []
        self.monsters = []
        self.platforms = [[400, 500, 0, 0]]
        self.generatePlatforms()
        self.playerx = 400
        self.playery = 400
        self.timer = None
        return old_score

    def run(self):
        clock = pygame.time.Clock()
        # x = 0
        while True:
            self.screen.fill((255, 255, 255))
            clock.tick(120)

            for event in pygame.event.get():
                if event.type == QUIT:
                    sys.exit()
            if self.die == 1 or (self.playery - self.cameray > 900):
                old_score = self.gameReboot()

            self.drawGrid()
            self.drawPlatforms()
            self.handlePlayerAction()
            self.updatePlatforms()
            self.screen.blit(self.font.render(
                str(self.score), -1, (0, 0, 0)), (25, 25))
            pygame.display.flip()


if __name__ == "__main__":
    path = ""
    game = DoodleJump()
    game.run()
