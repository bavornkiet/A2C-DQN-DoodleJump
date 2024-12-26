from rewards import formulate_reward
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
    def __init__(self, difficulty='EASY', server=False, reward_type=1, FPS=None, render_skip=0):
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

    def updatePlayer(self):
        if self.die == 1:
            self.screen.blit(self.playerdead, (self.playerx,
                             self.playery - self.cameray))
            return
        if not self.jump:
            self.playery += self.gravity
            self.gravity += 1
        elif self.jump:
            self.playery -= self.jump
            self.jump -= 1
        key = pygame.key.get_pressed()
        if key[K_RIGHT]:
            if self.xmovement < 10:
                self.xmovement += 1
            self.direction = 0

        elif key[K_LEFT]:
            if self.xmovement > -10:
                self.xmovement -= 1
            self.direction = 1
        else:
            if self.xmovement > 0:
                self.xmovement -= 1
            elif self.xmovement < 0:
                self.xmovement += 1
        if self.playerx > 850:
            self.playerx = -50
        elif self.playerx < -50:
            self.playerx = 850
        self.playerx += self.xmovement
        if self.playery - self.cameray <= 200:
            self.cameray -= 10
        if not self.direction:
            if self.jump:
                self.screen.blit(self.playerRight_1,
                                 (self.playerx, self.playery - self.cameray))
            else:
                self.screen.blit(self.playerRight,
                                 (self.playerx, self.playery - self.cameray))
        else:
            if self.jump:
                self.screen.blit(self.playerLeft_1,
                                 (self.playerx, self.playery - self.cameray))
            else:
                self.screen.blit(
                    self.playerLeft, (self.playerx, self.playery - self.cameray))

    def updatePlatforms(self):
        for p in self.platforms:
            rect = pygame.Rect(
                p[0], p[1], self.green.get_width() - 10, self.green.get_height())
            player = pygame.Rect(self.playerx, self.playery, self.playerRight.get_width(
            ) - 10, self.playerRight.get_height())

            if rect.colliderect(player) and self.gravity and self.playery < (p[1] - self.cameray):
                if p[2] != 2:
                    self.jump = 15
                    self.gravity = 0
                else:
                    if p[-1] != 1:
                        self.jump = 15  # jump even when you hit red broken platform
                        p[-1] = 1
                    else:
                        self.jump = 0

            # moving blue platform left and right
            if p[2] == 1:
                if p[-1] == 1:
                    p[0] += 5
                    if p[0] > 750:
                        p[-1] = 0
                else:
                    p[0] -= 5
                    if p[0] <= 0:
                        p[-1] = 1

    def drawPlatforms(self):
        score_increment = False
        spring_touch = False
        monster_touch = False

        for p in self.platforms:
            # print("platform, ",(self.platforms))
            check = self.platforms[0][1] - self.cameray
            if check > 800:
                x1 = random.randint(0, 700)
                platform1 = self.getNewPlatform(
                    x1, self.platforms[-1][1] - self.inter_platform_distance)
                self.platforms.append(platform1)

                second_platform_prob = random.randint(0, 1000)
                if second_platform_prob <= self.second_platform_prob:
                    x2 = x1
                    while abs(x1 - x2) < 200:
                        x2 = random.randint(0, 700)
                    platform2 = self.getNewPlatform(
                        x2, self.platforms[-2][1] - self.inter_platform_distance)
                    self.platforms.append(platform2)

                coords = self.platforms[-1]
                check = random.randint(0, 1000)

                if check > 900 and coords[2] == 0:
                    self.springs.append([coords[0], coords[1] - 25, 0])

                # monsters after 25k score
                elif check > 860 and coords[2] == 0 and self.score > 25_000:
                    self.monsters.append([coords[0], coords[1] - 50, 0])

                first_platform_popped = self.platforms.pop(0)
                # popping second platform on same level
                if self.platforms[0][1] == first_platform_popped[1]:
                    self.platforms.pop(0)

                self.score += 100
                score_increment = True

            if p[2] == 0:
                self.screen.blit(self.green, (p[0], p[1] - self.cameray))
            elif p[2] == 1:
                self.screen.blit(self.blue, (p[0], p[1] - self.cameray))
            elif p[2] == 2:
                if not p[3]:
                    self.screen.blit(self.red, (p[0], p[1] - self.cameray))
                else:
                    self.screen.blit(self.red_1, (p[0], p[1] - self.cameray))

        for spring in self.springs:
            if spring[-1]:
                self.screen.blit(
                    self.spring_1, (spring[0], spring[1] - self.cameray))
            else:
                self.screen.blit(
                    self.spring, (spring[0], spring[1] - self.cameray))
            if pygame.Rect(spring[0], spring[1], self.spring.get_width(), self.spring.get_height()).colliderect(pygame.Rect(self.playerx, self.playery, self.playerRight.get_width(), self.playerRight.get_height())):
                self.jump = 35
                self.cameray -= 40
                spring_touch = True

        for monster in self.monsters:
            self.screen.blit(
                self.monster, (monster[0], monster[1] - self.cameray))
            if pygame.Rect(monster[0], monster[1], self.monster.get_width(), self.monster.get_height()).colliderect(pygame.Rect(self.playerx, self.playery, self.playerRight.get_width(), self.playerRight.get_height())):
                self.screen.blit(self.monsterdead,
                                 (monster[0], monster[1] - self.cameray))
                self.die = 1
                monster_touch = True

        return score_increment, spring_touch, monster_touch

    def getNewPlatform(self, x, on):
        if self.score < 10_000:
            return [x, on, 0, 0]
        elif 10_000 <= self.score < 25_000:
            platform = random.randint(0, 1000)
            if platform < 850:
                platform = 0
            else:
                platform = 1
            return [x, on, platform, 0]
        else:
            platform = random.randint(0, 1000)
            if platform < 800:
                platform = 0
            elif platform < 900:
                platform = 1
            else:
                platform = 2
            return [x, on, platform, 0]

    def generatePlatforms(self):
        on = 800
        while on > -100:
            x1 = random.randint(100, 700)
            platform1 = self.getNewPlatform(x1, on)
            self.platforms.append(platform1)

            second_platform_prob = random.randint(0, 1000)
            if second_platform_prob <= self.second_platform_prob:
                x2 = x1
                while abs(x1 - x2) < 200:
                    x2 = random.randint(0, 700)
                platform2 = self.getNewPlatform(x2, on)
                self.platforms.append(platform2)

            on -= self.inter_platform_distance

    def drawGrid(self):
        for x in range(80):
            pygame.draw.line(self.screen, (222, 222, 222),
                             (x * 12, 0), (x * 12, 800))
            pygame.draw.line(self.screen, (222, 222, 222),
                             (0, x * 12), (800, x * 12))

    def updatePlayerByAction(self, actions):
        """
            - actions = ['ACTION_LEFT', 'NO_ACTION', 'ACTION_RIGHT']
            - Param:
                - actions: a list that contains three boolean value.
            - To be used by playStep function in game script.
        """
        if not self.jump:
            self.playery += self.gravity
            self.gravity += 1
        elif self.jump:
            self.playery -= self.jump
            self.jump -= 1

        if actions[2]:
            if self.xmovement < 10:
                self.xmovement += 1
            self.direction = 0

        elif actions[0]:
            if self.xmovement > -10:
                self.xmovement -= 1
            self.direction = 1

        else:  # action[1] is true
            if self.xmovement > 0:
                self.xmovement -= 1
            elif self.xmovement < 0:
                self.xmovement += 1

        if self.playerx > 850:
            self.playerx = -50
        elif self.playerx < -50:
            self.playerx = 850

        self.playerx += self.xmovement
        if self.playery - self.cameray <= 200:
            self.cameray -= 10
        if not self.direction:
            if self.jump:
                self.screen.blit(self.playerRight_1,
                                 (self.playerx, self.playery - self.cameray))
            else:
                self.screen.blit(self.playerRight,
                                 (self.playerx, self.playery - self.cameray))
        else:
            if self.jump:
                self.screen.blit(self.playerLeft_1,
                                 (self.playerx, self.playery - self.cameray))
            else:
                self.screen.blit(
                    self.playerLeft, (self.playerx, self.playery - self.cameray))

    def getFeatures(self, max_platforms=10, max_monsters=3):
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
        max_monsters = 3
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
    # player_screen_y = self.playery - self.cameray
    # platforms_above = [p for p in self.platforms if (
    #     p[1] - self.cameray) < player_screen_y]
    # platforms_above.sort(key=lambda x: x[1])  # Sort by actual platform y

    # plat_features = []
    # for i in range(3):
    #     if i < len(platforms_above):
    #         px = platforms_above[i][0] / float(self.screen_width)
    #         py = (platforms_above[i][1] -
    #               self.cameray) / float(self.screen_height)
    #         # p[2] is the platform type: 0=green, 1=blue(moving), 2=red(breaking)
    #         # normalize type to [0, 0.5, 1.0]
    #         p_type = platforms_above[i][2] / 2.0
    #         plat_features += [px, py, p_type]
    #     else:
    #         # If fewer than 3 platforms are available, pad with zeros
    #         plat_features += [0.0, 0.0, 0.0]

    # features = np.array([
    #     player_x_norm,
    #     player_y_norm,
    #     x_velocity_norm,
    #     jump_norm,
    #     gravity_norm
    # ] + plat_features, dtype=np.float32)

    # return features

    def getCurrentFrame(self):

        image_data = pygame.surfarray.array3d(pygame.display.get_surface())
        return image_data

    # def getFeatures(self):
    #     features = {}
    #     x = 0
    #     features['player_x'] = self.playerx
    #     features['player_y'] = self.playery - \
    #         self.cameray  # Adjusted for camera
    #     features['player_xmovement'] = self.xmovement
    #     features['player_ymovement'] = - \
    #         self.jump if self.jump else self.gravity
    #     features['score'] = self.score

    #     # Platforms that are currently visible on the screen
    #     visible_platforms = []
    #     for p in self.platforms:
    #         platform_y = p[1] - self.cameray
    #         if 0 <= platform_y <= 800:  # Screen height is 800 pixels
    #             visible_platforms.append({
    #                 'x': p[0],
    #                 'y': platform_y,
    #                 'type': p[2],  # Platform type: 0-green, 1-blue, 2-red
    #                 'state': p[3]  # State for moving/broken platforms
    #             })
    #             x += 1
    #     # Sort platforms by y-coordinate
    #     visible_platforms.sort(key=lambda p: p['y'])
    #     # Include all visible platforms
    #     features['platforms'] = visible_platforms
    #     # print(x)
    #     # Visible springs
    #     visible_springs = []
    #     for s in self.springs:
    #         spring_y = s[1] - self.cameray
    #         if 0 <= spring_y <= 800:
    #             visible_springs.append({
    #                 'x': s[0],
    #                 'y': spring_y,
    #                 'state': s[2]  # Spring state
    #             })
    #     features['springs'] = visible_springs

    #     # Visible monsters
    #     visible_monsters = []
    #     for m in self.monsters:
    #         monster_y = m[1] - self.cameray
    #         if 0 <= monster_y <= 800:
    #             visible_monsters.append({
    #                 'x': m[0],
    #                 'y': monster_y,
    #                 'state': m[2]  # Monster state
    #             })
    #     features['monsters'] = visible_monsters

    #     return features

    def playStep(self, actions):
        last_cameray = self.cameray
        terminal = False
        reward = formulate_reward(self.reward_type, "ALIVE")
        return_score = self.score

        pygame.display.flip()
        self.screen.fill((255, 255, 255))
        self.FPSCLOCK.tick(self.FPS)
        # self.frame_count += 1
        # if self.frame_count % self.render_skip == 0:
        #     pygame.display.flip()
        for event in pygame.event.get():
            if event.type == QUIT:
                sys.exit()
        if self.die == 1 or (self.playery - self.cameray > 900):
            # features = self.getFeatures()
            return_score = self.gameReboot()
            terminal = True
            reward = formulate_reward(self.reward_type, "DEAD")
            print("terminated: Agent Died")
            return reward, terminal, return_score

        self.drawGrid()
        score_inc, spring_touch, monster_touch = self.drawPlatforms()

        if score_inc:
            reward = formulate_reward(
                self.reward_type, "SCORED", spring_touch, monster_touch, self.score)
            self.timer = time.time()
        elif last_cameray == self.cameray:
            # Check if doodler is on the same place for past 100 sec
            if self.timer is None:
                self.timer = time.time()
            else:
                now_time = time.time()
                if (now_time - self.timer) > 10:
                    # features = self.getFeatures()
                    return_score = self.gameReboot()
                    terminal = True
                    reward = formulate_reward(self.reward_type, "STUCK")
                    print("terminated: Agent stuck")
                    return reward, terminal, return_score

        self.updatePlayerByAction(actions)
        self.updatePlatforms()
        self.screen.blit(self.font.render(
            str(self.score), -1, (0, 0, 0)), (25, 25))
        pygame.display.flip()
        # if not terminal:
        #     features = self.getFeatures()
        actual_fps = self.FPSCLOCK.get_fps()
        # print(f"Actual FPS: {actual_fps}")

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
            clock.tick(60)

            for event in pygame.event.get():
                if event.type == QUIT:
                    sys.exit()
            if self.die == 1 or (self.playery - self.cameray > 900):
                old_score = self.gameReboot()

            self.drawGrid()
            self.drawPlatforms()
            self.updatePlayer()
            self.updatePlatforms()
            self.screen.blit(self.font.render(
                str(self.score), -1, (0, 0, 0)), (25, 25))
            pygame.display.flip()


if __name__ == "__main__":
    game = DoodleJump()
    game.run()
