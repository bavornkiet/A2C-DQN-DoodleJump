from rewards import formulate_reward
import time
import random
import os
import pygame
from pygame.locals import *
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')

path = './game/'


class DoodleJump:
    def __init__(self, difficulty='EASY', server=False, reward_type=1):
        # To change the difficulty of the game, only tune these two parameters:
        # inter_platform_distance - distance between two platforms at two consecutive levels.
        # second_platform_prob - the probability with which you need two platforms at the same level.
        if difficulty == "HARD":
            self.inter_platform_distance = 100
            self.second_platform_prob = 700
        elif difficulty == "MEDIUM":
            self.inter_platform_distance = 90
            self.second_platform_prob = 750
        else:  # EASY
            self.inter_platform_distance = 80
            self.second_platform_prob = 850

        if server:
            os.environ['SDL_VIDEODRIVER'] = 'dummy'

        pygame.font.init()
        self.reward_type = reward_type
        self.screen = pygame.display.set_mode((800, 800))
        self.green = pygame.image.load("assets/green.png").convert_alpha()
        self.font = pygame.font.SysFont("Arial", 25)
        self.blue = pygame.image.load("assets/blue.png").convert_alpha()
        self.red = pygame.image.load("assets/red.png").convert_alpha()
        self.red_1 = pygame.image.load("assets/red_1.png").convert_alpha()
        self.playerRight = pygame.image.load(
            "assets/right.png").convert_alpha()
        self.playerRight_1 = pygame.image.load(
            "assets/right_1.png").convert_alpha()
        self.playerLeft = pygame.image.load(
            "assets/left.png").convert_alpha()
        self.playerLeft_1 = pygame.image.load(
            "assets/left_1.png").convert_alpha()
        self.playerdead = pygame.image.load(
            "assets/playerdead.png").convert_alpha()
        self.spring = pygame.image.load(
            "assets/spring.png").convert_alpha()
        self.spring_1 = pygame.image.load(
            "assets/spring_1.png").convert_alpha()
        self.monster = pygame.image.load(
            "assets/monster1.png").convert_alpha()
        self.monsterdead = pygame.image.load(
            "assets/monsterdead.png").convert_alpha()
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
        self.clock = pygame.time.Clock()
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
            x1 = random.randint(0, 700)
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

    def getCurrentFrame(self):
        """
            - No Param function
            - Returns
                - A screengrab of current pygame display
            - To be used by agent script to get current state
        """
        image_data = pygame.surfarray.array3d(pygame.display.get_surface())
        return image_data

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

    def getFeatures(self):
        features = {}
        features['player_x'] = self.playerx
        features['player_y'] = self.playery - \
            self.cameray  # Adjusted for camera
        features['player_xmovement'] = self.xmovement
        features['player_ymovement'] = - \
            self.jump if self.jump else self.gravity
        features['score'] = self.score

        # Nearby platforms within 400 pixels below the player
        nearby_platforms = []
        for p in self.platforms:
            platform_y = p[1] - self.cameray
            if features['player_y'] <= platform_y <= features['player_y'] + 400:
                nearby_platforms.append({
                    'x': p[0],
                    'y': platform_y,
                    'type': p[2],  # Platform type: 0-green, 1-blue, 2-red
                    'state': p[3]  # State for moving/broken platforms
                })
        # Sort platforms by y-coordinate
        nearby_platforms.sort(key=lambda p: p['y'])
        # Include up to 5 platforms
        features['platforms'] = nearby_platforms[:5]

        # Nearby springs
        nearby_springs = []
        for s in self.springs:
            spring_y = s[1] - self.cameray
            if features['player_y'] <= spring_y <= features['player_y'] + 400:
                nearby_springs.append({
                    'x': s[0],
                    'y': spring_y,
                    'state': s[2]  # Spring state
                })
        features['springs'] = nearby_springs

        # Nearby monsters
        nearby_monsters = []
        for m in self.monsters:
            monster_y = m[1] - self.cameray
            if features['player_y'] <= monster_y <= features['player_y'] + 400:
                nearby_monsters.append({
                    'x': m[0],
                    'y': monster_y,
                    'state': m[2]  # Monster state
                })
        features['monsters'] = nearby_monsters

        return features

    def playStep(self, actions):
        """
        - actions = ['ACTION_LEFT', 'NO_ACTION', 'ACTION_RIGHT']
        - Param:
            - actions: a list that contains three boolean values.
        - Returns:
            - reward: (int), terminal: (bool), score: (int), features: (dict)
        """
        last_cameray = self.cameray
        terminal = False
        reward = formulate_reward(self.reward_type, "DEFAULT")
        return_score = self.score

        pygame.display.flip()
        self.screen.fill((255, 255, 255))
        self.clock.tick(60)
        for event in pygame.event.get():
            if event.type == QUIT:
                sys.exit()
        if self.die == 1 or (self.playery - self.cameray > 900):
            features = self.getFeatures()
            return_score = self.gameReboot()
            terminal = True
            reward = formulate_reward(self.reward_type, "DEAD")
            print("terminated: Agent Died")
            return reward, terminal, return_score, features

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
                if (now_time - self.timer) > 100:
                    features = self.getFeatures()
                    return_score = self.gameReboot()
                    terminal = True
                    reward = formulate_reward(self.reward_type, "STUCK")
                    print("terminated: Agent stuck")
                    return reward, terminal, return_score, features

        self.updatePlayerByAction(actions)
        self.updatePlatforms()
        self.screen.blit(self.font.render(
            str(self.score), -1, (0, 0, 0)), (25, 25))
        pygame.display.flip()
        if not terminal:
            features = self.getFeatures()

        return reward, terminal, return_score, features

    def gameReboot(self):
        """
            - No Params
            - Returns:
                - old score before the reboot functions is called
            - resets all elements of the game
            - to be called when agent dies
        """
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
        x = 0
        while True:
            self.screen.fill((255, 255, 255))
            if (x == 0):
                print(self.getFeatures())
                x += 1
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