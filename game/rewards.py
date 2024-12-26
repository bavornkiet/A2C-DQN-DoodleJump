import os
import math


def formulate_reward(reward_type, reason, spring_touch=False, monster_touch=False, score=0):
    reward = None

    if reason == "ALIVE":
        reward = -1
    if reason == "DEAD":
        reward = -2
    if reason == "STUCK":
        reward = -2
    if reason == "SCORED":
        reward = 3 + math.log(score)
        if spring_touch:
            reward += 3
        if monster_touch:
            reward -= 4

    return reward
