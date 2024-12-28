import os
import math


def calculate_reward(reward_scheme, event, spring_touched=False, monster_touched=False, score=0):
    reward = 0.0

    # Define reward mappings
    alive_rewards = {
        1: -1,
        2: 0,
        3: 0
    }

    dead_stuck_reward = {
        1: -2,
        2: -2,
        3: -20
    }

    # Avoid math domain error
    scored_base_reward = 3 + math.log(score) if score > 0 else 3

    if event == "ALIVE":
        reward = alive_rewards.get(reward_scheme, 0)
    elif event in {"DEAD", "STUCK"}:
        reward = dead_stuck_reward.get(reward_scheme, 0)
    elif event == "SCORED":
        reward = scored_base_reward
        if spring_touched:
            reward += 3
        if monster_touched:
            reward -= 4
    else:
        # Handle unexpected events
        raise ValueError(f"Unhandled event type: {event}")

    return reward
