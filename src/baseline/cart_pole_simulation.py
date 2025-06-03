# All code sourced thought the link below
# https://aleksandarhaber.com/cart-pole-control-environment-in-openai-gym-gymnasium-introduction-to-openai-gym/

# Gynasium instead of gym as it is currently updated by the community
import gymnasium as gym
import numpy as np
import time

env=gym.make('CartPole-v1', render_mode='human')

# state vector values
# [cart_position, cart_velocity, angle_of_rotation, angular_velocity]
(state, _) = env.reset()

# Draw the screen
env.render()

# Push the cart left
env.step(0)

#Add time to observe
time.sleep(3)

episode_number = 10_000
time_steps=100

# Sample loop:
# take random actions and loop over to visualize what's going on
# just visualizing a random model, no learning of course
for episode_index in range(episode_number):
    initial_state = env.reset()
    print(episode_index)
    env.render()
    observations = []
    for time_index in range(time_steps):
        print(time_index)
        random_action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(random_action)
        observations.append(observation)
        
        # Pause the frame so we can see
        time.sleep(0.01)

        # If maximum angle (+12 or -12) is reached, the episode ends
        if terminated:
            time.sleep(1)
            break

env.close()