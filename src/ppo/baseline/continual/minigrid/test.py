import gymnasium as gym
import minigrid
env = gym.make("MiniGrid-DoorKey-5x5-v0")
import numpy as np

import inspect
print(inspect.getsource(env.unwrapped._reward.__func__))
obs, _ = env.reset()
done = False
while not done:
    action = env.action_space.sample()
    obs, reward, term, truncated, info = env.step(action)
    done = np.logical_or(term, truncated)
    print("Step reward:", reward)