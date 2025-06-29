# Load the model
import torch
import time
import gymnasium as gym
from CartPole_DQN_model import CartPoleModel

device = torch.device(
    'cuda' if torch.cuda.is_available() else
    'cpu'
)

print(f'Using {device}')

env = gym.make('CartPole-v1', render_mode='human')

# Later on add velocity and the other one for more complex changes
def create_task(length, gravity):

    temp_env = gym.make('CartPole-v1', render_mode='human')
    
    # Update the length of pole (randomize later)
    temp_env.unwrapped.length = length
    temp_env.unwrapped.gravity = temp_env.unwrapped.gravity * gravity

    return temp_env
    

env = create_task(1.5, 1.1)

# Store the number of possible actions from the Gym env
n_actions = env.action_space.n

# Get the number of state observations
state, info = env.reset()
n_observations = len(state)

n_nodes = 256
policy_net = CartPoleModel(n_observations, n_actions, n_nodes)

policy_net.load_state_dict(torch.load('src/baseline/cartpole/models/continual_policy_model.pt'))
policy_net.to(device=device)

state, _ = env.reset()

done = False
while not done:
    env.render()

    # time.sleep(2)
    state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    
    # Any pre-processing techniques need to be applied befor ewe use it
    with torch.no_grad():
        # INference the model and take the max Q-Value action
        action = policy_net(state_tensor).argmax(dim=1).item()

    # Apply action to env, and store the new state. Then loop again
    # If state is terminated, done will be updated and loop will end
    state, reward, done, _, _= env.step(action)
    
env.close() 