import os
import random
import time
from dataclasses import dataclass
# https://ncps.readthedocs.io/en/latest/examples/atari_ppo.html
import gymnasium as gym
import minigrid
# import minigrid.envs
import numpy as np
import torch
import torch.backends
import torch.backends.cudnn
import torch.nn as nn
import torch.optim as optim
import tyro
import datetime
import ModelVariations as models # Custom CfC class containing definitions for Actor and Critic CfC implementations
from collections import deque # used for buffer determining early stopping

# Import the Closed-form Continuous model 
# This is the liquid neural net we'll use
# Only use the Fully Connected wirings firs as it's the most conceptually similar
from ncps.torch import CfC

from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

@dataclass
class Args:
    # Experiment Name
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    # Seed of the experiemnt (for reproduction)
    seed: int = 1
    # if toggled, `torch.backends.cudnn.deterministic=False`
    torch_deterministic: bool = True
    # Toggles usage cuda, defaults to true
    cuda: bool = True
    # Toggles tracking with website Weights & Biases (look into this)
    track: bool = False
    # Information for Weights & Biases (if used)
    wandb_project_name: str = None
    wandb_entity: str = None
    # Toggling will capture video
    capture_video: bool = False

    # ALGORITHM ARGS
    # Name of the environment
    env_id: str = 'MiniGrid-DoorKey-5x5-v0'
    # Total timesteps allowed in the whole experiment
    # total_timesteps: int = 500_000
    total_timesteps: int = 1_000_000
    # # Learning rate for the optimizer
    # learning_rate: float = 2.5e-4
    learning_rate: float = 0.001
    # learning_rate: float = 0.001
    # Learning rate for the optimizer
    # learning_rate: float = 1e-4 # Better for CfC models
    # Number of environments for parallel game processing
    num_envs: int = 4
    # Total number of steps to run in each environment per policy rollout
    num_steps: int = 128
    # num_steps: int = 256
    # Size of the LNN hidden state
    hidden_state_size: int = 128
    #Toggles annealing for policy and value networks
    anneal_lr: bool = True
    # Gamma value
    gamma: float = 0.99
    # Lambda value for the General Advantage Estimation
    gae_lambda: float = 0.95
    # Number of mini-batches
    num_minibatches: int = 4
    # Epochs to update the policy at a time (generally 4 in a row)
    update_epochs: int = 4
    # Toggle advantages normalization
    norm_adv: bool = True
    # Clip epsilon
    # clip_coef: float = 0.2
    clip_coef: float = 0.2
    # Toggles usage of clipped loss for the value function
    clip_vloss: bool = True
    # coefficient of the entropy (encourages exploration)
    ent_coef: float = 0.03 # Changed from 0.01 for testing
    # Coefficient of the value function
    vf_coef: float = 0.5
    # Maximum norm for gradient clipping
    max_grad_norm: float = 0.5
    # Target KL divergence threshol (don't know what this is)
    target_kl: float = None
    # Length of reward buffer (used for curriculum learning)
    reward_buffer_length: int = num_steps * 2 # Twice the timestep (for now) for security it did learn the task
    # Early stopping mean requirement.
    es_mean: float = 0.95
    # Early stopping standard deviation requirement.
    es_std: float = 0.05 

    # TO be filled in runtime
    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0

# Make the environment
def make_env(env_id, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode='rgb_array')
            env = gym.wrappers.RecordVideo(env, f'videos/{run_name}')
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env
    
    return thunk

# Create the environments for parallelization
# passing in the env_name allows us to repeat this for each environment within the curriculum 
def create_sync_envs(env_name):
    return gym.vector.SyncVectorEnv(
        [make_env(env_name, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )

# Custom layer weight initialization
# Orthogonal weight setting ensures no neurons have any sort of correlation
# This is particularly important in reinforcement learning whent he dataset is small
# and exploration is required. Any inherent correlation could influence the results
# We also set all biases to 0
# def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
#     torch.nn.init.orthogonal_(layer.weight, std)
#     torch.nn.init.constant_(layer.bias, bias_const)
#     return layer

# This is called at the end of an environment, whether triggered by early stopping or naturally
# Just saves the model, and prints a message 
def end_environment(reward_buffer, curriculum_start_time):
    print(f'Curriculum training time: {datetime.timedelta(seconds=time.time() - curriculum_start_time)}')
    # Save the current version of the model for testing while curriculum continues (and reloding from previous point)
    mean_reward = round(np.mean(reward_buffer), 2)
    filename = f'{curriculum_env}_R-{round(mean_reward, 2)}'.replace('.', '_') + '.pt'
    torch.save(agent.state_dict(), f'./src/ppo/baseline/curriculum/minigrid/models/{filename}')

    # Empty the reward buffer to prevent cross-usage between environments.
    # Some harder environments don't 
    reward_buffer.clear()

# Check the reward_buffer to see if the model should stop training early
# If the model is already performing well, we don't want to train on the current rollout
# so we break out of the current rollout and move onto next environment
def early_stopping(reward_buffer):

    if len(reward_buffer) >= args.reward_buffer_length:
        mean = np.mean(reward_buffer)
        std = np.std(reward_buffer)
        if mean >= args.es_mean and std <= args.es_std:
            print('Early stopping triggered...')
            
            # Return True to signal Early Stopping, and mean to log the mean reward of model
            return True 
    
    # Otherwise return False to prevent early_stopping
    return False

# Extracted from MiniGrid GitHub envs. Needs to be hardcoded as there's no way to get all words used in MiniGrid from one location
# Each Mission has syntax ('get a', 'fetch a', etc.), a colour, and an object ('key', 'ball', etc.)
# Any updates or if incorporating BabyAI will need this list to change first to ensure the vocab is consistent across envs
# In case updates are needed, find the new words and give the list to a chatbot to provide a unique list of words instead of doing it manually
unique_words = [
    'get', 'a', 'go', 'fetch', 'you', 'must', 'to', 'the', 'matching',
    'object', 'at', 'end', 'of', 'hallway', 'traverse', 'rooms', 'goal',
    'put', 'near', 'and', 'open', 'red', 'door', 'then', 'blue', 'pick',
    'up', 'green', 'grey', 'purple', 'yellow', 'box', 'key', 'ball', 'square',
    'use', 'it', 'next', 'first', 'second', 'third'
]

PAD_IDX = 0
UNK_IDX = 1
# Start vocab indexing from 2 to leave 0 for PAD and 1 for UNK
word_dict = {word: idx + 2 for idx, word in enumerate(unique_words)}

# def process_mission(missions, max_mission_length=12):

#     batch_tokens = []
#     for m in missions:  # Iterate over each mission in the batch
#         words = m.lower().split()
        
#         # Truncate the text if exceeds maximum specified
#         if len(words) > max_mission_length:
#             words = words[:max_mission_length]

#         # Tokenize (use UNK_IDX if not in vocab)
#         tokens = [word_dict.get(word, UNK_IDX) for word in words]

#         # Applly padding where needed
#         if len(tokens) < max_mission_length:
#             tokens += [PAD_IDX] * (max_mission_length - len(tokens))

#         batch_tokens.append(tokens)

#     return torch.tensor(batch_tokens, dtype=torch.long, device=device)

def process_mission(missions, max_mission_length=12, device='cpu'):  # Increased from 9
    """Fixed mission processing with better debugging"""
    batch_tokens = []
    
    # print(missions)
    # Debug: Print missions to see what we're getting
    if isinstance(missions, (list, tuple)) and len(missions) > 0:
        # print(f"Sample mission: '{missions[0]}'")
        pass

    for m in missions:
        words = m.lower().split()
        
        # Debug: Check for unknown words
        unknown_words = [w for w in words if w not in word_dict]
        if unknown_words:
            print(f"Unknown words in mission '{m}': {unknown_words}")
        
        # Truncate if too long
        if len(words) > max_mission_length:
            words = words[:max_mission_length]
            print(f"Warning: Mission truncated from {len(m.split())} to {max_mission_length} words")

        # Tokenize with better UNK handling
        tokens = [word_dict.get(word, UNK_IDX) for word in words]

        # Apply padding
        if len(tokens) < max_mission_length:
            tokens += [PAD_IDX] * (max_mission_length - len(tokens))

        batch_tokens.append(tokens)

    return torch.tensor(batch_tokens, dtype=torch.long).to(device=device)

# A batch of tuples is passed. Extract the batch size, perform mission processing, then convert back into original batch
# def process_mission(mission, max_mission_length=9):

#     batch_size = len(mission)

#     # Split the mission by space into a list of individual words
#     words = mission[0].lower().split()

#     # Apply max_mission_length. I doubt any mission is greater than 9, but we'll see
#     if len(words) > max_mission_length:
#         words = words[:max_mission_length]

#     # Convert words into a list of tokens for model processing
#     tokens = [word_dict[word] for word in words]

#     # Apply padding if needed. If tokens is less than max_mission_length
#     if len(tokens) < max_mission_length:
#         tokens += [0] * (max_mission_length - len(tokens)) # Just append a 0 (max_mission_length - len(tokens)) amount of times

#     # Rebuild the list back into batch_size x tokens
#     tokens = [tokens for _ in range(batch_size)]
#     tokens = torch.tensor(tokens).to(device)

#     return tokens

if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size

    print(f'Batchsize: {args.batch_size}')
    print(f'MiniBatch: {args.minibatch_size}')
    print(f'Num Iterations: {args.num_iterations}')
    run_name = f'{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}'
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    
    # Setup logging 
    writer = SummaryWriter(f'./src/ppo/baseline/curriculum/minigrid/runs/{run_name}')
    writer.add_text(
        'hyperparameters',
        '|param|value|\n|-|-|\n%s' % ('\n'.join([f'|{key}|{value}|' for key, value in vars(args).items()]))
    )

    # Store a list of all the names of the curriculum tasks 
    # The model will sequentially be trained on these tasks to aquire new skills
    # Start simple for now
    curriculum = [
        # 'MiniGrid-Empty-5x5-v0',
        # 'MiniGrid-Empty-Random-5x5-v0',
        # 'MiniGrid-DoorKey-5x5-v0',
        # 'MiniGrid-MultiRoom-N2-S4-v0',
        # 'MiniGrid-FourRooms-v0',
        # 'MiniGrid-Unlock-v0',
        'MiniGrid-Fetch-5x5-N2-v0'
        # 'MiniGrid-KeyCorridorS3R1-v0',
        # 'MiniGrid-DistShift1-v0', # Needs to be tested on DistShift2 for generalization
        # 'MiniGrid-LavaGapS7-v0', 
    ]        

    # Start with 100 for now, may increase later for higher accuracy
    # This will store the rewards of the last x environments
    # Using this, we may determine if we want to stop training early
    # and move onto the next curriculum task, saving time and energy
    reward_buffer = deque(maxlen=args.reward_buffer_length)                                               

    # Initialize all seeds to be the same
    # Try not to modify
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    # Assign to GPU (or CPU if not connected)
    device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')
    print(device)

    # Create the first batch of environments with the first environment listed in the curriculum
    # This passes 'MiniGrid-Empty-5x5-v0' to the function, returning 4 environments synced to be able to run parallel
    # **
    # Most environments within MiniGrid share the same input and output space (image as input with 7 actions)
    # so this running this on-demand doesn't cause any shape issues. Environments with larger image inputs can be 
    # shrunk to fit the standard, or vice versa
    # ** 
    envs = create_sync_envs(curriculum[0])

    # Confirm the action space is Discrete (left, right, etc.), no Continuous 
    assert isinstance(envs.single_action_space, gym.spaces.Discrete) # Only disscrete action space is supported here

    # Create the agent
    # initialize the outputs (actions to take)
    # vocab length + 2 to incorporate PAD and UKN tokens
    agent = models.Agent(envs.single_action_space.n, vocab_size=len(word_dict) +2, hidden_dim=64, word_embedding_dim=32, text_embedding_dim=128, actor_cfc=False, critic_cfc=False).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # Alogirthm Logic
    #Storage
    # Reshape the observation space to match the shape of the image (3, 7, 7) 
    image_shape = envs.single_observation_space['image'].shape
    permuted_image = (image_shape[2], image_shape[0], image_shape[1])

    # Max token length of each mission
    max_mission_length = 12

    # Keep track of these per rollout. Ex. store only num_steps(128) rows with num_envs(4) columns for values
    # These will reset after every rollout. Each rollout has num_steps(128) collection points
    # so num_steps(128) steps of each var below will be stored, then replace each rollout
    obs = torch.zeros((args.num_steps, args.num_envs) + permuted_image).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    mission_tokens = torch.zeros((args.num_steps, args.num_envs, max_mission_length), dtype=torch.int).to(device)

    # Try not to modify
    # Start game
    global_step = 0

    # Track total time of training
    master_start_time = time.time()
    es_triggered = False
    # Iterate over each environment within the curriculum and train the model
    for indx, curriculum_env in enumerate(curriculum):
        print(f'Current curriculum environment: {curriculum_env}')

        # Keep track of how long each curriculum lasts, it should get longer with each
        curriculum_start_time = time.time()

        # The model stuggles with FourRooms; increase total timestep and exploration (entropy) for it alone
        if 'FourRooms' in curriculum_env:
            args.total_timesteps = 750_000
            args.ent_coef = 0.25
        # else:
        #     args.total_timesteps = 500_000
        #     args.ent_coef = 0.01

        # The first batch is created before this loop. If not the first batch, 
        # create the next batch of environments associated with the current curriculum environment
        if indx > 0:
            # Call the function to make envs for new curriculum
            envs = create_sync_envs(env_name=curriculum_env)

        # Reset the environment (initialize for new envs)
        next_obs, _ = envs.reset(seed=args.seed)

        # Permutate the observations to be (batch_size=4 (envs), channels=3, height, width)
        # Also, normalize the image to help the CNN layers
        next_obs['image'] = (
            # torch.tensor(next_obs['image'], dtype=torch.float32, device=device).permute(0, 3, 1, 2) / 255.0
            torch.tensor(np.array(next_obs['image']), dtype=torch.float32, device=device).permute(0, 3, 1, 2)
        )
        next_done = torch.zeros(args.num_envs).to(device)

        # Iterate over a rollout
        for iteration in range(1, args.num_iterations + 1):
            # 'Anneal' the learning rate if enabled
            # Adjust the learning rate as the model trains
            if args.anneal_lr:
                frac = 1.0 - (iteration - 1.0) / args.num_iterations
                lrnow = frac * args.learning_rate
                optimizer.param_groups[0]['lr'] = lrnow

            # Perform a collection of states in batches of usually 128
            for step in range(0, args.num_steps):
                global_step += args.num_envs # account for each step in every env
                obs[step] = next_obs['image']
                dones[step] = next_done

                # Pass the mission of the current environment to be tokenized for the embedding layer
                tokenized_mission = process_mission(next_obs['mission'], max_mission_length, device)
                mission_tokens[step] = tokenized_mission

                # Algorithm logic: action logic
                # During the collection of data, we pass the state to the model
                # and sample a random action and store it. Data collection is stochastic
                # Learning occurs after data is collected and epochs are run
                with torch.no_grad():
                    # Get the action the actor takes. And get the value the critic assigns said action
                    action, logprob, _, value = agent.get_action_and_value(next_obs['image'], tokenized_mission)

                    values[step] = value.flatten()
                actions[step] = action
                logprobs[step] = logprob

                # Try note to modify: play the game and log 
                next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
                reward = np.maximum(reward, 0.0)
                next_done = np.logical_or(terminations, truncations)
                rewards[step] = torch.tensor(reward).to(device)

                # Once again we need to extract the image from the obs and permute for processing
                next_obs['image'] = (
                    torch.tensor(np.array(next_obs['image']), dtype=torch.float32, device=device).permute(0, 3, 1, 2)
                    # torch.tensor(next_obs['image'], dtype=torch.float32, device=device).permute(0, 3, 1, 2) / 255.0
                )
                next_done = torch.Tensor(next_done).to(device)

                # Log the rewards after each episode of every environment ends
                if "episode" in infos and "_episode" in infos:
                    for i, finished in enumerate(infos["_episode"]):
                        if finished:
                            print(f"global_step={global_step}, env={i}, episodic_return={infos['episode']['r'][i]}", flush=True)
                            writer.add_scalar("charts/episodic_return", infos['episode']['r'][i], global_step)
                            writer.add_scalar("charts/episodic_length", infos['episode']['l'][i], global_step)
                            
                            # Append the latest environment reward to the reward_buffer for early stopping criteria
                            reward_buffer.append(infos['episode']['r'][i])
            
            # Check if early stopping should kick-in
            es_triggered = early_stopping(reward_buffer)
            if es_triggered:
                # Process the end of the environment
                end_environment(reward_buffer, curriculum_start_time)
                break

            # Calculate advantages for future usage in algorithm 
            with torch.no_grad():
                # Get the critics thoughts on the value of the next state (for all envs)
                next_value = agent.get_value(next_obs['image'], mission_tokens[-1]).reshape(1, -1)

                # allocate space
                advantages = torch.zeros_like(rewards).to(device)
                
                # Holds the running Generalized Advantage Estimation (GAE) lambda accumulation
                last_gae_lam = 0

                # Loop backwards through the entire rollout to calculate advantage for each state
                for t in reversed(range(args.num_steps)):
                    # Get the values of the next state. We actually have these values
                    # and we go in reverse to grab the value of the next state 
                    # (values are still generated by the critic though)
                    # This just grabs the value of the next state depending on if it's terminal or not
                    if t == args.num_steps - 1:
                        next_non_terminal = 1.0 - next_done
                        next_values = next_value
                    else:
                        next_non_terminal = 1.0 - dones[t + 1]
                        next_values = values[t + 1]

                    # The core calculation behind GAE that gives us the advantage value
                    # reward (reward from the state) + (discount * next_state_value (generated by critic)) - current_value (generated by critic)
                    # This gives us delta, which is how different the resulting state is vs what we expected
                    delta = rewards[t] + args.gamma * next_values * next_non_terminal - values[t]

                    # Recursively calculate advantage backwards through time
                    # This gives us an increasing advantage as we move backwards, and a decreasing advantage as the model moves forward
                    # Advantage ​= δt​ + (γλ)δt+1 ​+(γλ)2δt+2 ​+ (γλ)3δt+3 ​+ …
                    # Gamma controls how much we care about the future (discount)
                    # lambda controls how much we smooth by including multiple steps. The bigger lambda is, the more steps are included
                    # The advantage will shrink from left to right as we care less and less about future rewards, but still want to consider them
                    # this is done recursiveley though so it appears to grow
                    # gamma * lambda will incorporate the decay needed
                    # Bottom line: lambda is just a weight applied to each delta (difference in values) applied through time
                    # the earlier time steps will be more impactful than later steps. Lambda is similar to Gamma in a way
                    advantages[t] = last_gae_lam = delta + args.gamma * args.gae_lambda * next_non_terminal * last_gae_lam
                
                returns = advantages + values

            # Flatten batch
            b_obs = obs.reshape((-1,) + permuted_image)
            b_logprobs = logprobs.reshape(-1)
            b_actions = actions.reshape((-1,), envs.single_action_space.shape)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values.reshape(-1)
            b_mission_tokens = mission_tokens.reshape(-1, max_mission_length)

            # print(b_mission_tokens.shape)
            # print(f'OBS: {b_obs.shape}')


            # Optimize the policy (actor) and value (critic) networks
            b_inds = np.arange(args.batch_size)
            clip_fracs = []

            # This is how many times we perfrom backprop on the model after batch collection (usually 4 times)
            for epoch in range(args.update_epochs):
                # Shuffle the batch so it's not sequential
                np.random.shuffle(b_inds)
                # Iterate over minibatches for incremental learning
                for start in range(0, args.batch_size, args.minibatch_size):
                    end = start + args.minibatch_size
                    mb_inds = b_inds[start:end]

                    # print(f'MB Tokens: ', b_mission_tokens[mb_inds])
                    # Forward pass (with grad) the minibatch through to the model to get values associated with the provided action
                    _, new_log_prob, entropy, new_value = agent.get_action_and_value(b_obs[mb_inds], b_mission_tokens[mb_inds], action=b_actions.long()[mb_inds])
                    
                    # Ratio of the probablity of the new policy vs the old policy for taking provided action
                    # This is a key component in the PPO equation
                    log_ratio = new_log_prob - b_logprobs[mb_inds]
                    ratio = log_ratio.exp()

                    with torch.no_grad():
                        # Copied from cleanrl directly (I don't know what this is doing)
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        old_approx_kl = (-log_ratio).mean()
                        approx_kl = ((ratio - 1) - log_ratio).mean()
                        clip_fracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                    # Store minibatch advantages
                    mb_advantages = b_advantages[mb_inds]

                    # Not sure i'll use this
                    if args.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                    # Calculate the Policy loss
                    pg_loss1 = -mb_advantages * ratio # In the equation
                    # Here's where any potential clipping would take place
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Calculate the value loss
                    new_value = new_value.view(-1)
                    # Clips the v_loss (enabled by default)
                    if args.clip_vloss:
                        v_loss_unclipped = (new_value - b_returns[mb_inds]) ** 2

                        v_clipped = b_values[mb_inds] + torch.clamp(
                            new_value - b_values[mb_inds],
                            -args.clip_coef,
                            args.clip_coef
                        )

                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()

                    else:
                        v_loss = 0.5 * ((new_value - b_returns[mb_inds]) ** 2).mean()

                    entropy_loss = entropy.mean()
                    # what is this; find out
                    loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                    # Apply learning to agent
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                    optimizer.step()

                if args.target_kl is not None and approx_kl > args.target_kl:
                    break

            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)

            # Also, what is this; find out
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

            # Copied for logging
            # TRY NOT TO MODIFY: record rewards for plotting purposes
            writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
            writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
            writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
            writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
            writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
            writer.add_scalar("losses/clipfrac", np.mean(clip_fracs), global_step)
            writer.add_scalar("losses/explained_variance", explained_var, global_step)
            print("SPS:", int(global_step / (time.time() - master_start_time)))
            writer.add_scalar("charts/SPS", int(global_step / (time.time() - master_start_time)), global_step)
       
        # Save the current version of the model for testing while curriculum continues (and reloding from previous point)
        if not es_triggered:
            end_environment(reward_buffer, curriculum_start_time)
        
        # Reset early stopping flag for next env
        es_triggered = False

    print(f'Final training time: {datetime.timedelta(seconds=time.time() - master_start_time)}')
    envs.close()
    writer.close()
    torch.save(agent.state_dict(), f'./src/ppo/baseline/curriculum/minigrid/models/final_curriculum_model.pt')