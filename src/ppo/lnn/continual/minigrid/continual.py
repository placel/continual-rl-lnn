import os
import random
import time
from dataclasses import dataclass
# https://ncps.readthedocs.io/en/latest/examples/atari_ppo.html
import gymnasium as gym
import minigrid # Import needed for MiniGrid environments to run
import numpy as np
import torch
import torch.backends
import torch.backends.cudnn
import torch.nn as nn
import torch.optim as optim
import tyro
import datetime
from collections import deque # used for buffer determining early stopping
from torch.utils.tensorboard import SummaryWriter

import sys
import os
# Needed to import utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../")))

from utils import evaluation
from utils.EarlyStopping import EarlyStopping
from utils import models # Custom class containing definitions for the main model implementations

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
    env_id: str = 'MiniGrid-Empty-5x5-v0'
    # Total timesteps allowed in the whole experiment
    total_timesteps: int = 500_000
    # Choose if the model should utilize CfC for Actor and Critic Head
    # If both are false, the Baseline model will be used
    cfc_actor: bool = False
    cfc_critic: bool = False
    # total_timesteps: int = 1_000_000
    # # Learning rate for the optimizer
    # learning_rate: float = 2.5e-4
    # learning_rate: float = 0.001
    # Learning rate for the optimizer
    learning_rate: float = 1.5e-4 # Better for CfC models
    # Number of environments for parallel game processing
    num_envs: int = 4
    # Total number of steps to run in each environment per policy rollout
    num_steps: int = 128
    # num_steps: int = 256
    # Size of the LNN hidden state
    hidden_state_size: int = 64
    #Toggles annealing for policy and value networks
    # anneal_lr: bool = True
    anneal_lr: bool = False
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
    ent_coef: float = 0.1 # Changed from 0.01 for testing
    # Coefficient of the value function
    vf_coef: float = 0.5
    # Maximum norm for gradient clipping
    max_grad_norm: float = 0.5
    # Target KL divergence threshol (don't know what this is)
    target_kl: float = None
    # Toggle early stopping  
    es_flag: bool = True

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
# passing in the env_name allows us to repeat this for each environment within the sequence 
def create_sync_envs(env_name):
    return gym.vector.SyncVectorEnv(
        [make_env(env_name, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )

# This is called at the end of an environment, whether triggered by early stopping or naturally
# Just saves the model, and prints a message 
def end_environment(cur_task_indx):
    print(f'Environment training time: {datetime.timedelta(seconds=time.time() - sequence_start_time)}')

    # Calculate average reward per environment
    rewards = evaluation.mean_reward(agent, sequence)
    # Assign rewards to the current task index
    perf_matrix[cur_task_indx] = rewards

    # Save this version of the model (in case of crashes later on)
    torch.save(agent.state_dict(), f'{os.path.dirname(__file__)}/models/{sequence[cur_task_indx]}.pt')

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
    writer = SummaryWriter(f'{os.path.dirname(__file__)}/runs/{run_name}')
    writer.add_text(
        'hyperparameters',
        '|param|value|\n|-|-|\n%s' % ('\n'.join([f'|{key}|{value}|' for key, value in vars(args).items()]))
    )

    # Store a list of all the names of the curriculum tasks 
    # The model will sequentially be trained on these tasks to aquire new skills
    # Start simple for now
    # sequence = [
    #     'MiniGrid-Empty-5x5-v0',
    #     'MiniGrid-Empty-Random-5x5-v0',
    #     # 'MiniGrid-LavaGapS5-v0',
    #     'MiniGrid-DoorKey-5x5-v0',
    #     'MiniGrid-Unlock-v0',
    #     # 'MiniGrid-MultiRoom-N2-S4-v0',
    #     # 'MiniGrid-Dynamic-Obstacles-5x5-v0'
    #     # 'MiniGrid-KeyCorridorS3R1-v0',
    #     # 'MiniGrid-DistShift1-v0', # Needs to be tested on DistShift2 for generalization
    #     # 'MiniGrid-LavaGapS7-v0', 
    # ]        

    sequence = [
        'MiniGrid-Empty-5x5-v0'
        # 'MiniGrid-Empty-8x8-v0',
        # 'MiniGrid-Empty-16x16-v0',
        # 'MiniGrid-FourRooms-v0'
        # 'MiniGrid-Unlock-v0',
        # 'MiniGrid-KeyCorridorS3R1-v0'
        # 'MiniGrid-DoorKey-5x5-v0'
    ]           

    sequence_patience = [
        5
        # 5,
        # 6,
        # 12
        # 15
    ]                             

    # Initialize all seeds to be the same
    # Try not to modify
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    # Assign to GPU (or CPU if not connected)
    device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')
    print(f'Device: {device}')

    # Create the first batch of environments with the first environment listed in the curriculum
    # This passes 'MiniGrid-Empty-5x5-v0' to the function, returning 4 environments synced to be able to run parallel
    # **
    # Most environments within MiniGrid share the same input and output space (image as input with 7 actions)
    # so this running this on-demand doesn't cause any shape issues. Environments with larger image inputs can be 
    # shrunk to fit the standard, or vice versa
    # ** 
    envs = create_sync_envs(sequence[0])

    # Confirm the action space is Discrete (left, right, etc.), no Continuous 
    assert isinstance(envs.single_action_space, gym.spaces.Discrete) # Only disscrete action space is supported here

    # Create the agent
    # initialize the outputs (actions to take)
    # vocab length + 2 to incorporate PAD and UKN tokens
    hidden_dim = 64
    config = {
        'action_space': envs.single_action_space.n,
        'hidden_dim': hidden_dim,
        'actor_cfc': args.cfc_actor,
        'critic_cfc': args.cfc_critic
    }
    agent = models.Agent(config).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # Collect baseline model performance (performance with no training)
    # This is needed to compute Forward Transfer
    # Returns mean reward for each task
    baselines = []
    print('Collecting random baselines...')
    # baselines = evaluation.mean_reward(agent, sequence)

    # Alogirthm Logic
    #Storage
    # Reshape the observation space to match the shape of the image (3, 7, 7) 
    image_shape = envs.single_observation_space['image'].shape
    permuted_image = (image_shape[2], image_shape[0], image_shape[1])

    # Keep track of these per rollout. Ex. store only num_steps(128) rows with num_envs(4) columns for values
    # These will reset after every rollout. Each rollout has num_steps(128) collection points
    # so num_steps(128) steps of each var below will be stored, then replace each rollout
    obs = torch.zeros((args.num_steps, args.num_envs) + permuted_image).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    actor_h_state = torch.zeros((args.num_steps, args.num_envs, args.hidden_state_size)).to(device)

    # Try not to modify
    # Start game
    global_step = 0

    # Track total time of training
    master_start_time = time.time()
    es_triggered = False
    # Keep track of average rewards per task 
    perf_matrix = np.zeros((len(sequence), len(sequence)))
    early_stopping_interval = 10
    # Iterate over each environment within the curriculum and train the model
    for indx, cur_env in enumerate(sequence):
        print(f'Current environment: {cur_env}')

        # Keep track of how long each env lasts, it should get longer with each
        sequence_start_time = time.time()
        
        # initialize the EarlyStopping class at start of every episode to ensure no overlap between task performance
        es_monitor = EarlyStopping(patience=sequence_patience[indx], min_delta=0.01, mode='max', verbose=True, restore_weights_flag=True) 

        # The first batch is created before this loop. If not the first batch, 
        # create the next batch of environments associated with the current curriculum environment
        if indx > 0:
            # Call the function to make envs for new curriculum
            envs = create_sync_envs(env_name=cur_env)

        # Reset the environment (initialize for new envs)
        next_obs, _ = envs.reset(seed=args.seed)

        # Permutate the observations to be (batch_size=4 (envs), channels=3, height, width)
        # Also, normalize the image to help the CNN layers
        next_obs['image'] = (torch.tensor(next_obs['image'], dtype=torch.float32, device=device).permute(0, 3, 1, 2) / 255.0)
        next_done = torch.zeros(args.num_envs).to(device)
        # Iterate over a rollout
        for iteration in range(1, args.num_iterations + 1):
            # 'Anneal' the learning rate if enabled
            # Adjust the learning rate as the model trains
            if args.anneal_lr:
                frac = 1.0 - (iteration - 1.0) / args.num_iterations
                lrnow = frac * args.learning_rate
                optimizer.param_groups[0]['lr'] = lrnow

            # Initialize hidden states to 0 at the start of each episode
            actor_state = torch.Tensor(torch.zeros((args.num_envs, args.hidden_state_size)).to(device))

            # Perform a collection of states in batches of usually 128
            for step in range(0, args.num_steps):
                global_step += args.num_envs # account for each step in every env
                obs[step] = next_obs['image']
                dones[step] = next_done

                # Store the current steps states for later retrieval during update training
                # Intialized to zero but update over time
                actor_h_state[step] = actor_state[0]

                # Algorithm logic: action logic
                # During the collection of data, we pass the state to the model
                # and sample a random action and store it. Data collection is stochastic
                # Learning occurs after data is collected and epochs are run
                with torch.no_grad():
                    # Get the action the actor takes. And get the value the critic assigns said action
                    action, logprob, _, value, new_actor_state = agent.get_action_and_value(next_obs['image'], actor_state)

                    # Update the actor_states with new states, resetting hidden states where the next_state is terminal
                    new_actor_state = new_actor_state * (1.0 - next_done.unsqueeze(1))
                    actor_state = new_actor_state

                    values[step] = value.flatten()
                actions[step] = action
                logprobs[step] = logprob

                # Try note to modify: play the game and log 
                next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
                next_done = np.logical_or(terminations, truncations)
                rewards[step] = torch.tensor(reward).to(device)

                # Once again we need to extract the image from the obs and permute for processing
                # Normalize it too
                next_obs['image'] = (torch.tensor(next_obs['image'], dtype=torch.float32, device=device).permute(0, 3, 1, 2) / 255.0)
                next_done = torch.Tensor(next_done).to(device)

                # Log the rewards after each episode of every environment ends
                if "episode" in infos and "_episode" in infos:
                    for i, finished in enumerate(infos["_episode"]):
                        if finished:
                            print(f"global_step={global_step}, env={i}, episodic_return={infos['episode']['r'][i]}", flush=True)
                            writer.add_scalar("charts/episodic_return", infos['episode']['r'][i], global_step)
                            writer.add_scalar("charts/episodic_length", infos['episode']['l'][i], global_step)

            # Calculate advantages for future usage in algorithm 
            with torch.no_grad():
                # Get the critics thoughts on the value of the next state (for all envs)
                next_value = agent.get_value(next_obs['image']).reshape(1, -1)

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
            b_h_states = actor_h_state.reshape(-1, args.hidden_state_size)

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

                    # Extract states for this mini-batch
                    # and pass to model as one array
                    mb_states = torch.Tensor(b_h_states[mb_inds])

                    # print(f'MB Tokens: ', b_mission_tokens[mb_inds])
                    # Forward pass (with grad) the minibatch through to the model to get values associated with the provided action
                    _, new_log_prob, entropy, new_value, _ = agent.get_action_and_value(b_obs[mb_inds], states=mb_states, action=b_actions.long()[mb_inds])
                    
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
            
            # EARLY STOPPING
            # If early stopping is enabled, check if model training on current task should end
            # Ensure the model trains for at least k iterations before checking if early stopping should trigger
            # And only check early_stopping every k iterations
            if args.es_flag and (iteration > early_stopping_interval) and (iteration % early_stopping_interval == 0):
                
                # Evaluate the mode on the current environment to check if early stopping should trigger
                es_mean = evaluation.mean_reward(agent=agent, envs=[cur_env], episodes=5) # Increase episodes for more reliable mean_reward
                es_triggered = es_monitor.update(es_mean[0], model=agent)

                if es_triggered:
                    es_monitor.restore_weights(agent)
                    end_environment(indx)
                    break

        # Save the current version of the model for testing while curriculum continues (and reloding from previous point)
        if not es_triggered:
            end_environment(indx)
        
        # Reset early stopping flag for next env
        es_triggered = False
    
    print('Computing Metrics...')
    # fwt = evaluation.compute_fwt(perf_matrix, baselines)
    bwt = evaluation.compute_bwt(perf_matrix)
    # Take the mean of all averaged rewards to get a singular average
    mean_reward = np.mean(evaluation.mean_reward(agent, sequence))

    evaluation.plot_perf_matrix(perf_matrix, save_path=f'{os.path.dirname(__file__)}/metrics/performance_matrix.png')
    # evaluation.plot_metrics(fwt, bwt, mean_reward, save_path=f'{os.path.dirname(__file__)}/metrics/metrics.png')

    print(f'Final training time: {datetime.timedelta(seconds=time.time() - master_start_time)}')
    envs.close()
    writer.close()
    torch.save(agent.state_dict(), f'{os.path.dirname(__file__)}/models/final_model.pt')