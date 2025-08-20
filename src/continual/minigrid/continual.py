import os
import random
import time
import json
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
from typing import Optional

import csv
import pathlib
import sys
import os
# Needed to import utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

# Custom class imports
from utils import evaluation
from utils.EarlyStopping import EarlyStopping
from utils import models # Custom class containing definitions for the main model implementations

@dataclass
class Args:
    # Experiment Name
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    # Trial number (for HPO)
    trial_id: str = None
    # Accepts a string and will create a txt file to define the current experiment
    exp_def: str = "Testing baseline model" 
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
    # Enable CfC Actor
    cfc_actor: bool = False
    # Enable CfC Critic
    cfc_critic: bool = False
    # Toggle the usage of an LSTM in the Baseline model
    use_lstm: bool = False
    # Size of hidden dimensions 
    hidden_dim: int = 64
    # Size of the reccurent states.  
    hidden_state_dim: Optional[int] = 128
    # Toggle early stopping (es)
    es_flag: bool = False
    # Restore weights with Early Stopping (generally False)
    es_restore_weights: bool = False
    # Toggle Elastic Weight Consolidation (EWC)
    ewc: bool = False
    # EWC Weight (hyperparameter)
    ewc_weight: float = 0.0
    # Load pretrained model
    pretrained_model: str = None

    # ALGORITHM ARGS
    # Name of the environment
    env_id: str = 'MiniGrid-Empty-5x5-v0'
    # Total timesteps allowed in the whole experiment
    total_timesteps: int = 250_000
    # Learning rate for the optimizer
    learning_rate: float = 1.5e-4 # Better for CfC models
    # Number of environments for parallel game processing
    num_envs: int = 4
    # Total number of steps to run in each environment per policy rollout
    num_steps: int = 128
    # num_steps: int = 256
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
    ent_coef: float = 0.01 # Changed from 0.01 for testing
    # Coefficient of the value function
    vf_coef: float = 0.5
    # Maximum norm for gradient clipping
    max_grad_norm: float = 0.5
    # Target KL divergence threshol (don't know what this is)
    target_kl: float = None

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
# Just saves the model, computes perf_matrix values, and prints a message 
def end_environment(cur_task_indx, es_triggered=False):
    print(f'Environment training time: {datetime.timedelta(seconds=time.time() - sequence_start_time)}')

    # Calculate average reward per environment
    # If args.trial_id is not None, HPO is running, and an average across 3 different seed should be used
    if args.trial_id is not None:
        seeds = [92, 73, 11] # HPO Seeds
        # seeds = [22, 33, 44, 55, 66] # Experiment Seeds
        means, stds = [], []
        for s in seeds:
            mean, std, _ = evaluation.mean_reward(agent, sequence_keys, seed=s)
            means.append(mean)
            stds.append(std)

        # Convert into one np array, and take the mean of each environment across all seeds
        means = np.stack(means).mean(axis=0)
        stds = np.stack(stds).std(axis=0, ddof=1)
    else:
        seeds = None
        means, stds, _ = evaluation.mean_reward(agent, sequence_keys)

    # Assign rewards to the current task index
    perf_matrix[cur_task_indx] = means
    perf_std_matrix[cur_task_indx] = stds

    # Log the performance matrix
    for j, m in enumerate(means):
        eval_w.writerow([
            str(args.trial_id),
            int(cur_task_indx),
            int(j),
            float(m), 
            int(10),                                                # Fixed episode count (see evaluation.py)
            str(seeds) if args.trial_id is None else int(42),       # If an experiment or HPO trial was evaluated on a list of seeds, log that, otherwise use static 42 (evaluation.py)
            time.time() - master_start_time
        ])
        eval_f.flush() 

    # If EWC is enabled, compute it's loss
    if args.ewc:
        print('Registering Buffers')
        # Update EWC Loss
        ewc.register_buffers(b_obs, b_actions.long(), b_cfc_h_states, num_steps=args.num_steps, num_envs=args.num_envs)

    # Save this version of the model (in case of crashes later on)
    torch.save(agent.state_dict(), f'{model_path}/{sequence_keys[cur_task_indx]}.pt')

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

    ppo_hyperparameters = {
        'total_timesteps': args.total_timesteps,
        'lr': args.learning_rate,
        'ent_coef': args.ent_coef,
    }

    # Environment: early_stopping weight times
    sequence = {
        'MiniGrid-Empty-5x5-v0': 5,
        'MiniGrid-DoorKey-5x5-v0': 8, 
        'MiniGrid-Unlock-v0': 18, 
        # 'MiniGrid-KeyCorridorS3R1-v0': 12,
        # 'MiniGrid-LavaGapS5-v0': 15
    }   

    pretrain_sequence = {
        'MiniGrid-KeyCorridor-v0': 12,
        'MiniGrid-LavaGapS5-v0': 12
    }  

    # sequence = {
    #     args.env_id: 0
    # }

    # Create a list of environment keys. Prevents the need for manually managing two env_id and patience lists
    sequence_keys = list(sequence.keys())

    # Initialize all seeds to be the same
    # Try not to modify
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    # Assign to GPU (or CPU if not connected)
    # device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')
    device = torch.device('cpu') # 'cpu' is actually faster in Gym at this small scale 
    print(f'Device: {device}')

    # Create the first batch of environments with the first environment listed in the curriculum
    # This passes 'MiniGrid-Empty-5x5-v0' to the function, returning 4 environments synced to be able to run parallel
    # **
    # Most environments within MiniGrid share the same input and output space (image as input with 7 actions)
    # so this running this on-demand doesn't cause any shape issues. Environments with larger image inputs can be 
    # shrunk to fit the standard, or vice versa
    # ** 
    envs = create_sync_envs(sequence_keys[0])

    # Confirm the action space is Discrete (left, right, etc.), no Continuous 
    assert isinstance(envs.single_action_space, gym.spaces.Discrete) # Only disscrete action space is supported here

    # Pre-training prints
    if args.cfc_actor and args.cfc_critic:
        print('CfC A&C')
        model_name = f"CfC_A&C{'_ewc' if args.ewc else ''}_{int(time.time())}"
    elif args.cfc_actor and not args.cfc_critic:
        print('CfC Actor')
        model_name = f"CfC_Actor{'_ewc' if args.ewc else ''}_{int(time.time())}"
    elif args.cfc_critic and not args.cfc_actor:
        print('CfC Critic')
        model_name = f"CfC_Critic{'_ewc' if args.ewc else ''}_{int(time.time())}"
    elif args.use_lstm:
        print('LSTM')
        model_name = f"LSTM{'_ewc' if args.ewc else ''}_{int(time.time())}"
    else:
        print('MLP')
        model_name = f"MLP{'_ewc' if args.ewc else ''}_{int(time.time())}"

    # Pick the right path depending on if the run is for HPO or Experiments
    if args.trial_id is None:
        model_path = f'{os.path.dirname(__file__)}/experiments/{args.exp_name}/models/{model_name}'    
    else:
        model_name = f"hpo_trial_{args.trial_id}_{args.seed}"
        model_path = f'{os.path.dirname(__file__)}/HPO/{args.exp_name}/models/{model_name}'  
    pathlib.Path(model_path).mkdir(parents=True, exist_ok=True) # Create the path

    # Create the agent
    lstm_layers = 1 # Keep as one, maybe toy with it later
    model_config = {
        'action_space': int(envs.single_action_space.n),
        'hidden_dim': int(args.hidden_dim),
        'hidden_state_dim': int(args.hidden_state_dim),
        'actor_cfc': bool(args.cfc_actor),
        'critic_cfc': bool(args.cfc_critic),
        'use_lstm': bool(args.use_lstm),
        'ewc': bool(args.ewc),
        'ewc_weight': bool(args.ewc_weight),
        'seed': int(args.seed)
    }

    agent = models.Agent(model_config).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # Update model_config with sequence and patience, and store as json in each model folder 
    # model_config.update(sequence)
    with open(f'{model_path}/config.json', 'w') as f: # 'x' creates the file
        combined_config = {
            'model_config': model_config,
            'ppo_hyperparameters': ppo_hyperparameters,
            'sequence': sequence
        }
        json.dump(combined_config, f, indent=2)

    # CLAUDE CODE BELOW - > MAKE SEXIER LATER
    rollout_log_file = f'{model_path}/training_log_{args.exp_name}_{args.seed}.csv'
    
    # Create csv logging files for rollout episodes, updates, and evaluations
    # Return File object, and the opened 'writer' function for direct logging (faster)
    def make_writer(path, header):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        f = open(path, "a", newline="")
        w = csv.writer(f)
        if f.tell() == 0:
            w.writerow(header)
        return f, w

    episode_f, episode_w = make_writer(f'{model_path}/episodes.csv',
        ["trial_id","task_index","global_step","task_step","env_id","seed","episodic_return","episodic_length","wall_time"])

    update_f, update_w = make_writer(f'{model_path}/updates.csv',
        ["trial_id","task_index","global_step", "policy_loss","value_loss","entropy","approx_kl","clipfrac",
        "explained_var","lr","sps","wall_time"])

    eval_f, eval_w = make_writer(f'{model_path}/evals.csv',
        ["trial_id","task_index","eval_task","mean_return","n_episodes","seed","wall_time"])

    # Collect baseline model performance (performance with no training)
    # This is needed to compute Forward Transfer
    # Returns mean reward for each task
    baselines = []
    print('Collecting random baselines...')
    baselines = evaluation.mean_reward(agent, sequence_keys, return_all=False)

    # EWC Initialization
    if args.ewc:
        from utils.stabilization import ElasticWeightConsolidation
        ewc_strength = args.ewc_weight # The original EWC Paper uses 400 inside of the Atari Suite with DQN for reference, That's probably too much
        ewc = ElasticWeightConsolidation(agent, ewc_strength)
        print(f'EWC Initialized with strength {ewc_strength}...')

    # Alogirthm Logic
    # Storage
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

    # Initialize the states depending on if CfC is used for either Actor or Critic
    if args.cfc_actor or args.cfc_critic:
        cfc_h_state = torch.zeros((args.num_steps, args.num_envs, args.hidden_state_dim)).to(device)
    else: 
        cfc_h_state = None

    # Create tensors for LSTM states. LSTM states are expected to be tuples, however, tuples are immutable, so values must first be created as tensors
    if args.use_lstm:
        lstm_h_state = torch.zeros((args.num_steps, lstm_layers, args.num_envs, args.hidden_state_dim)).to(device)
        lstm_c_state = torch.zeros((args.num_steps, lstm_layers, args.num_envs, args.hidden_state_dim)).to(device)
    else: 
        lstm_h_state = None
        lstm_c_state = None

    # Start training
    global_step = 0

    # Track total time of training
    master_start_time = time.time()
    es_triggered = False
    # Keep track of average rewards per task 
    perf_matrix = np.zeros((len(sequence_keys), len(sequence_keys)))
    perf_std_matrix = np.zeros((len(sequence_keys), len(sequence_keys)))
    early_stopping_interval = 10
    # Iterate over each environment within the curriculum and train the model
    for indx, cur_env in enumerate(sequence_keys):
        print(f'Current environment: {cur_env}')

        # Keep track of how long each env lasts, it should get longer with each
        sequence_start_time = time.time()
        
        if args.es_flag:
            # initialize the EarlyStopping class at start of every episode to ensure no overlap between task performance
            es_monitor = EarlyStopping(patience=sequence[cur_env], min_delta=0.01, mode='max', verbose=True, restore_weights_flag=args.es_restore_weights) 

        # The first batch is created before this loop. If not the first batch, 
        # create the next batch of environments associated with the current curriculum environment
        if indx > 0:
            # Call the function to make envs for new curriculum
            envs = create_sync_envs(env_name=cur_env)

        # Reset the environment (initialize for new envs)
        next_obs, _ = envs.reset(seed=args.seed)

        # Permutate the observations to be (batch_size=4 (envs), channels=3, height, width)
        next_obs['image'] = torch.tensor(np.array(next_obs['image']), dtype=torch.float32, device=device).permute(0, 3, 1, 2)
        next_done = torch.zeros(args.num_envs).to(device)

        env_step_count = 0
        for iteration in range(1, args.num_iterations + 1):
            # Anneal the learning rate if enabled
            # Will be reset at the start of every environment
            if args.anneal_lr:
                frac = 1.0 - (iteration - 1.0) / args.num_iterations
                lrnow = frac * args.learning_rate
                optimizer.param_groups[0]['lr'] = lrnow

            # Prepare CfC or LSTM states accordingly
            if args.cfc_actor or args.cfc_critic:
                cfc_state = torch.zeros((args.num_envs, args.hidden_state_dim)).to(device)
            else:
                cfc_state = None

            if args.use_lstm:
                t_lstm_h_state = torch.zeros((lstm_layers, args.num_envs, args.hidden_state_dim)).to(device)
                t_lstm_c_state = torch.zeros((lstm_layers, args.num_envs, args.hidden_state_dim)).to(device)
            else: 
                t_lstm_h_state = None
                t_lstm_c_state = None

            # Perform a collection of states in batches of usually 128
            for step in range(0, args.num_steps):
                global_step += args.num_envs # account for each step in every env
                env_step_count += args.num_envs
                obs[step] = next_obs['image']
                dones[step] = next_done

                # Store the current steps states for later retrieval during update training
                # Intialized to zero but update over time
                if args.cfc_actor or args.cfc_critic:
                    cfc_h_state[step] = cfc_state
                
                if args.use_lstm:
                    lstm_h_state[step] = t_lstm_h_state
                    lstm_c_state[step] = t_lstm_c_state
                
                # Convert to proper tuple format before forward passing
                lstm_states = (t_lstm_h_state, t_lstm_c_state)
                
                # Algorithm action logic:
                # During the collection of data, we pass the state to the model
                # and sample a stochastic action and store it. Data collection is stochastic
                # Learning occurs after data is collected and epochs are run
                with torch.no_grad():
                    # Get the action the actor takes. And get the value the critic assigns said action
                    action, logprob, _, value, new_cfc_state, new_lstm_states, _ = agent.get_action_and_value(next_obs['image'], cfc_state, lstm_states)

                    # If CfC is used, Update the states with new states, resetting hidden states where the next_state is terminal
                    if not new_cfc_state is None:
                        new_cfc_state = new_cfc_state * (1.0 - next_done.unsqueeze(1))
                        cfc_state = new_cfc_state

                    # Same logic for LSTM, altered to incorporate the 'layer' dimension needed is LSTMs
                    if not new_lstm_states == (None, None):
                        new_h_state, new_c_state = new_lstm_states
                        
                        done_mask = (1.0 - next_done).unsqueeze(0).unsqueeze(2)  # (1, batch, 1)
                        
                        new_h_state = new_h_state * done_mask
                        new_c_state = new_c_state * done_mask
                        
                        t_lstm_h_state = new_h_state
                        t_lstm_c_state = new_c_state

                    values[step] = value.flatten()
                actions[step] = action
                logprobs[step] = logprob

                # Try note to modify: play the game and log 
                next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
                next_done = np.logical_or(terminations, truncations)
                rewards[step] = torch.tensor(reward).to(device)

                # Once again we need to extract the image from the obs and permute for processing
                next_obs['image'] = torch.tensor(next_obs['image'], dtype=torch.float32, device=device).permute(0, 3, 1, 2)
                next_done = torch.Tensor(next_done).to(device)

                # Log the rewards after each episode of every environment ends
                # Easier environments finish much faster after being learned (Empty is only 5 steps and learns instantly)
                # so the easier tasks are logged more often. This doesn't affect training though
                if "episode" in infos and "_episode" in infos:
                    for i, finished in enumerate(infos["_episode"]):
                        if finished:
                            episode_w.writerow([
                                str(args.trial_id),
                                int(indx),
                                int(global_step),
                                int(env_step_count),
                                str(cur_env),
                                int(args.seed),
                                float(infos['episode']['r'][i]),
                                int(infos['episode']['l'][i]),
                                time.time() - master_start_time
                            ])
                            # Only flush occasionally, not every row (faster)
                            if global_step % 256 == 0: episode_f.flush()

                            print(f"global_step={global_step}, task_step_{indx}={env_step_count}, env={i}, episodic_return={infos['episode']['r'][i]}", flush=True)
                            writer.add_scalar("charts/episodic_return", infos['episode']['r'][i], global_step)
                            writer.add_scalar("charts/episodic_length", infos['episode']['l'][i], global_step)

            # Calculate advantages for future usage in algorithm 
            with torch.no_grad():
                lstm_states = (t_lstm_h_state, t_lstm_c_state)
                # Get the critics thoughts on the value of the next state (for all envs)
                next_value = agent.get_value(next_obs['image'], cfc_states=cfc_state, lstm_states=lstm_states).reshape(1, -1)

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
            b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values.reshape(-1)

            if args.cfc_actor or args.cfc_critic:
                b_cfc_h_states = cfc_h_state.reshape(-1, args.hidden_state_dim)
            else:
                b_cfc_h_states = None

            if args.use_lstm:
                b_lstm_h_states = lstm_h_state.reshape(-1, lstm_layers, args.hidden_state_dim)
                b_lstm_c_states = lstm_c_state.reshape(-1, lstm_layers, args.hidden_state_dim)
            else:
                b_lstm_h_states = None
                b_lstm_c_states = None

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
                    if args.cfc_actor or args.cfc_critic:
                        mb_cfc_states = b_cfc_h_states[mb_inds].clone()
                    else:
                        mb_cfc_states = None

                    if args.use_lstm:
                        mb_lstm_states = (
                            b_lstm_h_states[mb_inds].transpose(0, 1).clone(),  # (mb, 1, hidden) -> (1, mb, hidden)
                            b_lstm_c_states[mb_inds].transpose(0, 1).clone()
                        )
                    else:
                        mb_lstm_states = None

                    # Forward pass (with grad) the minibatch through to the model to get values associated with the provided action
                    _, new_log_prob, entropy, new_value, _, _, _ = agent.get_action_and_value(b_obs[mb_inds], cfc_states=mb_cfc_states, lstm_states=mb_lstm_states, action=b_actions.long()[mb_inds])
                    
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

                    if args.ewc:
                        # If still on the first task, don't compute EWC loss
                        if indx > 0:
                            # Apply Elastic Weight Consolidation
                            print(f'Total loss: {loss}')
                            loss += ewc.compute_ewc_loss() # Is 0.0 on first run, but updates over time

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

            current_sps = int(global_step / (time.time() - master_start_time))

            update_w.writerow([
                str(args.trial_id),
                int(indx),
                int(global_step),
                float(pg_loss.item()),
                float(v_loss.item()),
                float(entropy_loss.item()),
                float(approx_kl.item()),
                float(np.mean(clip_fracs)),
                float(explained_var),
                float(optimizer.param_groups[0]["lr"]), # Capture the annealing lr, not args.lr
                float(current_sps),
                time.time() - master_start_time 
            ])
            if indx % 64 == 0: update_f.flush()

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
            print("SPS:", current_sps)
            writer.add_scalar("charts/SPS", current_sps, global_step)
            
            # EARLY STOPPING
            # If early stopping is enabled, check if model training on current task should end
            # Ensure the model trains for at least k iterations before checking if early stopping should trigger
            # And only check early_stopping every k iterations
            if args.es_flag and (iteration > early_stopping_interval) and (iteration % early_stopping_interval == 0):
                
                # Evaluate the mode on the current environment to check if early stopping should trigger
                es_mean = evaluation.mean_reward(agent=agent, envs=[cur_env], episodes=5) # Increase episodes for more reliable mean_reward
                es_triggered = es_monitor.update(es_mean[0], model=agent)

                if es_triggered:
                    if args.es_restore_weights:
                        es_monitor.restore_weights(agent)
                    end_environment(indx, es_triggered)
                    break

        # Save the current version of the model for testing while curriculum continues (and reloding from previous point)
        if not es_triggered:
            end_environment(indx)
        
        # Reset early stopping flag for next env
        es_triggered = False
    
    # Close open files
    envs.close()
    writer.close()
    episode_f.close()
    update_f.close()
    eval_f.close()
    
    print('Computing Metrics...')
    fwt = evaluation.compute_fwt(perf_matrix, baselines)
    bwt = evaluation.compute_bwt(perf_matrix)
    # Take the mean of all averaged rewards to get a singular average
    rewards, _, _ = evaluation.mean_reward(agent, sequence_keys)
    mean_reward = np.mean(rewards)


    # Make plots and save in current directory
    evaluation.plot_perf_matrix(perf_matrix, sequence=sequence_keys, save_path=f'{model_path}/performance_matrix.svg') # Save as SVG to prevent blur
    evaluation.plot_metrics(fwt, bwt, mean_reward, save_path=f'{model_path}/metrics.svg')
    evaluation.plot_reward(path=model_path, sequence=sequence_keys, save_path=f'{model_path}/reward.svg')
    evaluation.plot_loss_curve(path=model_path, sequence=sequence_keys, save_path=f'{model_path}/loss.svg')

    # Save the performance matrices as a NumPy array
    np.save(f'{model_path}/performance_matrix.npy', perf_matrix)
    np.save(f'{model_path}/performance_std_matrix.npy', perf_std_matrix)

    print(f'Final training time: {datetime.timedelta(seconds=time.time() - master_start_time)}')
    

    torch.save(agent.state_dict(), f'{model_path}/final_model.pt')
    # Print a BEL character to screen. Will play an audible sound in terminal indicating it's done
    print('\a')