import os
import random
import time
import json
from dataclasses import dataclass
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
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))) # Needed to import utils

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
    # Trial number (for HPO)
    exp_id: str = None
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

    # STABILIZATION ARGS
    # Toggle early stopping (es)
    es_flag: bool = False
    # Restore weights with Early Stopping (generally False)
    es_restore_weights: bool = False
    # Toggle Elastic Weight Consolidation (EWC)
    ewc: bool = False
    # EWC Weight (hyperparameter)
    ewc_weight: float = 0.0
    # Toggle CLEAR stabilization technique
    clear: bool = False
    # Size of the replay buffer per task in CLEAR (in trajectories)
    clear_capacity: int = 32
    # Ratio of old to new experiences for CLEAR replay
    clear_ratio: float = 0.25 # Between 0.0 and 1.0
    # All below values were taken from the CLEAR paper
    # V-trace is a deviation from PPO, and is used in the original CLEAR paper. Experiment by using no v-trace with CLEAR, but KL-Divergence penalty will still be applied
    vtrace: bool = False # Disabled by default as this project focuses on PPO. V-trace disabled PPO
    vtrace_clip_rho: float = 1.0
    vtrace_clip_c: float = 1.0
    clear_kl_coef: float = 0.5
    clear_value_coef: float = 0.5

    # ALGORITHM ARGS
    # Name of the environment
    env_id: str = 'MiniGrid-Empty-5x5-v0'
    # Total timesteps allowed in the whole experiment
    total_timesteps: int = 200_000
    # Learning rate for the optimizer
    learning_rate: float = 1.5e-4 # Better for CfC models
    # Number of environments for parallel game processing
    num_envs: int = 8
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
    update_epochs: int = 3 # 3 instead of 4 to balance out the increase to 8 envs (instead of 4)
    # Toggle advantages normalization
    norm_adv: bool = True
    # Clip epsilon
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

# AsyncVectorEnv is significantly faster than SyncVectorEnv (true parallelization)
from gymnasium.vector import AsyncVectorEnv
def create_envs(env_name):
    return AsyncVectorEnv(
        [make_env(env_name, i, args.capture_video, run_name) for i in range(args.num_envs)],
        shared_memory=False,  # True doesn't work in MiniGrid
        context='spawn',     # safer on Windows/Mac too
    )

# This is called at the end of an environment, whether triggered by early stopping or naturally
# Just saves the model, computes perf_matrix values, and prints a message 
def end_environment(cur_task_indx, es_triggered=False):
    print(f'Environment training time: {datetime.timedelta(seconds=time.time() - sequence_start_time)}')

    # Calculate average reward per environment
    # If args.trial_id is not None, HPO is running, and an average across 3 different seed should be used
    if (args.trial_id is not None) or (args.exp_id is not None):
        seeds = [222, 333, 444] # HPO Seeds
        # seeds = [2222, 3333, 4444] # EWC Optuna Seeds
        # seeds = [10, 20, 30] # Experiment Seeds
        # seed 
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
            int(10), # Fixed episode count (see evaluation.py)
            str(seeds) if (args.trial_id is not None) or (args.exp_id is not None) else int(42), # If an experiment or HPO trial was evaluated on a list of seeds, log that, otherwise use static 42 (evaluation.py)
            time.time() - master_start_time
        ])
        eval_f.flush() 

    # If EWC is enabled, register buffers for loss computation on update
    # Don't register buffers if the last task is finished, as buffers for this task are not needed; wasted compute
    if args.ewc and not cur_task_indx == len(sequence) - 1: 
        print('Registering Buffers')
        if args.cfc_actor or args.cfc_critic:
            ewc.register_buffers(b_obs, b_actions.long(), b_dones=b_dones, model_type='cfc', batch_size=args.minibatch_size, hidden_state_dim=args.hidden_state_dim)
        elif args.use_lstm:
            ewc.register_buffers(b_obs, b_actions.long(), b_dones=b_dones, model_type='lstm', batch_size=args.minibatch_size, hidden_state_dim=args.hidden_state_dim, lstm_layers=lstm_layers)
        else:
            ewc.register_buffers(b_obs, b_actions.long(), b_dones=b_dones, model_type='mlp')

    torch.save(agent.state_dict(), f'{model_path}/{sequence_keys[cur_task_indx]}.pt') # Checkpoint

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

    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['OMP_NUM_THREADS'] = '1'

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
        'clear_kl_coef': args.clear_kl_coef,
        'clear_value_coef': args.clear_value_coef
    }

    # Environment: early_stopping weight times
    sequence = {
        'MiniGrid-Empty-5x5-v0': 5,
        'MiniGrid-DoorKey-5x5-v0': 8, 
        'MiniGrid-Unlock-v0': 18, 
        # 'MiniGrid-KeyCorridorS3R1-v0': 12,
        # 'MiniGrid-LavaGapS5-v0': 15
    }   

    # Create a list of environment keys. Prevents the need for manually managing two env_id and patience lists
    sequence_keys = list(sequence.keys())

    # Initialize all seeds to be the same
    # Try not to modify
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.backends.cudnn.deterministic = args.torch_deterministic

    # Assign to GPU (or CPU if not connected)
    # device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')
    device = torch.device('cpu') # 'cpu' is actually faster in Gym at this small scale 
    print(f'Device: {device}')

    # Create the first batch of environments with the first environment listed in the curriculum
    envs = create_envs(sequence_keys[0])

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
    action_space = envs.single_action_space.n
    model_config = {
        'action_space':     int(action_space),
        'hidden_dim':       int(args.hidden_dim),
        'hidden_state_dim': int(args.hidden_state_dim),
        'actor_cfc':        bool(args.cfc_actor),
        'critic_cfc':       bool(args.cfc_critic),
        'use_lstm':         bool(args.use_lstm),
        'ewc':              bool(args.ewc),
        'ewc_weight':       float(args.ewc_weight),
        'clear':            bool(args.clear),
        'v-trace':          bool(args.vtrace),
        'seed':             int(args.seed)
    }

    agent = models.Agent(model_config).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # Experimental SPS Speedups
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if not args.torch_deterministic:
        torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    # Update model_config with sequence and patience, and store as json in each model folder 
    # model_config.update(sequence)
    with open(f'{model_path}/config.json', 'w') as f: # 'x' creates the file
        combined_config = {
            'model_config': model_config,
            'ppo_hyperparameters': ppo_hyperparameters,
            'sequence': sequence
        }
        json.dump(combined_config, f, indent=2)

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

     # Initialize the CSV files with proper headers
    episode_f, episode_w = make_writer(f'{model_path}/episodes.csv',
        ["trial_id","task_index","global_step","task_step","env_id","seed","episodic_return","episodic_length","wall_time"])

    update_f, update_w = make_writer(f'{model_path}/updates.csv',
        ["trial_id","task_index","global_step", "policy_loss","value_loss","entropy","approx_kl","clipfrac",
        "explained_var","lr","sps","wall_time"])

    eval_f, eval_w = make_writer(f'{model_path}/evals.csv',
        ["trial_id","task_index","eval_task","mean_return","n_episodes","seed","wall_time"])

    # Collect baseline model performance (performance with no training)
    # This is needed to compute Forward Transfer. Returns mean reward for each task
    baselines = []
    print('Collecting random baselines...')
    baselines = evaluation.mean_reward(agent, sequence_keys, return_all=False)

    # EWC Initialization
    if args.ewc:
        from utils.stabilization import EWC
        ewc_strength = args.ewc_weight # The original EWC Paper uses 400 inside of the Atari Suite with DQN for reference
        ewc = EWC(agent, ewc_strength)
        print(f'EWC Initialized with strength {ewc_strength}...')

    if args.clear:
        from utils.stabilization import CLEAR
        clear = CLEAR(args.clear_capacity * len(sequence) - 1) # Create a buffer with size args.clear_capacity (ex. 32) per task, exluding the last as it's not needed for replay
        print(f'CLEAR initialized...')

    # Alogirthm Logic
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
    logits = torch.zeros((args.num_steps, args.num_envs, action_space)).to(device) 
    bootstrap_values = torch.zeros((args.num_envs)).to(device)

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
        sequence_start_time = time.time()

        # The first batch is created before this loop. If not the first batch, create the next batch of environments associated with the current curriculum environment
        if indx > 0: envs = create_envs(env_name=cur_env)

        # Initialize the states depending on if CfC is used for either Actor or Critic. Reset on each task
        if args.cfc_actor or args.cfc_critic:
            next_cfc_state = torch.zeros((args.num_envs, args.hidden_state_dim)).to(device)
        else: 
            next_cfc_state = None

        # Create tensors for LSTM states. LSTM states are expected to be tuples, however, tuples are immutable, so values must first be created as tensors
        if args.use_lstm:
            next_lstm_state = (
                torch.zeros((lstm_layers, args.num_envs, args.hidden_state_dim)).to(device),
                torch.zeros((lstm_layers, args.num_envs, args.hidden_state_dim)).to(device)
            )
        else:
            next_lstm_state = (None, None)
        
        if args.es_flag:
            # initialize the EarlyStopping class at start of every episode to ensure no overlap between task performance
            es_monitor = EarlyStopping(patience=sequence[cur_env], min_delta=0.01, mode='max', verbose=True, restore_weights_flag=args.es_restore_weights) 

        # Reset the environment (initialize for new envs)
        next_obs, _ = envs.reset(seed=args.seed)

        # Permutate the observations to be (batch_size=4 (envs), channels=3, height, width)
        next_obs['image'] = torch.tensor(np.array(next_obs['image']), dtype=torch.float32, device=device).permute(0, 3, 1, 2)
        next_done = torch.zeros(args.num_envs).to(device)

        env_step_count = 0
        for iteration in range(1, args.num_iterations + 1):
            initial_cfc_state  = next_cfc_state.clone() if args.cfc_actor or args.cfc_critic else None
            initial_lstm_state = (next_lstm_state[0].clone(), next_lstm_state[1].clone()) if args.use_lstm else (None, None)

            # Anneal the learning rate if enabled. Will be reset at the start of every environment
            if args.anneal_lr:
                frac = 1.0 - (iteration - 1.0) / args.num_iterations
                lrnow = frac * args.learning_rate
                optimizer.param_groups[0]['lr'] = lrnow

            # Perform a collection of states in batches of usually 128
            for step in range(0, args.num_steps):
                global_step += args.num_envs # account for each step in every env
                env_step_count += args.num_envs
                obs[step] = next_obs['image']
                dones[step] = next_done
                
                # Algorithm action logic:
                # During the collection of data, we pass the state to the model and sample a stochastic action and store it. 
                # Data collection is stochastic. Learning occurs after data is collected and epochs are run
                with torch.no_grad():
                    # Get the action the actor takes. And get the value the critic assigns said action
                    action, logprob, _, value, next_cfc_state, next_lstm_state, new_logits = agent.get_action_and_value(next_obs['image'], next_cfc_state, next_lstm_state, dones=next_done)
                    values[step] = value.flatten()
                actions[step] = action
                logprobs[step] = logprob
                logits[step] = new_logits

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

            # Get the bootstrap value for advantage estimation
            with torch.no_grad():
                # Get the critics thoughts on the value of the next state (for all envs)
                next_value = agent.get_value(next_obs['image'], cfc_states=next_cfc_state, lstm_states=next_lstm_state, dones=next_done).reshape(1, -1)
                bootstrap_values = next_value.squeeze(0)

            rollout = {
                'obs': obs, 'actions': actions, 'behav_logp': logprobs,
                'rewards': rewards, 'dones': dones, 'values_behav': values,
                'behav_logits': logits, 'bootstrap_values': bootstrap_values,
            }

            rollout['states'] = {
                'cfc': initial_cfc_state,
                'lstm': initial_lstm_state
            }

            # CLEAR Logic
            if args.clear:
                # Update buffer every rollout, except for the final task. Replay data should only contain data from old tasks
                if indx < (len(sequence_keys) - 1):
                    clear.update_buffer(rollout)
                
                is_replay = torch.zeros((args.num_steps, args.num_envs), dtype=torch.bool, device=device) # Default no replay envs

                # If V-trace is used, blend old with the new for V-trace calculation
                if indx > 0 and args.vtrace:
                    rollout, is_replay = clear.blend_rollout(
                        rollout, int(args.clear_ratio * args.num_envs)
                    )
            else:
                is_replay = torch.zeros((args.num_steps, args.num_envs), dtype=torch.bool, device=device) # No CLEAR, just mark everything on-policy

            # Extract for convenience
            O = rollout['obs']
            A = rollout['actions']
            R = rollout['rewards']
            D = rollout['dones']
            Vb = rollout['values_behav']
            Lb = rollout['behav_logp']
            B  = rollout['behav_logits']
            boot_b = rollout['bootstrap_values']
            states = rollout['states']

            # Number of steps, number of envs
            T, N = A.shape[:2]
            advantages = torch.zeros_like(R, device=device)
            # V-trace Claculation
            if args.clear and args.vtrace:
                # initial hidden states for ALL columns. These are overwritten every pass, similarly to the collection phase as we need to unroll through time
                cfc_state = states['cfc'] if states['cfc'] is not None else None
                lstm_states = (
                    states['lstm'][0],
                    states['lstm'][1]
                ) if args.use_lstm else (None, None)

                with torch.no_grad():
                    # current policy stats along the unroll for ALL columns
                    logp_all = torch.empty((T, N), device=device)
                    Vcur_all = torch.empty((T, N), device=device)

                    # Can't flatten as we need states. Iterate over each step (of args.num_steps), passing each batch of args.num_envs 
                    for t in range(T):
                        logp_t, V_t, cfc_state, lstm_states = agent.evaluate_actions(
                            O[t], A[t].long(),
                            cfc_states=cfc_state,
                            lstm_states=lstm_states,
                            dones=D[t],
                        )
                        logp_all[t] = logp_t.squeeze(-1)  # keep [N]
                        Vcur_all[t] = V_t.squeeze(-1)
                    
                    # Importance weight calculation
                    iw = (logp_all - Lb).exp()
                    rho = iw.clamp(max=args.vtrace_clip_rho)
                    c = iw.clamp(max=args.vtrace_clip_c)

                    vs = torch.empty_like(Vcur_all)
                    for t in reversed(range(T)):
                        v_t   = Vcur_all[t]
                        v_tp1 = boot_b if t == T - 1 else Vcur_all[t + 1]
                        delta = R[t] + args.gamma * v_tp1 * (1.0 - D[t]) - v_t
                        if t == T - 1:
                            vs[t] = v_t + rho[t] * delta
                        else:
                            vs[t] = v_t + rho[t] * delta + args.gamma * c[t] * (vs[t + 1] - Vcur_all[t + 1])

                    next_vs = torch.empty_like(Vcur_all)
                    next_vs[:-1] = vs[1:]
                    next_vs[-1]  = boot_b
                    advantages = (R + args.gamma * next_vs * (1.0 - D) - Vcur_all).detach()
                    returns    = vs.detach()

            # V-trace disabled; GAE as normal
            else:
                last_gae_lam = 0.0
                for t in reversed(range(T)):
                    if t == T - 1:
                        next_non_terminal = 1.0 - next_done
                        next_values = boot_b
                    else:
                        next_non_terminal = 1.0 - D[t + 1]
                        next_values = Vb[t + 1]
                    delta = R[t] + args.gamma * next_values * next_non_terminal - Vb[t]
                    last_gae_lam = delta + args.gamma * args.gae_lambda * next_non_terminal * last_gae_lam
                    advantages[t] = last_gae_lam
                returns = advantages + Vb
            
            # Flatten batch
            b_obs = O.reshape((-1,) + permuted_image)
            b_actions = A.reshape((-1,) + envs.single_action_space.shape)
            b_logprobs = Lb.reshape(-1)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = Vb.reshape(-1)
            b_dones = D.reshape(-1)
            b_behav_logits = B.reshape(-1, action_space)
            b_values_behav = Vb.reshape(-1)

            # Optimize the policy (actor) and value (critic) networks
            assert args.num_envs % args.num_minibatches == 0
            envs_per_batch = args.num_envs // args.num_minibatches
            env_inds = np.arange(args.num_envs)
            flat_inds = np.arange(args.batch_size).reshape(args.num_steps, args.num_envs)
            flat_is_replay = is_replay.reshape(-1)
            clip_fracs = []
            
            if args.norm_adv:
                flat_is_replay = is_replay.reshape(-1).bool()
                on_pol = ~flat_is_replay
                m = b_advantages[on_pol].mean()
                s = b_advantages[on_pol].std(unbiased=False).clamp_min(1e-8)
                b_advantages = (b_advantages - m) / s

            # if using PPO + KL-Divergence cloning, no v-trace, calculate scaling per minibatch
            apply_bc = args.clear and (not args.vtrace) and indx > 0
            if apply_bc:
                num_minibatches = args.num_envs // envs_per_batch
                # scale so total BC weight per update equals (clear_*_coef)
                bc_kl_scale = args.clear_kl_coef / (args.update_epochs * num_minibatches)
                bc_v_scale  = args.clear_value_coef  / (args.update_epochs * num_minibatches)

            for epoch in range(args.update_epochs):
                if apply_bc: # Extract relevant CLEAR + Cloning samples from replay buffer
                    bc = clear.sample_rollouts(count=args.num_envs) # Sample num_envs number trajectories at every epoch
                    O_bc, A_bc, D_bc = bc['obs'], bc['actions'], bc['dones']
                    Bc = bc['behav_logits'].detach()
                    Vb_bc = bc['values_behav'].detach()
                    S_bc = bc['states']
                    burn_len = min(max(1, int(args.clear_ratio * args.num_steps)), O_bc.shape[0]) # Number of 'burn-in' steps. We need states to unroll, but not the full sequence

                np.random.shuffle(env_inds)
                for start in range(0, args.num_envs, envs_per_batch):
                    end = start + envs_per_batch
                    mb_env_inds = env_inds[start:end]
                    mb_inds = flat_inds[:, mb_env_inds].ravel()
                    mb_mask = flat_is_replay[mb_inds].bool()

                    # Prepare hidden states
                    cfc_pass = initial_cfc_state[mb_env_inds] if args.cfc_actor or args.cfc_critic else None
                    # We use [:, mb_env_inds] to take the initial state of each Layer in the LSTM. Although this is typically one 1. Also why we don't do it for CfC 
                    lstm_pass = (initial_lstm_state[0][:, mb_env_inds], initial_lstm_state[1][:, mb_env_inds]) if args.use_lstm else (None, None)

                    # Forward pass (with grad) the minibatch through to the model to get values associated with the provided action
                    _, new_log_prob, entropy, new_value, _, _, new_logits = agent.get_action_and_value(
                        b_obs[mb_inds], 
                        cfc_states=cfc_pass, 
                        lstm_states=lstm_pass, 
                        action=b_actions.long()[mb_inds],
                        dones=b_dones[mb_inds]
                    )
                    
                    # Ratio of the probablity of the new policy vs the old policy for taking provided action
                    # This is a key component in the PPO equation
                    log_ratio = new_log_prob - b_logprobs[mb_inds]
                    ratio = log_ratio.exp()

                    with torch.no_grad():
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        old_approx_kl = (-log_ratio).mean()
                        approx_kl = ((ratio - 1) - log_ratio).mean()
                        clip_fracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                    # Flatten minibatch selections
                    mb_adv = b_advantages[mb_inds]      # from V-trace (if clear) or GAE
                    mb_ret = b_returns[mb_inds]         # targets: v_s (if clear) or GAE returns
                    mb_val = b_values[mb_inds]          # critic baseline at storage
                    mb_logp_mu = b_logprobs[mb_inds]        # behavior logπμ(a|s)
                    mb_newlogp = new_log_prob                 # current logπθ(a|s)
                    mb_entropy = entropy                    # entropy from current policy
                    mb_newvalue = new_value.view(-1)          # critic prediction from forward pass
                    # Importance weights ρ = πθ / μ
                    rho_mb = ((mb_newlogp - mb_logp_mu).exp().clamp(max=args.vtrace_clip_rho)).detach()
                    
                    # Normalize before minbatch instead of here when CLEAR is enabled to prevent skewing
                    # if args.norm_adv:
                    #     mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                    # Policy loss
                    if args.clear and args.vtrace:
                        # CLEAR paper: V-trace PG for all samples
                        pg_loss = (-rho_mb * mb_newlogp * mb_adv).mean()
                    else:
                        # Standard PPO clipped objective
                        log_ratio = mb_newlogp - mb_logp_mu
                        ratio = log_ratio.exp()
                        pg1 = -mb_adv * ratio
                        pg2 = -mb_adv * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                        pg_loss = torch.max(pg1, pg2).mean()

                    # Value loss
                    if args.clip_vloss:
                        v_unclipped = (mb_newvalue - mb_ret) ** 2
                        v_clipped = mb_val + torch.clamp(mb_newvalue - mb_val, -args.clip_coef, args.clip_coef)
                        v_clipped = (v_clipped - mb_ret) ** 2
                        v_loss = 0.5 * torch.max(v_unclipped, v_clipped).mean()
                    else:
                        v_loss = 0.5 * ((mb_newvalue - mb_ret) ** 2).mean()

                    # CLEAR policy/value cloning on replay only
                    kl_bc = torch.tensor(0.0, device=mb_newvalue.device)
                    value_clone = torch.tensor(0.0, device=mb_newvalue.device)

                    # Apply KL-Divergence to replay experiences only
                    if args.vtrace and mb_mask.any() and indx > 0:
                        # Policy cloning
                        behav_logits_mb = b_behav_logits[mb_inds][mb_mask].detach()
                        new_logits_mb = new_logits[mb_mask]
                        logp_old = torch.log_softmax(behav_logits_mb, dim=-1)
                        logp_new = torch.log_softmax(new_logits_mb, dim=-1)
                        p_old = logp_old.exp()
                        kl_bc = (p_old * (logp_old - logp_new)).sum(dim=-1).mean()

                        # Value cloning is MSE
                        v_old = b_values_behav[mb_inds][mb_mask].detach()
                        v_new = mb_newvalue[mb_mask]
                        value_clone = ((v_new - v_old) ** 2).mean()

                    # final loss before any potential Behaviour Cloning below if CLEAR no V-trace
                    loss = pg_loss - args.ent_coef * mb_entropy.mean() + args.vf_coef * v_loss

                    # Behaviour Cloning (BC) per-minibatch cloning on PPO (only on replay data)
                    if apply_bc:
                        # initial states for selected replay columns
                        cfc_state = (S_bc['cfc'][mb_env_inds] if (args.cfc_actor or args.cfc_critic) else None)
                        if args.use_lstm:
                            h0, c0 = S_bc['lstm']
                            lstm_state = (h0[:, mb_env_inds, :].contiguous(), c0[:, mb_env_inds, :].contiguous())
                        else:
                            lstm_state = (None, None)

                        is_recurrent = args.use_lstm or args.cfc_actor or args.cfc_critic

                        # If the model is recurrent, states need to be unrolled to gather proper representation of current policy ('burn-in')
                        if is_recurrent:
                            kl_sum = torch.zeros((), device=device)
                            vmse_sum = torch.zeros((), device=device)

                            for t in range(burn_len):
                                _, _, _, v_new_bc, cfc_state, lstm_state, logits_new_bc = agent.get_action_and_value(
                                    O_bc[t, mb_env_inds],
                                    cfc_states=cfc_state, lstm_states=lstm_state,
                                    action=A_bc[t, mb_env_inds].long(), dones=D_bc[t, mb_env_inds]
                                )
                                # Policy cloning
                                logp_old = torch.log_softmax(Bc[t, mb_env_inds], dim=-1)
                                logp_new = torch.log_softmax(logits_new_bc, dim=-1)
                                p_old = logp_old.exp()
                                kl_sum  = kl_sum  + (p_old * (logp_old - logp_new)).sum(dim=-1).mean()
                                # Value cloning (MSE)
                                vmse_sum = vmse_sum + ((v_new_bc.view(-1) - Vb_bc[t, mb_env_inds].view(-1)) ** 2).mean()

                            kl_bc_mb   = kl_sum  / burn_len
                            vmse_bc_mb = vmse_sum / burn_len

                        else: # MLP doesn't need burn-in, just batch process the minibatch
                            sl = slice(0, burn_len)
                            obs_be  = O_bc[sl, mb_env_inds].flatten(0, 1).contiguous()
                            act_be  = A_bc[sl, mb_env_inds].flatten().long()
                            done_be = D_bc[sl, mb_env_inds].flatten()

                            # Current policy on old data (single forward)
                            _, _, _, v_new_bc, _, _, logits_new_bc = agent.get_action_and_value(
                                obs_be, cfc_states=None, lstm_states=(None, None),
                                action=act_be, dones=done_be
                            )

                            # Detached behavior targets
                            old_logits = Bc[sl, mb_env_inds].flatten(0, 1).detach()
                            old_vals   = Vb_bc[sl, mb_env_inds].flatten().detach()

                            # Policy Cloning
                            logp_old = torch.log_softmax(old_logits, dim=-1)
                            logp_new = torch.log_softmax(logits_new_bc, dim=-1)
                            kl_bc_mb = (logp_old.exp() * (logp_old - logp_new)).sum(dim=-1).mean()

                            # Value cloning (MSE)
                            vmse_bc_mb = ((v_new_bc.view(-1) - old_vals) ** 2).mean()
                            
                        # Apply per-minibatch BC penalty (already scaled by epochs * minibatches)
                        loss = loss + bc_kl_scale * kl_bc_mb + bc_v_scale * vmse_bc_mb

                    if args.clear and indx > 0:
                        loss = loss + args.clear_kl_coef * kl_bc + args.clear_value_coef * value_clone

                    # Apply Elastic Weight Consolidation if toggled
                    if args.ewc:
                        if indx > 0: # If still on the first task, don't compute EWC loss
                            ewc_loss = ewc.compute_ewc_loss()
                            # Since EWC strength is so large, scale it as a ratio, but don't let it beat PPO loss
                            scale = min(1.0, (loss.detach().abs() / (ewc_loss + 1e-8)).item()) # + 1e-8 incase loss is 0.0 -> Divide by 0 error
                            loss = loss + (scale * ewc_loss)

                    # Update agent
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                    optimizer.step()

                if args.target_kl is not None and approx_kl > args.target_kl:
                    break

            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)

            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
            current_sps = int(global_step / (time.time() - master_start_time))

            update_w.writerow([
                str(args.trial_id),
                int(indx),
                int(global_step),
                float(pg_loss.item()),
                float(v_loss.item()),
                float(mb_entropy.mean().item()),
                float(approx_kl.item()),
                float(np.mean(clip_fracs)),
                float(explained_var),
                float(optimizer.param_groups[0]["lr"]), # Capture the annealing lr, not args.lr
                float(current_sps),
                time.time() - master_start_time 
            ])
            
            update_f.flush() # Flush every other iteration

            # TRY NOT TO MODIFY: record rewards for plotting purposes
            writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
            writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
            writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            writer.add_scalar("losses/entropy", mb_entropy.mean().item(), global_step)
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
    envs.close(); writer.close(), episode_f.close(); update_f.close(); eval_f.close()
    
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
    print('\a') # Print a BEL character to screen. Will play an audible sound in terminal indicating it's done