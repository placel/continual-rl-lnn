import os
import random
import time
from dataclasses import dataclass
# https://ncps.readthedocs.io/en/latest/examples/atari_ppo.html
import gymnasium as gym
import numpy as np
import torch
import torch.backends
import torch.backends.cudnn
import torch.nn as nn
import torch.optim as optim
import tyro
import datetime
import CfCVariations as lnn_models

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
    env_id: str = 'CartPole-v1'
    # Total timesteps allowed in the whole experiment
    total_timesteps: int = 500_000
    # Learning rate for the optimizer
    learning_rate: float = 2.5e-4
    # Number of environments for parallel game processing
    num_envs: int = 4
    # Total number of steps to run in each environment per policy rollout
    num_steps: int = 128
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
    clip_coef: float = 0.2
    # Toggles usage of clipped loss for the value function
    clip_vloss: bool = True
    # coefficient of the entropy
    ent_coef: float = 0.01
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

# Custom layer weight initialization
# Orthogonal weight setting ensures no neurons have any sort of correlation
# This is particularly important in reinforcement learning whent he dataset is small
# and exploration is required. Any inherent correlation could influence the results
# We also set all biases to 0
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()

        obs_space, action_space = np.array(envs.single_observation_space.shape).prod(), np.array(envs.single_action_space).prod()
        
        # Create critic model, only pass obs_space as action_space isn't needed in a critic model
        self.critic = lnn_models.CriticCfC(obs_space)
        
        # Create the actor model. Pass in the envs which will adapt depending on the tasks at hand
        self.actor = lnn_models.ActorCfC(obs_space, action_space.n)

    def get_value(self, x):
        return self.critic(x)
    
    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)

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
    
    writer = SummaryWriter(f'./src/ppo/lnn/cartpole/runs/{run_name}')
    writer.add_text(
        'hyperparameters',
        '|param|value|\n|-|-|\n%s' % ('\n'.join([f'|{key}|{value}|' for key, value in vars(args).items()]))
    )

    # Try not to modify
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')
    print(device)
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )

    assert isinstance(envs.single_action_space, gym.spaces.Discrete) # Only disscrete action space is supported here

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # Alogirthm Logic
    #Storage
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # Try not to modify
    # Start game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    for iteration in range(1, args.num_iterations + 1):
        # 'Anneal' the learning rate if enabled
        # Adjust the learning rate as the model trains
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]['lr'] = lrnow

        for step in range(0, args.num_steps):
            global_step += args.num_envs # account for each step in every env
            obs[step] = next_obs
            dones[step] = next_done

            # Algorithm logic: action logic
            # During the collection of data, we pass the state to the model
            # and sample a random action and store it. Data collection is stochastic
            # Learning occurs after data is collected and epochs are run
            with torch.no_grad():
                # Get the action the actor takes. And get the value the critic assigns said action
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # Try note to modify: play the game and log 
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

            # If the the game is over, log the stats
            if 'final_info' in infos:
                for info in infos['final_info']:
                    if info and 'episode' in info:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        writer.add_scalar('charts/episodic_return', info['episode']['r'], global_step)
                        writer.add_scalar('charts/episodic_length', info['episode']['l'], global_step)
        
        # Calculate advantages for future usage in algorithm 
        with torch.no_grad():
            # Get the critics thoughts on the value of the next state (for all envs)
            next_value = agent.get_value(next_obs).reshape(1, -1)

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
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,), envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

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

                # Forward pass (with grad) the minibatch through to the model to get values associated with the provided action
                _, new_log_prob, entropy, new_value = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                
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
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    print(f'Final training time: {datetime.timedelta(seconds=time.time() - start_time)}')
    envs.close()
    writer.close()
    torch.save(agent.state_dict(), f'./src/ppo/lnn/cartpole/models/double_lnn_agent.pt')