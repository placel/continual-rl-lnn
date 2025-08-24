import time
import torch
import numpy as np
from torch import autograd
import torch.nn.functional as F

# Adpated for PPO by https://github.com/shivamsaboo17/Overcoming-Catastrophic-forgetting-in-Neural-Networks/blob/master/elastic_weight_consolidation.py
class EWC:
    # Agent= Model, Strength = Lambda (In EWC calculation)
    # Lambda is a hyperparam determining how strong consoldiation should be
    def __init__(self, agent, strength):
        self.agent = agent
        self.strength = strength
        self.ewc_loss = 0.0 # Default to 0.0 as the initial task shouldn't be penalized
        self.device = next(self.agent.parameters()).device

    # Take the final rollout states and respective actions as input
    # We compare the predicted action of the model with the actual action taken
    def _update_fisher_params(self, b_obs, b_actions, b_states=None, num_steps=128, num_envs=4, model_type='mlp'):        
        # Calculate how many times the model collects softmax values
        # If num_step=128 and num_envs=4, then the result is 512
        # However, since 4 envs are being processed at a time, 512 * 4 = 2048 total states will be seen (experiment with this)
        num_batch = num_steps * num_envs

        # Batch process the states instead of iterating through
        if model_type=='cfc':
            _, _, _, _, _, _, logits = self.agent.get_action_and_value(b_obs, cfc_states=b_states)
        elif model_type=='lstm':
            _, _, _, _, _, _, logits = self.agent.get_action_and_value(b_obs, lstm_states=b_states)
        else:
            _, _, _, _, _, _, logits = self.agent.get_action_and_value(b_obs, None)

        output = F.log_softmax(logits, dim=1)

        # Collect indices for the batch to be extracted
        batch_indices = torch.arange(num_batch, device=self.device)
        # Extract the softmax probabilites from the batched results
        log_liklihoods = output[batch_indices, b_actions]

        # Calculuate the estimated fisher matrix
        log_liklihoods = log_liklihoods.mean() # Don't need torch.cat as log_liklihoods is already a tensor
        log_liklihoods.requires_grad_(True)
        grad_log_liklihood = autograd.grad(log_liklihoods, self.agent.parameters(), allow_unused=True, create_graph=False, retain_graph=False)
        _buff_param_names = [param[0].replace('.', '__') for param in self.agent.named_parameters()]
        for _buff_param_name, param in zip(_buff_param_names, grad_log_liklihood):
            if param is not None: # Some parameters are never used and will cause error within compute_ewc_loss() if not stored
                self.agent.register_buffer(_buff_param_name + '_estimated_fisher', param.data.clone() ** 2)
        
    def _update_mean_params(self):
        # Iterate over named parameters in the model, and register them to the model in a buffer 
        for param_name, param in self.agent.named_parameters():
            _buff_param_name = param_name.replace('.', '__')
            self.agent.register_buffer(_buff_param_name + '_estimated_mean', param.data.clone())

    def register_buffers(self, b_obs, b_actions, b_states=None, num_steps=128, num_envs=4, model_type='mlp'):
        # Register params are called here to limit how many calls are needed in main training loop
        self._update_fisher_params(b_obs, b_actions, b_states, num_steps, num_envs, model_type)
        self._update_mean_params()

    # If adding decay to weight (for Online EWC), later add strength here for easier decay implementation
    def compute_ewc_loss(self):

        # Iterate over all parameters and penalize large differences from original vs current
        losses = []
        for param_name, param in self.agent.named_parameters():
            __buff_param_name = param_name.replace('.', '__')
            mean_buffer_name = f'{__buff_param_name}_estimated_mean'
            fisher_buffer_name = f'{__buff_param_name}_estimated_fisher'
            
            # Not all parameters are updated in _update_fisher_matrix(). This often causes an error.
            # Instead, check for which params do exist, then sum loss from those. Removes the need for a try/except
            if (hasattr(self.agent, mean_buffer_name) and hasattr(self.agent, fisher_buffer_name)):
                estimated_mean = getattr(self.agent, f'{__buff_param_name}_estimated_mean')
                estimated_fisher = getattr(self.agent, f'{__buff_param_name}_estimated_fisher')
                losses.append((estimated_fisher * (param - estimated_mean) ** 2).sum())
        
        # Summate the current ewc losses and apply the strength hyperparameter 
        # Don't return loss, just update the instance variable for easier access during training
        self.ewc_loss = (self.strength / 2) * sum(losses)
        return self.ewc_loss

# Based on: https://en.wikipedia.org/wiki/Reservoir_sampling. Adapted for PPO rollout chunking
class ResevoirSampler:
    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = [] # Stores a list of experiences (128 states each)
        self.n = 0

    def add(self, batch):
        self.n += 1

        # If there's room in the buffer, add
        if len(self.buffer) < self.max_size:
            self.buffer.append(batch)
        # Otherwise, randomly replace an existing batch with the new 
        else:
            j = np.random.randint(self.n)
            if j <=self.max_size:
                self.buffer[j] = batch

    def add_batched_chunks(self, batches):
        for b in batches: self.add(b)

    def sample(self, samples):
        return np.random.choice(self.buffer, samples)

class CLEAR:
    def __init__(self, buffer_size):
        self.replay_buffer = ResevoirSampler(buffer_size)
    
    # Used externally only on the first task for collection
    def update_buffer(self, rollout):
        # Extract values (obs, action, logits, etc.) from each env and append to buffer 
        envs = torch.unbind(rollout, dim=2) # Unbind extracts each env column (dim 2) into it's own tensor
        self.replay_buffer.add_batched_chunks(envs)

    # Returns a new rollout mixed with x old replay experiences (usually 1)
    def blend_rollout(self, rollout, samples):
        # Select old experiences from the buffer
        old_exps = self.replay_buffer.sample(samples)

        is_old = torch.zeros(rollout.shape[1], rollout.shape[2])
        is_old[:, -samples] = 1.0
        
        # Replace the last x number of envs with old information
        rollout[:, :, -samples] = old_exps[:, :]

        return torch.unbind(rollout, dim=0), is_old