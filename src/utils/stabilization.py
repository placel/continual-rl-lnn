import torch
import numpy as np
from torch import autograd
import torch.nn.functional as F
import time

# TODO
# create EWC, SI, and CLEAR methods/classes
# Adpoted for PPO by https://github.com/shivamsaboo17/Overcoming-Catastrophic-forgetting-in-Neural-Networks/blob/master/elastic_weight_consolidation.py
class ElasticWeightConsolidation:

    # Agent= Model, Strength = Lambda (In EWC calculation)
    # Lambda is a hyperparam determining how strong consoldiation should be
    def __init__(self, agent, strength):
        self.agent = agent
        self.strength = strength
        self.has_cfc = hasattr(self.agent, 'cfc1')
        # EWC Loss is only updated after every task. 
        # Just store it as variable instead of re-computing it every PPO update
        self.ewc_loss = 0.0 # Default to 0.0 as the initial task shouldn't be penalized
        self.device = next(self.agent.parameters()).device

    # Take the final rollout states and respective actions as input
    # We compare the predicted action of the model with the actual action taken
    def _update_fisher_params(self, b_obs, b_actions, b_states=None, num_steps=128, num_envs=4):        
        # Calculate how many times the model collects softmax values
        # If num_step=128 and num_envs=4, then the result is 512
        # However, since 4 envs are being processed at a time, 512 * 4 = 2048 total states will be seen (experiment with this)
        num_batch = num_steps * num_envs

        # Batch process the states instead of iterating through
        _, _, _, _, _, logits = self.agent.get_action_and_value(b_obs, b_states)
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
            if param is not None:
                print(f"Creating fisher buffer for: {_buff_param_name}")
                self.agent.register_buffer(_buff_param_name + '_estimated_fisher', param.data.clone() ** 2)
            else:
                print(f"No gradient for parameter: {_buff_param_name}")
        
    # def _update_fisher_params(self, b_obs, b_actions, b_states, num_steps, num_envs):
    #     num_batch = num_steps * num_envs
        
    #     # Process the entire batch at once
    #     _, _, _, _, _, logits = self.agent.get_action_and_value(b_obs, b_states)
    #     output = F.log_softmax(logits, dim=1)
        
    #     batch_indices = torch.arange(num_batch, device=b_obs.device)
    #     log_liklihoods = output[batch_indices, b_actions]
    #     log_liklihoods = log_liklihoods.mean()
        
    #     # Collect parameters that actually get gradients
    #     used_params = []
    #     used_param_names = []
        
    #     for param_name, param in self.agent.named_parameters():
    #         if param.requires_grad:
    #             try:
    #                 # Test if this parameter gets a gradient
    #                 grad = autograd.grad(log_liklihoods, [param], retain_graph=True, allow_unused=True)[0]
    #                 if grad is not None:
    #                     used_params.append(param)
    #                     used_param_names.append(param_name)
    #             except RuntimeError:
    #                 # Parameter not connected to the computation graph
    #                 continue
        
    #     print(f"Found {len(used_params)} parameters with gradients out of {len(list(self.agent.parameters()))} total")
        
    #     if len(used_params) == 0:
    #         print("No parameters have gradients! Something is wrong with the computation graph.")
    #         return
        
    #     # Now compute gradients for only the used parameters
    #     grad_log_liklihood = autograd.grad(
    #         log_liklihoods, 
    #         used_params, 
    #         create_graph=False,
    #         retain_graph=False
    #     )
        
    #     # Create Fisher buffers only for parameters that were actually used
    #     buffers_created = 0
    #     for param_name, grad in zip(used_param_names, grad_log_liklihood):
    #         _buff_param_name = param_name.replace('.', '__')
    #         self.agent.register_buffer(_buff_param_name + '_estimated_fisher', grad.data.clone() ** 2)
    #         buffers_created += 1
    #         print(f"Created Fisher buffer for: {param_name}")
        
    #     print(f"Total Fisher buffers created: {buffers_created}")
    
    def _update_mean_params(self):
        # Iterate over named parameters in the model, and register them to the model in a buffer 
        for param_name, param in self.agent.named_parameters():
            _buff_param_name = param_name.replace('.', '__')
            self.agent.register_buffer(_buff_param_name + '_estimated_mean', param.data.clone())

    def register_buffers(self, b_obs, b_actions, b_states=None, num_steps=128, num_envs=4):
        # Register params are called here to limit how many calls are needed in main training loop
        self._update_fisher_params(b_obs, b_actions, b_states, num_steps, num_envs)
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