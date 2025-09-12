import time
import torch
import numpy as np
from torch import autograd
import torch.nn.functional as F
from dataclasses import dataclass

# Adpated with ChatGPT for PPO by https://github.com/shivamsaboo17/Overcoming-Catastrophic-forgetting-in-Neural-Networks/blob/master/elastic_weight_consolidation.py
# Includes the fix for sum of squared grads instead of square of summed grads found here: 
# https://github.com/shivamsaboo17/Overcoming-Catastrophic-forgetting-in-Neural-Networks/issues/7#issuecomment-1790310147
class EWC:
    # Agent= Model, Strength = Lambda (In EWC calculation)
    # Lambda is a hyperparam determining how strong consoldiation should be
    def __init__(self, agent, strength):
        self.agent = agent
        self.strength = strength
        self.ewc_loss = 0.0  # Default to 0.0 as the initial task shouldn't be penalized
        self.device = next(self.agent.parameters()).device

    @torch.no_grad()
    def _update_mean_params(self):
        # Iterate over named parameters in the model, and register them to the model in a buffer
        for param_name, param in self.agent.named_parameters():
            _buff = param_name.replace('.', '__')
            key = f'{_buff}_estimated_mean'
            if hasattr(self.agent, key):
                getattr(self.agent, key).copy_(param.data)
            else:
                self.agent.register_buffer(key, param.data.clone())

    # Take the final rollout states and respective actions as input
    # We compare the predicted action of the model with the actual action taken
    def _update_fisher_params(self, b_obs, b_actions, b_dones=None, model_type='mlp', batch_size=64, hidden_state_dim=None, lstm_layers=1):
        self.agent.eval()

        N = b_obs.shape[0]
        acts = b_actions.view(-1).long().to(self.device)
        assert acts.numel() == N, "actions length must match obs length"

        dones_flat = None
        if b_dones is not None:
            dones_flat = b_dones.view(-1).to(self.device)
            assert dones_flat.numel() == N, "dones length must match obs length"

        named_params = list(self.agent.named_parameters())
        fisher_sums = {n: torch.zeros_like(p, device=self.device) for n, p in named_params}

        # Minibatch through the data collecting gradients
        num_chunks = 0
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            if end <= start: continue
            B = end - start

            # Select minibatch
            obs_c   = b_obs[start:end]
            acts_c  = acts[start:end]
            dones_c = dones_flat[start:end] if dones_flat is not None else None

            # Forward per chunk with zeroed recurrent state if needed
            if model_type == 'cfc':
                cfc_state = torch.zeros((B, hidden_state_dim), device=self.device)
                *_, logits_c = self.agent.get_action_and_value(obs_c, cfc_states=cfc_state, dones=dones_c)
            elif model_type == 'lstm':
                lstm_states = (
                    torch.zeros((lstm_layers, B, hidden_state_dim), device=self.device),
                    torch.zeros((lstm_layers, B, hidden_state_dim), device=self.device)
                )
                *_, logits_c = self.agent.get_action_and_value(obs_c, lstm_states=lstm_states, dones=dones_c)
            else:
                *_, logits_c = self.agent.get_action_and_value(obs_c)

            logp = F.log_softmax(logits_c, dim=-1).gather(1, acts_c.view(-1, 1)).squeeze(1)  # [B]
            lp   = logp.mean()

            grads = autograd.grad(lp, [p for _, p in named_params], allow_unused=True, create_graph=False, retain_graph=False)
            for (name, p), g in zip(named_params, grads):
                if g is not None:
                    fisher_sums[name] += (g.detach() ** 2)
            num_chunks += 1

        # Average and store/update buffers
        denom = max(1, num_chunks)
        for name, p in named_params:
            Fdiag = fisher_sums[name] / denom
            _buff = name.replace('.', '__')
            f_key = f'{_buff}_estimated_fisher'
            if hasattr(self.agent, f_key):
                getattr(self.agent, f_key).copy_(Fdiag)
            else:
                self.agent.register_buffer(f_key, Fdiag)

        self.agent.train()

    def register_buffers(self, b_obs, b_actions, b_dones=None, model_type='mlp', batch_size=128, hidden_state_dim=128, lstm_layers=1):
        # Register params are called here to limit how many calls are needed in main training loop
        self._update_fisher_params(b_obs, b_actions, b_dones, model_type, batch_size, hidden_state_dim, lstm_layers)
        self._update_mean_params()

    # If adding decay to weight (for Online EWC), later add strength here for easier decay implementation
    def compute_ewc_loss(self):
        # Iterate over all parameters and penalize large differences from original vs current
        losses = []
        for param_name, param in self.agent.named_parameters():
            __buff = param_name.replace('.', '__')
            mean_key = f'{__buff}_estimated_mean'
            fish_key = f'{__buff}_estimated_fisher'
            if hasattr(self.agent, mean_key) and hasattr(self.agent, fish_key):
                theta0 = getattr(self.agent, mean_key)
                Fdiag  = getattr(self.agent, fish_key)
                losses.append((Fdiag * (param - theta0) ** 2).sum())

        # Summate the current ewc losses and apply the strength hyperparameter
        self.ewc_loss = (self.strength / 2) * sum(losses)
        return self.ewc_loss

# Based on: https://en.wikipedia.org/wiki/Reservoir_sampling. Adapted for PPO rollout chunking
# Stores full trajectories from rollouts
class ReservoirSampler:
    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        self.buffer = []
        self.seen = 0

    def add(self, item):
        self.seen += 1
        if len(self.buffer) < self.capacity:
            self.buffer.append(item)
        else:
            j = np.random.randint(0, self.seen)
            if j < self.capacity:
                self.buffer[j] = item

    def sample(self, count: int):
        count = min(count, len(self.buffer))
        idx = np.random.choice(len(self.buffer), size=count, replace=False)
        return [self.buffer[i] for i in idx]

# Implemented based on the paper: https://arxiv.org/abs/1811.11682
# Adapted with ChatGPT 
class CLEAR:
    def __init__(self, buffer_size):
        self.replay = ReservoirSampler(buffer_size)

    # Add a rollout to buffer (per-env columns)
    def update_buffer(self, rollout):
        num_envs = rollout['actions'].shape[1]

        # states expected as dict: {'cfc': [N,H] or None, 'lstm': (h0[Layers,N,H], c0[Layers,N,H]) or (None,None)}
        states = rollout.get('states', {'cfc': None, 'lstm': (None, None)})
        cfc_all = states.get('cfc', None)
        h_all, c_all = states.get('lstm', (None, None))

        for e in range(num_envs):
            # per-env initial states
            cfc_e = cfc_all[e].contiguous() if (cfc_all is not None) else None
            if h_all is not None and c_all is not None:
                h0_e = h_all[:, e, :].contiguous()
                c0_e = c_all[:, e, :].contiguous()
                lstm_e = (h0_e, c0_e)
            else:
                lstm_e = (None, None)

            item = {
                "obs":               rollout["obs"][:, e].contiguous(),        
                "actions":           rollout["actions"][:, e].contiguous(),     
                "behav_logp":        rollout["behav_logp"][:, e].contiguous(),
                "rewards":           rollout["rewards"][:, e].contiguous(),
                "dones":             rollout["dones"][:, e].contiguous(),   
                "values_behav":      rollout["values_behav"][:, e].contiguous(),
                "behav_logits":      rollout["behav_logits"][:, e].contiguous(),
                "bootstrap_values":  rollout["bootstrap_values"][e].unsqueeze(0).contiguous(),
                "states": {
                    "cfc":  cfc_e,          # [H] or None
                    "lstm": lstm_e,         # (h0[Layers,H], c0[Layers,H]) or (None,None)
                },
            }
            self.replay.add(item)

    # Overwrite last k env columns with replay; also copy initial states into rollout['states']
    # Used for v-trace calculation where blending is required. 
    def blend_rollout(self, rollout, k_replay):
        timesteps, num_envs = rollout["actions"].shape
        k = min(k_replay, len(self.replay.buffer), num_envs)
        replay_items = self.replay.sample(k)

        # mask: 0 = on-policy, 1 = replay
        is_replay = torch.zeros((timesteps, num_envs), device=rollout["actions"].device)
        if k == 0:
            return rollout, is_replay

        # Get starting point for mixing of envs. If len(envs) is 8 and k_replay is 2, replace envs 6 and 7 with replay data 
        start = num_envs - k
        is_replay[:, start:] = 1.0

        # ensure dict container for states
        rs = rollout.get('states', {'cfc': None, 'lstm': (None, None)})
        if not isinstance(rs, dict):
            rs = {'cfc': None, 'lstm': (None, None)}

        for i, item in enumerate(replay_items):
            e = start + i
            rollout["obs"][:, e]           = item["obs"]
            rollout["actions"][:, e]       = item["actions"]
            rollout["behav_logp"][:, e]    = item["behav_logp"]
            rollout["rewards"][:, e]       = item["rewards"]
            rollout["dones"][:, e]         = item["dones"]
            rollout["values_behav"][:, e]  = item["values_behav"]
            rollout["behav_logits"][:, e]  = item["behav_logits"]
            rollout["bootstrap_values"][e] = item["bootstrap_values"].squeeze(0)

            # Copy initial states if containers exist
            init_states = item.get("states", {"cfc": None, "lstm": (None, None)})
            if init_states["cfc"] is not None and rs.get("cfc", None) is not None:
                rs["cfc"][e] = init_states["cfc"]
            if init_states["lstm"][0] is not None and init_states["lstm"][1] is not None:
                lstm_rs = rs.get("lstm", (None, None))
                if lstm_rs[0] is not None and lstm_rs[1] is not None:
                    lstm_rs[0][:, e, :] = init_states["lstm"][0]
                    lstm_rs[1][:, e, :] = init_states["lstm"][1]
                    rs["lstm"] = lstm_rs

        rollout["states"] = rs
        return rollout, is_replay

    # Buffer-only sampler; returns rollout-like batch with N' columns and states
    # Used for PPO + Behaviour Cloning, no v-trace. Data is not mixed here, only sampled for cloning loss defined in CLEAR paper
    def sample_rollouts(self, count):
        k = min(int(count), len(self.replay.buffer))

        # Extract a sample of experiences
        items = self.replay.sample(k)

        # All experiences need to be contiguous. This just saves space
        def stack(key): return torch.stack([it[key] for it in items], dim=1).contiguous()

        batch = {
            "obs":              stack("obs"),
            "actions":          stack("actions"),
            "behav_logp":       stack("behav_logp"),
            "rewards":          stack("rewards"),
            "dones":            stack("dones"),
            "values_behav":     stack("values_behav"),
            "behav_logits":     stack("behav_logits"),
            "bootstrap_values": torch.cat([it["bootstrap_values"] for it in items], dim=0).contiguous(),
        }

        # Extract and load states correctly (if used)
        states = items[0].get("states", {"cfc": None, "lstm": (None, None)})
        cfc = states.get("cfc", None)
        lstm = states.get("lstm", (None, None))

        if cfc is not None:
            cfc_stack = torch.stack([it["states"]["cfc"] for it in items], dim=0).contiguous()
            lstm_stack = (None, None)
        elif lstm[0] is not None and lstm[1] is not None:
            h_list = [it["states"]["lstm"][0] for it in items]
            c_list = [it["states"]["lstm"][1] for it in items]
            h0 = torch.stack(h_list, dim=1).contiguous()
            c0 = torch.stack(c_list, dim=1).contiguous()
            cfc_stack = None
            lstm_stack = (h0, c0)
        else:
            cfc_stack = None
            lstm_stack = (None, None)

        batch["states"] = {"cfc": cfc_stack, "lstm": lstm_stack}

        # Return a batch of experiences for BC calculation
        return batch