import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from ncps.torch import CfC
from torch.distributions.categorical import Categorical

# This is needed for the Embedding -> GRU layers. 
# Because the missions are variable length and need padding, the GRU needs to know when and where not to train on padded tokens
# These functions prepare the resulting embedding outputs for input into the GRU.
# use pack_padded_sequence before the GRU, use pad_packed_sequence afterwards to get values 
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

# def prepare_mision_gru(mission):

# Perform mean pooling on embeddings before concatenation with CNN features
def mean_pooling(embeddings, mission_tokens):
    # Average out the embeddings to reduce shape from 3D to 2D for concat later
    # use masked mean pooling for now, but switch to GRU later for complex missions
    # Create a new dimension of floats for each tokens
    # Each token is now either True or False, True for actual token, False for padding
    mask = (mission_tokens != 0).unsqueeze(-1).float()

    # padding_idx in Embedding only prevents training on padded tokens, but embeddings will still contain values for padded tokens. We don't want these
    # Using the mask generated above, we multiply embeddings by mask (0, 1; True, False) which will update the masked layer with embeddings we want, and keeping padded tokens 0
    embedding_mask = embeddings * mask

    # Sum the total number in dim=1 which returns the total over each values under each row
    # so in token[0] = [1, 2, 4], sum = 7
    sum_values = embedding_mask.sum(dim=1) 

    # Get total number of real tokens in each mission. clamp(1e-8) is just in case an entire row contains only 0's. 
    # It'll replace them with 0.00000001 so no divide_by_zero errors happen later 
    count = mask.sum(dim=1).clamp(min=1e-8)

    # Return the average 
    return torch.tensor(sum_values / count)

class SharedEmbedding(nn.Module):
    def __init__(self, vocab_size, hidden_dim, word_embedding_dim, text_embedding_dim):
        super().__init__()
    
        self.embedding = nn.Embedding(vocab_size, word_embedding_dim, padding_idx=0)

        # Initiate the GRU with input size of embeeding_dim (as the embedding layer outputs will be passed here)
        # and with hidden_szie also == to embedding_dim. It's a hyperparam to play with
        self.gru = nn.GRU(word_embedding_dim, text_embedding_dim, batch_first=True) # Set the batch_first to true as the (4) envs will be passed through

        # Process the image with regularization techinques like BatchNormalization (maybe dropout later)
        # Use stride=2 to reduce size instead of using avg_pooling
        self.conv1 = layer_init(nn.Conv2d(3, 16, kernel_size=(2, 2), stride=1, padding=1), std=1.0)    
        self.conv2 = layer_init(nn.Conv2d(16, 32, kernel_size=(2, 2), stride=1, padding=1), std=1.0)
        self.conv3 = layer_init(nn.Conv2d(32, 64, kernel_size=(2, 2), stride=1, padding=1), std=1.0)
        self.max_poool = nn.MaxPool2d((2,2))
        # flattene image for prediction
        self.flatten = nn.Flatten()
        self.linear1 = nn.LazyLinear(hidden_dim)
        
        self.layer_norm = nn.LayerNorm(hidden_dim + text_embedding_dim)
        self.linear2 = layer_init(nn.Linear(hidden_dim + text_embedding_dim, hidden_dim))

        # TODO Store model params in json or something
    
    def forward(self, image, mission):

        # Prepare tokenized and padded mission for Embedding input
        # padded_mission = torch.tensor((mission != 0).sum(dim=1), dtype=torch.int64).cpu()
        # padded_mission = mission.long()
        # padded_mission = padded_mission.to(image.device)

        # embedded_mission = self.embedding(padded_mission)

        # lengths = (mission != 0).sum(dim=1).clamp(min=1).cpu()
        # packed = pack_padded_sequence(embedded_mission, lengths, batch_first=True, enforce_sorted=False)
        # _, hidden_states = self.gru(packed) 
        # print(mission)
        embedded_mission = self.embedding(mission.long())
        # lengths = (mission != 0).sum(dim=1).clamp(min=1).cpu()
        # print(mission.device)
        lengths = (mission != 0).sum(dim=1).clamp(min=1).cpu()

        # packed = pack_padded_sequence(embedded_mission, lengths, batch_first=True, enforce_sorted=False)
        # gru_out, hidden_states = self.gru(packed)

        _, hidden_states = self.gru(embedded_mission)
        
        # Take the last hidden state only
        text_features = hidden_states[-1] 
        
        # Process the image through CNN
        features = F.relu(self.conv1(image))
        features = self.max_poool(features)
        features = F.relu(self.conv2(features))
        features = F.relu(self.conv3(features))

        # Flatten the image for concatenation with text embeddings
        features = self.flatten(features)
        features = F.relu(self.linear1(features))

        # Concatenate image features with the final hidden state from GRU
        shared_features = torch.cat([features, text_features], dim=1)
        # shared_features = self.layer_norm(shared_features)
        return shared_features
        # Pass shared features through a linear layer and return
        # return F.relu(self.linear2(shared_features))
        
class CriticHead(nn.Module):

    def __init__(self, shared_embedding, embedding_dim, hidden_dim=128, use_cfc=False):
        super().__init__()
        
        self.shared_embedding = shared_embedding

        # If we're using a CfC layer for the Actor Model, apply CfC
        # Otherwise use simple MLP
        if use_cfc:
            self.cfc1 = CfC(embedding_dim, hidden_dim, batch_first=True) # batch_first=True is required with the ncps library
        else:
            self.linear1 = layer_init(nn.Linear(embedding_dim, hidden_dim))

        # Final output space of 1 as we want a value, not an action
        self.output = layer_init(nn.Linear(hidden_dim, 1))
    
    def forward(self, image, mission, cfc_states=None):

        x = self.shared_embedding(image, mission)

        # If the model uses a CfC Layer, states need to be managed
        # This results in a different forward structure
        if cfc_states:
            # Reshaping to add the time dimension in the 1st index so we have (batch_size, time_dim, features)
            # Features is all feature maps sequentially combined  
            x = x.unsqueeze(1)

            # Process through CfC layer
            x, new_s1 = self.cfc1(x, cfc_states)
            
            # Remove extra time dimension
            x = x.squeeze(1)

            # TODO Finish implementing 
            return self.output(x)
        
        # Otherwise, process as normal
        x = torch.tanh(self.linear1(x))
        return self.output(x)

class ActorHead(nn.Module):

    def __init__(self, shared_embedding, embedding_dim, action_space, hidden_dim=128, use_cfc=False):
        super().__init__()

        self.shared_embedding = shared_embedding

        # If we're using a CfC layer for the Actor Model, apply CfC
        # Otherwise use simple MLP
        if use_cfc:
            self.cfc1 = CfC(embedding_dim, hidden_dim, batch_first=True) # batch_first=True is required with the ncps library
        else:
            self.linear1 = layer_init(nn.Linear(embedding_dim, hidden_dim), std=1.0)

        # Final output space of action_space to choose an action
        self.output = layer_init(nn.Linear(hidden_dim, action_space))
    
    def forward(self, image, mission, cfc_states=None):

        x = self.shared_embedding(image, mission)

        # If the model uses a CfC Layer, states need to be managed
        # This results in a different forward structure
        if cfc_states:
            # Reshaping to add the time dimension in the 1st index so we have (batch_size, time_dim, features)
            # Features is all feature maps sequentially combined  
            x = x.unsqueeze(1)

            # Process through CfC layer
            x, new_s1 = self.cfc1(x, cfc_states)
            
            # Remove extra time dimension
            x = x.squeeze(1)

            # TODO Finish implementing 
            return self.output(x)
        
        # Otherwise, process as normal
        x = torch.tanh(self.linear1(x))
        return self.output(x)

# Combine the Actor & Critic into a single model 
class Agent(nn.Module):
    
    def __init__(self, action_space, vocab_size, hidden_dim, word_embedding_dim, text_embedding_dim, actor_cfc=False, critic_cfc=False):
        super().__init__()

        self.shared_embedding = SharedEmbedding(vocab_size, hidden_dim, word_embedding_dim, text_embedding_dim)
        
        # Compute the output size dynamically
        with torch.no_grad():
            dummy_image = torch.zeros(1, 3, 7, 7)  # Example MiniGrid obs (C=3,H=7,W=7)
            dummy_text = torch.zeros(1, 10, dtype=torch.long)  # Example mission length = 10
            shared_out = self.shared_embedding(dummy_image, dummy_text)
            embedding_dim = shared_out.shape[1]
        
        # Create the critic model with the same variables as the CfC
        self.critic = CriticHead(self.shared_embedding, embedding_dim, hidden_dim, critic_cfc)

        # Extract the action space of the environment
        # action_space = np.array(envs.single_action_space).prod()

        # Create the actor model. Pass in the envs which will adapt depending on the tasks at hand
        self.actor = ActorHead(self.shared_embedding, embedding_dim, action_space, hidden_dim, actor_cfc)

    def get_value(self, image, mission, states=None):
        return self.critic(image, mission, states)
    
    def get_action_and_value(self, image, mission, states=None, action=None):
        # Pass the image, states, and mission to the model to get an action
        logits = self.actor(image, mission, states)
        probs = Categorical(logits=F.log_softmax(logits, dim=1))
        
        if action is None:
            action = probs.sample()

        return action, probs.log_prob(action), probs.entropy(), self.critic(image, mission, states)