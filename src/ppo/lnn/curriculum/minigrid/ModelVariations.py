import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from ncps.torch import CfC

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

class CriticBaseline(nn.Module):
    
    def __init__(self, vocab_size, hidden_dim=128, word_embedding_dim=32, text_embedding_dim=128):
        super().__init__()

        self.concat_dim = hidden_dim + text_embedding_dim
        self.embedding = nn.Embedding(vocab_size, word_embedding_dim, padding_idx=0)

        # Initiate the GRU with input size of embeeding_dim (as the embedding layer outputs will be passed here)
        # and with hidden_szie also == to embedding_dim. It's a hyperparam to play with
        self.gru = nn.GRU(word_embedding_dim, text_embedding_dim, batch_first=True) # Set the batch_first to true as the (4) envs will be passed through

        # Process the image with regularization techinques like BatchNormalization (maybe dropout later)
        # Use stride=2 to reduce size instead of using avg_pooling
        self.conv1 = layer_init(nn.Conv2d(3, 16, kernel_size=(2, 2), stride=2, padding=1))    
        self.conv2 = layer_init(nn.Conv2d(16, 32, kernel_size=(2, 2), stride=2, padding=1))
        self.conv3 = layer_init(nn.Conv2d(32, 64, kernel_size=(2, 2), stride=2, padding=1))
        self.bn2 = nn.BatchNorm2d(64)

        # flattene image for prediction
        self.flatten = nn.Flatten()
        self.linear1 = nn.LazyLinear(hidden_dim)
        
        self.linear2 = layer_init(nn.Linear(self.concat_dim, hidden_dim))
        
        self.output = layer_init(nn.Linear(hidden_dim, 1), std=1.0)

    def forward(self, image, mission_tokens):

        embeddings = self.embedding(mission_tokens)

        # Prepare the padded mission tokens for input to the GRU
        padding_sequences = torch.tensor((mission_tokens != 0).sum(dim=1), dtype=torch.int64).cpu()
        gru_inputs = pack_padded_sequence(embeddings, padding_sequences, batch_first=True)
        
        # Pass the packed_padding_embeddings into the GRU
        # For now, we just want to experiment with passing the final hidden state to concatenate with CNN
        # A better method is available, but this should be fine for now. A sequential understanding is still present
        _, hidden_states = self.gru(gru_inputs)

        # print(f'GRU Outputs: {gru_output}')
        # Process the obs image first
        cnn_features = F.relu(self.conv1(image))
        # cnn_features = self.avg_pooling(cnn_features)
 
        cnn_features = F.relu(self.conv2(cnn_features))
        # cnn_features = self.avg_pooling(cnn_features)
 
        cnn_features = F.relu(self.conv3(cnn_features))
        
        # Flatten the image here
        cnn_features = self.flatten(cnn_features)
        cnn_features = F.relu(self.linear1(cnn_features))

        # Apply mean pooling to embeddings before concat
        # embeddings = mean_pooling(embeddings, mission_tokens)

        x = torch.cat([cnn_features, hidden_states[-1]], dim=1)

        # Pass through another linear layer
        x = F.relu(self.linear2(x))

        return self.output(x)

class ActorBaseline(nn.Module):

    def __init__(self, action_space, vocab_size, hidden_dim=128, word_embedding_dim=32, text_embedding_dim=128):
        super().__init__()

        self.concat_dim = hidden_dim + text_embedding_dim
        self.embedding = nn.Embedding(vocab_size, word_embedding_dim, padding_idx=0)

        # Initiate the GRU with input size of embeeding_dim (as the embedding layer outputs will be passed here)
        # and with hidden_szie also == to embedding_dim. It's a hyperparam to play with
        self.gru = nn.GRU(word_embedding_dim, text_embedding_dim, batch_first=True) # Set the batch_first to true as the (4) envs will be passed through

        # Process the image with regularization techinques like BatchNormalization (maybe dropout later)
        # Use stride=2 to reduce size instead of using avg_pooling
        self.conv1 = layer_init(nn.Conv2d(3, 16, kernel_size=(3, 3), stride=2, padding=1))
        self.conv2 = layer_init(nn.Conv2d(16, 32, kernel_size=(3, 3), stride=2, padding=1))
        self.conv3 = layer_init(nn.Conv2d(32, 64, kernel_size=(3, 3), stride=2, padding=1))

        # flattene image for prediction
        self.flatten = nn.Flatten()
        self.linear1 = nn.LazyLinear(hidden_dim)
        
        self.linear2 = layer_init(nn.Linear(self.concat_dim, hidden_dim))
        
        self.output = layer_init(nn.Linear(hidden_dim, action_space), std=1.0)

    def forward(self, image, mission_tokens):

        embeddings = self.embedding(mission_tokens)

        # Prepare the padded mission tokens for input to the GRU
        padding_sequences = torch.tensor((mission_tokens != 0).sum(dim=1), dtype=torch.int64).cpu()
        gru_inputs = pack_padded_sequence(embeddings, padding_sequences, batch_first=True)
        
        # Pass the packed_padding_embeddings into the GRU
        # For now, we just want to experiment with passing the final hidden state to concatenate with CNN
        # A better method is available, but this should be fine for now. A sequential understanding is still present
        _, hidden_states = self.gru(gru_inputs)

        # print(f'GRU Outputs: {gru_output}')
        # Process the obs image first
        cnn_features = F.relu(self.conv1(image))
        # cnn_features = self.avg_pooling(cnn_features)
 
        cnn_features = F.relu(self.conv2(cnn_features))
        # cnn_features = self.avg_pooling(cnn_features)
 
        cnn_features = F.relu(self.conv3(cnn_features))
        
        # Flatten the image here
        cnn_features = self.flatten(cnn_features)
        cnn_features = F.relu(self.linear1(cnn_features))

        # Apply mean pooling to embeddings before concat
        # embeddings = mean_pooling(embeddings, mission_tokens)

        x = torch.cat([cnn_features, hidden_states[-1]], dim=1)

        # Pass through another linear layer
        x = F.relu(self.linear2(x))

        return self.output(x)

class CriticCfC(nn.Module):

    def __init__(self, obs_space):
        super().__init__()

        # Create the CfC layer used for the actor, but keep the Dense layer in the end as normal passing it through layer_init()
        self.cfc1 = CfC(obs_space, 64)
        self.cfc2 = CfC(64, 64)
        self.output = layer_init(nn.Linear(64, 1), std=0.01)

    def forward(self, x, states):
        # We only want the outputs for now; ignore hidden state
        s1, s2 = states
        x, s1 = self.cfc1(x, s1)
        x, s2 = self.cfc2(x, s2)
        return self.output(x), [s1, s2]

class ActorCfC(nn.Module):

    def __init__(self, action_space, vocab_size, hidden_dim=128, word_embedding_dim=32, text_embedding_dim=128):
        super().__init__()

        self.concat_size = hidden_dim + text_embedding_dim
        # Embedding layer for processing mission text (usually like 'go to red key' where colour and object are randomized each episode)
        # this teaches the model to distinguish colour and objects from one another to 'understand' instructions 
        self.embedding = nn.Embedding(vocab_size, word_embedding_dim, padding_idx=0)

        # Initiate the GRU with input size of embeeding_dim (as the embedding layer outputs will be passed here)
        # and with hidden_szie also == to embedding_dim. It's a hyperparam to play with
        self.gru = nn.GRU(word_embedding_dim, text_embedding_dim, batch_first=True) # Set the batch_first to true as the (4) envs will be passed through

        # Process the image with regularization techinques like BatchNormalization (maybe dropout later)
        self.conv1 = layer_init(nn.Conv2d(3, 16, kernel_size=(2, 2), stride=2, padding=1))
        self.conv2 = layer_init(nn.Conv2d(16, 32, kernel_size=(2,2), stride=2, padding=1))
        self.conv3 = layer_init(nn.Conv2d(32, 64, kernel_size=(2,2), stride=2, padding=1))

        # flattene image for prediction
        self.flatten = nn.Flatten()
        self.linear1 = nn.LazyLinear(hidden_dim) # Can't pass through layer_init due to un-initialized weights until forward pass

        self.linear2 = layer_init(nn.Linear(self.concat_size, hidden_dim))
        # self.dropout = nn.Dropout(0.2)

        # Create the CfC layer used for the actor, but keep the Dense layer in the end as normal passing it through layer_init()
        self.cfc1 = CfC(hidden_dim, hidden_dim, batch_first=True)
        self.cfc2 = CfC(hidden_dim, hidden_dim, batch_first=True)
        self.output = layer_init(nn.Linear(hidden_dim, action_space), std=1.0)

    def forward(self, image, states, mission_tokens):
        # State storage
        s1, _ = states

        # Pass the mission into the embedding layer for a vector representation
        embeddings = self.embedding(mission_tokens)

        # Prepare the padded mission tokens for input to the GRU
        padding_sequences = torch.tensor((mission_tokens != 0).sum(dim=1), dtype=torch.int64).cpu()
        gru_inputs = pack_padded_sequence(embeddings, padding_sequences, batch_first=True)
        
        # Pass the packed_padding_embeddings into the GRU
        _, hidden_states = self.gru(gru_inputs)

        # Process the obs image first
        cnn_features = F.relu(self.conv1(image))
        cnn_features = F.relu(self.conv2(cnn_features))
        cnn_features = F.relu(self.conv3(cnn_features))
        
        # Flatten the image here
        cnn_features = self.flatten(cnn_features)
        cnn_features = F.relu(self.linear1(cnn_features))

        # Apply mean pooling to embeddings before concat
        # embeddings = mean_pooling(embeddings, mission_tokens)

        # COncatenate the flattened CNN layers and the encoded mission into one layer for processing in the CfC layer 
        # this results in a (batch_size, cnn_features + embedding space) shape
        x = torch.cat([cnn_features, hidden_states[-1]], dim=1)
        
        # Apply linear layer before CfC
        x = F.relu(self.linear2(x))

        # Reshaping to add the time dimension in the 1st index so we have (batch_size, time_dim, features)
        # Features is all feature maps sequentially combined  
        x = x.unsqueeze(1)

        # Pass features to model along with state_1
        x, new_s1 = self.cfc1(x, s1)
        # Pass updated features along with new_states from self.cfc1
        # and store new_s2
        x, new_s2 = self.cfc2(x, new_s1)

        # Remove the time dimension we added earlier as it's not needed anymore
        x = x.squeeze(1)

        # Pass batch into output layer (we don't need the temporal states for a fully connected)
        # return states as well for update epoch training
        return self.output(x), [new_s1, new_s2]
