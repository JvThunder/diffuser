import torch
import torch.nn as nn
import torch.nn.functional as F

class InverseModel(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64):
        super(InverseModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def loss(self, observations, actions):
        # observations: [batch_size, traj_dim, observation_dim]
        # actions: [batch_size, traj_dim, action_dim]
        
        # make observations and actions 2D
        observations = observations.view(-1, observations.shape[-1])
        actions = actions.view(-1, actions.shape[-1])

        pred = self(observations)
        loss = F.mse_loss(pred, actions)
        return loss, {'mse_loss': loss.item()}