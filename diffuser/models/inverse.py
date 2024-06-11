import torch
import torch.nn as nn
import torch.nn.functional as F

class InverseModel(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64):
        super(InverseModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim//2)
        self.fc3 = nn.Linear(hidden_dim//2, hidden_dim//4)
        self.fc4 = nn.Linear(hidden_dim//4, hidden_dim//8)
        self.fc5 = nn.Linear(hidden_dim//8, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x
    
    def loss(self, observations, actions):
        # observations: [batch_size, traj_dim, observation_dim]
        # actions: [batch_size, traj_dim, action_dim]
        
        # make observations and actions 2D
        observations = observations.view(observations.shape[0], -1)
        actions = actions.view(actions.shape[0], -1)

        pred = self(observations)
        loss = F.mse_loss(pred, actions)
        return loss, {'mse_loss': loss.item()}