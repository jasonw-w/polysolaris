import torch
import torch.nn as nn
import torch.nn.functional as f
class ResBlock(nn.Module):
    def __init__(self, hidden_size, ):
        super(ResBlock, self).__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.ln1 = nn.LayerNorm(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)
    def forward(self, x):
        identity = x
        x = self.fc1(x)
        x = self.ln1(x)
        x = self.fc2(x)
        x = self.ln2(x)
        x += identity
        return x

class Actor_Critic(nn.Module):
    def __init__(
        self,
        num_input,
        num_output
    ):
        super(Actor_Critic, self).__init__()
        self.fc1 = nn.Linear(num_input, 512)
        self.ln1 = nn.LayerNorm(512)
        self.res_block1 = ResBlock(512)
        self.res_block2 = ResBlock(512)
        # self.critic1 = ResBlock(512)
        self.critic1 = nn.Linear(512, 256)
        self.critic2 = nn.Linear(256, 1)
        
        self.actor_res   = ResBlock(512)
        self.actor_out = nn.Linear(512, num_output)
        self.log_std = nn.Parameter(torch.zeros(num_output))

    def forward(self, x):
        x = self.fc1(x)
        x = self.ln1(x)
        x = f.relu(x)
        x = self.res_block1(x)
        x = f.relu(x)
        x = self.res_block2(x)
        x = f.relu(x)

        action_mean = self.actor_res(x)
        action_mean = f.relu(action_mean)
        action_mean = self.actor_out(action_mean)

        value = self.critic1(x)
        value = f.relu(value)
        value = self.critic2(value)
        # value = self.critic3(value)
        return action_mean, value, self.log_std