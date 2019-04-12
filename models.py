import torch
import torch.nn as nn
import torch.nn.functional as F

"""
the input x in both networks should be [o, g], where o is the observation and g is the goal.

"""

# define the actor network
class actor(nn.Module):
    def __init__(self, env_params, input_num):
        super(actor, self).__init__()
        self.max_action = env_params['action_max']
        self.fc1 = nn.Linear(input_num, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.action_out = nn.Linear(64, env_params['action'])

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        actions = self.max_action * torch.tanh(self.action_out(x))

        return actions

class critic(nn.Module):
    def __init__(self, env_params, input_num):
        super(critic, self).__init__()
        self.max_action = env_params['action_max']
        self.fc1 = nn.Linear(input_num, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.q_out = nn.Linear(64, 1)

    def forward(self, x, actions):
        x = torch.cat([x, actions / self.max_action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        q_value = self.q_out(x)

        return q_value

class actor_recurrent(nn.Module):
    def __init__(self, env_params, input_num):
        super(actor_recurrent, self).__init__()
        self.max_action = env_params['action_max']
        self.hidden_size = 64
        self.gru = nn.GRU(input_num, self.hidden_size)
        self.fc1 = nn.Linear(self.hidden_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.action_out = nn.Linear(64, env_params['action'])

    def _forward_gru(self, x, hxs):
        # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
        x, hxs = self.gru(
            x,
            hxs
        )

        return x, hxs

    def forward(self, x, hidden):
        x, hidden = self._forward_gru(x, hidden)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        actions = self.max_action * torch.tanh(self.action_out(x))

        return actions, hidden

class critic_recurrent(nn.Module):
    def __init__(self, env_params, input_num):
        super(critic_recurrent, self).__init__()
        self.max_action = env_params['action_max']
        self.hidden_size = 64
        self.gru = nn.GRU(input_num, self.hidden_size)
        self.fc1 = nn.Linear(self.hidden_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.q_out = nn.Linear(64, 1)

    def _forward_gru(self, x, hxs):
        # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
        N = hxs.size(0)
        T = int(x.size(0) / N)

        # unflatten
        x = x.view(T, N, x.size(1))

        x, hxs = self.gru(
            x,
            hxs.view(1, -1, 1)
        )

        # flatten
        x = x.view(T * N, -1)
        hxs = hxs.squeeze(0)

        return x, hxs

    def forward(self, x, actions, hidden):
        x = torch.cat([x, actions / self.max_action], dim=1)
        x, hidden = self._forward_gru(x, hidden)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        q_value = self.q_out(x)

        return q_value, hidden
