import torch
import torch as th
import numpy as np
import torch.nn.functional as F

from typing import Dict
from torch import nn
from torch.optim import Adam
# from tianshou.data import to_torch, Batch

class WorldModel(nn.Module):
    def __init__(self, num_agent, layer_num, state_shape, hidden_units=128, device='cpu', wm_noise_level=0.0):
        super().__init__()
        self.device = device
        self.state_shape = state_shape
        # plus one for the action
        self.model = [
            nn.Linear(np.prod(state_shape) + 1, hidden_units), # action:[0,1,2,3,4,5,enemy*num]
            nn.ReLU()]
        for i in range(layer_num - 1):
            self.model += [nn.Linear(hidden_units, hidden_units), nn.ReLU()]
        self.model += [nn.Linear(hidden_units, np.prod(state_shape))]
        self.num_agent = num_agent
        self.model = nn.Sequential(*self.model)
        self.optim = Adam(self.model.parameters(), lr=1e-3)
        self.wm_noise_level = wm_noise_level

    def forward(self, s, **kwargs):
        s = s.to(self.device)
        batch = s.shape[0]
        s = s.view(batch, -1)
        logits = th.tanh(self.model(s))
        if self.wm_noise_level != 0.0:
            logits += torch.normal(torch.zeros(logits.size()), self.wm_noise_level).to(logits.device)
        return logits



