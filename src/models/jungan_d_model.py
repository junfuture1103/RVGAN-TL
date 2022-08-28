import torch
from torch import nn
from torch.nn.utils.parametrizations import spectral_norm

import src

class JUNGANDModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.step_1 = nn.Sequential(
            nn.Linear(src.models.x_size, 512),
            nn.LayerNorm(512),
            nn.CELU(),
            nn.Linear(512, 128),
            nn.LayerNorm(128),
            nn.CELU(),
            nn.Linear(128, 32),
            nn.LayerNorm(32),
            nn.CELU(),
            nn.Linear(32, 8),
            nn.LayerNorm(8),
            nn.CELU(),
        )
        self.step_2 = spectral_norm(nn.Linear(8, 1))
        self.hidden_output = None

        self.apply(src.utils.init_weights)

    def forward(self, x: torch.Tensor):
        self.hidden_output = self.step_1(x)
        output = self.step_2(self.hidden_output)
        return output
