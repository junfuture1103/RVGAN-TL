import torch
from torch import nn
from torch.nn.utils.parametrizations import spectral_norm

import src


class SNGANGModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.process = nn.Sequential(
            spectral_norm(nn.Linear(src.models.x_size, 32)),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Linear(32, 128)),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Linear(128, 512)),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Linear(512)),
            nn.LeakyReLU(0.2),
        )

        self.apply(src.utils.init_weights)

    def forward(self, x: torch.Tensor):
        x_hat = self.process(x)
        return x_hat
