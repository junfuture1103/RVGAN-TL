import torch
from torch import nn
from torch.nn.utils.parametrizations import spectral_norm

import src

class FDGANDModel(nn.Module):
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

class FDGAND_ABLTEST_Model(nn.Module):
    def __init__(self, ablation_config=None):
        super().__init__()
        
        # Ablation 실험을 위한 기본 설정
        if ablation_config is None:
            ablation_config = {
                'use_spectral_norm': True,
                'use_celu': True,
                'use_layernorm': True,
                'step_1_layers': [512, 128, 32, 8]
            }
        
        layers = []
        input_size = src.models.x_size
        for layer_size in ablation_config['step_1_layers']:
            layers.append(nn.Linear(input_size, layer_size))
            if ablation_config['use_layernorm']:
                layers.append(nn.LayerNorm(layer_size))
            if ablation_config['use_celu']:
                layers.append(nn.CELU())
            else:
                layers.append(nn.ReLU())
            input_size = layer_size
        
        self.step_1 = nn.Sequential(*layers)
        
        if ablation_config['use_spectral_norm']:
            self.step_2 = spectral_norm(nn.Linear(input_size, 1))
        else:
            self.step_2 = nn.Linear(input_size, 1)
        
        self.hidden_output = None

        self.apply(src.utils.init_weights)

    def forward(self, x: torch.Tensor):
        self.hidden_output = self.step_1(x)
        output = self.step_2(self.hidden_output)
        return output
