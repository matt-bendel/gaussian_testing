import torch
from torch import nn


class Discriminator(nn.Module):
    '''
    Multilayer Perceptron.
  '''

    def __init__(self, d):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2*d, 64),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(64, 32),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(32, 16),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(16, 1)
        )

    def forward(self, x, y):
        '''Forward pass'''
        inp = torch.cat([x, y], dim=1)
        return self.layers(inp)
