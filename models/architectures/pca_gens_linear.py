from torch import nn
import torch

class MMSELinear(nn.Module):
    '''
    Multilayer Perceptron.
  '''

    def __init__(self, d):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(d, d),
        )

    def forward(self, y):
        '''Forward pass'''
        return self.layers(y)

class PCALinear(nn.Module):
    '''
    Multilayer Perceptron.
  '''

    def __init__(self, d):
        super().__init__()
        self.d = d
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2*d, d * d),
        )

    def forward(self, y, x_hat):
        '''Forward pass'''
        return self.layers(torch.cat([y, x_hat], dim=-1)).view(y.shape[0], self.d, -1)