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
        self.layers = []
        for i in range(self.d):
            self.layers.append(
                nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(2*d, d)
                )
            )

    def forward(self, y, x_hat):
        '''Forward pass'''
        return torch.cat([layer(torch.cat([y, x_hat], dim=-1)).unsqueeze(1) for layer in self.layers], dim=1)