from torch import nn


class GeneratorLinear(nn.Module):
    '''
    Multilayer Perceptron.
  '''

    def __init__(self, d):
        super().__init__()
        self.layers_y = nn.Sequential(
            nn.Flatten(),
            nn.Linear(d, d),
        )

        self.layers_z = nn.Sequential(
            nn.Flatten(),
            nn.Linear(d, d, bias=False),
        )

    def forward(self, y, z):
        '''Forward pass'''
        return self.layers_y(y) + self.layers_z(z)
