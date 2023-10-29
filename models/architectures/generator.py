from torch import nn


class Generator(nn.Module):
    '''
    Multilayer Perceptron.
  '''

    def __init__(self, in_mult=2, out_mult=1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(10*in_mult, 64),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(64, 128),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(128, 32),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(32, 10*out_mult)
        )

    def forward(self, x):
        '''Forward pass'''
        return self.layers(x)
