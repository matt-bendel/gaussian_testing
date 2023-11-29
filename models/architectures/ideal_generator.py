from torch import nn

class IdealGenerator(nn.Module):
    '''
    Multilayer Perceptron.
  '''

    def __init__(self, d, Z_weights, Y_weights, Y_bias):
        super().__init__()
        lin_y = nn.Linear(d, d)
        lin_y.weight.copy_(Y_weights)
        lin_y.bias.copy_(Y_bias[:, 0])

        self.layers_y = nn.Sequential(
            nn.Flatten(),
            lin_y
        )

        lin_z = nn.Linear(d, d, bias=False)
        lin_z.weight.copy_(Z_weights)

        self.layers_z = nn.Sequential(
            nn.Flatten(),
            lin_z
        )

    def forward(self, y, z):
        '''Forward pass'''
        return self.layers_y(y) + self.layers_z(z)
