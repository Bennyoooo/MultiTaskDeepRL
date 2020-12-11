#from cs285.infrastructure.psp_layer import *
from cs285.infrastructure.psp_layer2 import *

class HashNet(nn.Module):
    def __init__(self, input_dim, output_dim, layer_size,
                 activation, n_layers,
                 period, key_pick, layer_type):
        super(HashNet, self).__init__()
        self.n_units = layer_size
        self.activation = activation

        layer_units = [np.prod(input_dim)]
        layer_units += [layer_size for i in range(n_layers)]
        layer_units += [output_dim]

        layers = []
        for i in range(len(layer_units)-1):
            layers.append(layer_type(layer_units[i],
                                     layer_units[i+1],
                                     period, key_pick, False))
        self.layers = nn.ModuleList(layers)

class RealHashNet(HashNet):
    def forward(self, x, time):
        preactivations = []
        r = x
        for layer_i, layer in enumerate(self.layers):
            if layer_i > 0:
                r = self.activation(r)
            r = layer(r, time)
            preactivations.append(r)

        return r, None, preactivations



class ComplexHashNet(HashNet):
    def forward(self, x, time):
        preactivations = []
        r_a, r_b = x, torch.zeros_like(x)
        for layer_i, layer in enumerate(self.layers):
            if layer_i > 0:
                r_a = self.activation(r_a)
                r_b = self.activation(r_b)
            r_a = layer(r_a, time)
            r_b = layer(r_b, time)
            preactivations.append(r_a)
            preactivations.append(r_b)
        return r_a, r_b, preactivations
