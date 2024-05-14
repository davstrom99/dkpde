import math
import torch
from torch import nn
import torch.nn.functional as F

# sin activation
class Sine(nn.Module):
    def __init__(self, w0 = 30.):
        super().__init__()
        self.w0 = w0
    def forward(self, x):
        return torch.sin(self.w0 * x)

# siren layer
class Siren(nn.Module):
    def __init__(self, dim_in, dim_out, w0 = 30., c = 6., is_first = False, use_bias = True, activation = None):
        super().__init__()
        self.dim_in = dim_in
        self.is_first = is_first

        weight = torch.zeros(dim_out, dim_in)
        bias = torch.zeros(dim_out) if use_bias else None
        self.init_(weight, bias, c = c, w0 = w0)

        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias) if use_bias else None
        self.activation = Sine(w0) if activation is None else activation

    def init_(self, weight, bias, c, w0):
        dim = self.dim_in

        w_std = (1 / dim) if self.is_first else (math.sqrt(c / dim) / w0)
        weight.uniform_(-w_std, w_std)

        if bias is not None:
            bias.uniform_(-w_std, w_std)

    def forward(self, x):
        out =  F.linear(x, self.weight, self.bias)
        out = self.activation(out)
        return out

# siren resnet
class SirenResnet(nn.Module):
    def __init__(self, dim_in=3, dim_out=1, num_resnet_blocks=3, num_layers_per_block=2, num_neurons=128,
                 device='cuda', tune_beta=False):
        super(SirenResnet, self).__init__()

        self.num_resnet_blocks = num_resnet_blocks
        self.num_layers_per_block = num_layers_per_block
        self.device = device

        if tune_beta:
            self.beta0 = nn.Parameter(torch.ones(1, 1))
            self.beta = nn.Parameter(torch.ones(self.num_resnet_blocks, self.num_layers_per_block))

        else:
            self.beta0 = torch.ones(1, 1)
            self.beta = torch.ones(self.num_resnet_blocks, self.num_layers_per_block)

        self.resnet_activation =  Sine(w0 = 30)

        self.first = Siren(dim_in, num_neurons, w0 = 30, is_first = True)
        self.last = Siren(num_neurons, dim_out, w0 = 30, activation= nn.Identity())

        self.resblocks = nn.ModuleList([
            nn.ModuleList([Siren(num_neurons, num_neurons, w0 = 30, activation= nn.Identity()) for _ in range(num_layers_per_block)])
            for _ in range(num_resnet_blocks)])

    def forward(self, x):
        x = self.first(x)

        for i in range(self.num_resnet_blocks):
            z = self.resnet_activation(self.beta[i][0] * self.resblocks[i][0](x))
            for j in range(1, self.num_layers_per_block):
                z = self.resnet_activation(self.beta[i][j] * self.resblocks[i][j](z))
            x = z + x

        out = self.last(x)

        return out
    def model_capacity(self):
        """
        Prints the number of parameters and the number of layers in the network
        """
        number_of_learnable_params =sum(p.numel() for p in self.parameters() if p.requires_grad)
        num_layers = len(list(self.parameters()))
        print("\n\nThe number of layers in the model: %d" % num_layers)
        print("\nThe number of learnable parameters in the model: %d" % number_of_learnable_params)



