from .SIREN import Siren
import torch
from torch import nn


class SIRENNet(nn.Module):
    def __init__(self, input_size = 4, hidden_size = 512, output_size = 4, num_hidden_layers=3, activation='identity',
                 t_min = None,t_max = None,space_min = None,space_max = None):
        super(SIRENNet, self).__init__()

        if t_min is not None:
            self.t_min = t_min
            self.t_max = t_max

            self.space_min = space_min
            self.space_max = space_max
            self.do_normalize_input = True
        else:
            self.do_normalize_input = False

        self.input_size = input_size
        self.output_size = output_size
        
        # Initialize MLP layers
        self.layers = nn.ModuleList()
        for i in range(num_hidden_layers):
            if i == 0:
                hidden_size_0 = input_size
                self.layers.append(Siren(hidden_size_0, hidden_size, w0=30, c=6.))
            else:
                hidden_size_0 = hidden_size
                self.layers.append(Siren(hidden_size_0, hidden_size, w0=30., c=6.))

        self.output_layer = Siren(hidden_size, output_size, w0=30., c=6., activation=nn.Identity())


        
    def forward(self, x):
        # Normalize input to [-1, 1]
        if self.do_normalize_input:
            x_scaled_space = 2*(x[:,0:3]-self.space_min)/(self.space_max-self.space_min)-1
            x_scaled_time = 2*(x[:,-1]-self.t_min)/(self.t_max-self.t_min)*100-1
            x = torch.cat((x_scaled_space,x_scaled_time.view(-1,1)),1)

        alpha = x
        for layer in self.layers:
            alpha = layer(alpha)
        output = self.output_layer(alpha)
        
        
        return output
    
    def print_number_of_parameters(self):
        num_params = sum(p.numel() for p in self.parameters())
        print(f"Number of parameters: {num_params}")
        return num_params