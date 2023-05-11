import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from .posenc import positional_encoding


class SineLayer(nn.Module):
    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.

    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a
    # hyperparameter.

    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)

    def __init__(self, in_features, out_features, bias=True, is_first=False, omega_0=30, init_c=6, first_layer_init_c=1):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.init_weights(init_c=init_c, first_layer_init_c=first_layer_init_c)

    def init_weights(self, init_c, first_layer_init_c):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-first_layer_init_c / self.in_features, first_layer_init_c / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(init_c / self.in_features) / self.omega_0, np.sqrt(init_c / self.in_features) / self.omega_0)

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))

    def forward_with_intermediate(self, input):
        # For visualization of activation distributions
        intermediate = self.omega_0 * self.linear(input)
        return torch.sin(intermediate), intermediate


class TensorfSiren(nn.Module):
    def __init__(self, inChanel, viewpe=6, feape=6, featureC=128):
        super().__init__()

        bias = True
        omega_0 = 30
        hidden_omega_0 = first_omega_0 = omega_0
        outermost_linear = True

        init_c = 6
        first_layer_init_c = 1
        self.in_mlpC = 2 * viewpe * 3 + 2 * feape * inChanel + 3 + inChanel
        self.viewpe = viewpe
        self.feape = feape

        out_features = 3
        in_features = self.in_mlpC
        hidden_features = featureC
        hidden_layers = 1

        self.net = []
        self.net.append(SineLayer(in_features, hidden_features, is_first=True, omega_0=first_omega_0, first_layer_init_c=first_layer_init_c))

        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features, is_first=False, omega_0=hidden_omega_0, init_c=init_c))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)

            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(init_c / hidden_features) / hidden_omega_0, np.sqrt(init_c / hidden_features) / hidden_omega_0)

            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features, is_first=False, omega_0=hidden_omega_0))

        self.net = nn.Sequential(*self.net)

    def forward(self, pts, viewdirs, features):
        indata = [features, viewdirs]
        if self.feape > 0:
            indata += [positional_encoding(features, self.feape)]
        if self.viewpe > 0:
            indata += [positional_encoding(viewdirs, self.viewpe)]

        mlp_in = torch.cat(indata, dim=-1)
        output = self.net(mlp_in)
        output = torch.sigmoid(output)
        return output
