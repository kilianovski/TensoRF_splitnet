import torch
from torch import nn
import torch.nn.functional as F

import os
import math
import numpy as np

from .posenc import positional_encoding


def init_siren(W, fan_in, omega=30, init_c=24, flic=2, is_first=False):
    if is_first:
        c = flic / fan_in
    else:
        c = np.sqrt(init_c / fan_in) / omega
    W.uniform_(-c, c)


class SplitLayer(nn.Module):
    def __init__(self, input_dim, output_dim, m=1.0, omegas=(1, 1, 1.0, 1), use_bias=True):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim * 4, bias=use_bias)
        self.dropout = nn.Dropout(0)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.m = m
        self.omegas = omegas
        # self.init_weights()

    def init_weights(self):
        s = self.output_dim
        fan_in = self.input_dim

        W = self.linear.weight.data
        self.linear.bias.data.uniform_(0, 0)
        c = np.sqrt(1 / fan_in) / self.omegas[0]
        # print('the c', c)
        W[:s].uniform_(-c, c)
        # init_siren(W[:s], init_c=1, fan_in=fan_in, is_first=False, omega=self.omegas[0])

        init_siren(W[s : s * 2], init_c=6, fan_in=fan_in, is_first=False, omega=self.omegas[1])
        init_siren(W[s * 2 :], fan_in=fan_in, is_first=False, omega=self.omegas[2])

    def forward(self, x):
        h, acts = self.forward_with_activations(x)
        return h

    def forward_with_activations(self, x):
        preact = self.linear(x)
        preacts = preact.chunk(4, dim=-1)
        preacts = list(preacts)

        for i in range(len(preacts)):
            preacts[i] = self.omegas[i] * preacts[i]

        preact_tanh, preact_sigmoid, preact_sin, preact_cos = preacts
        act_tanh, act_sigmoid, act_sin, act_cos = preact_tanh.tanh(), preact_sigmoid.sigmoid(), preact_sin.sin(), preact_cos.cos()
        h = act_tanh * act_sigmoid * act_sin * act_cos

        h = h * self.m

        return h, [x, preact, preact_tanh, preact_sigmoid, preact_sin, preact_cos, act_tanh, act_sigmoid, act_sin, act_cos]


class SimpleSplitNet(nn.Module):
    def __init__(self, in_features, hidden_layers, out_features, outermost_linear=False, use_bias=True, omegas=(1, 1, 1.0, 1), m=1.0):
        super().__init__()

        if not hasattr(m, "__len__"):
            m = [m] * (len(hidden_layers) + 2)

        is_layerwise_omegas = hasattr(omegas[0], "__len__")

        if not is_layerwise_omegas:
            omegas = [omegas] * (len(hidden_layers) + 2)

        net = [SplitLayer(in_features, hidden_layers[0], use_bias=use_bias, m=m[0], omegas=omegas[0])]

        fan_out = hidden_layers[0]
        for i, (fan_in, fan_out) in enumerate(zip(hidden_layers, hidden_layers[1:])):
            net.append(SplitLayer(fan_in, fan_out, use_bias=use_bias, m=m[i + 1], omegas=omegas[i + 1]))

        if outermost_linear:
            net.append(nn.Linear(fan_out, out_features))
        else:
            net.append(SplitLayer(fan_out, out_features, m=m[-1]))

        self.net = nn.Sequential(*net)

    def forward(self, x):
        return self.net(x)

    def forward_with_activations(self, x):
        h = x
        intermediate_acts = []

        for layer in self.net:
            if isinstance(layer, SplitLayer):
                h, acts = layer.forward_with_activations(h)
            else:
                h = layer(h)
                acts = []

            intermediate_acts.append((h, acts))

        return h, intermediate_acts


class ParallelSplitNet(nn.Module):
    def __init__(self, inChanel, viewpe=6, feape=6, featureC=42, model_configs=None):
        super().__init__()

        if model_configs is None:
            model_configs = [{"hidden_layers": [featureC, featureC], "m": [10.0, 10.0, 1.0], "outermost_linear": True}]

        self.in_mlpC = 2 * viewpe * 3 + 2 * feape * inChanel + 3 + inChanel
        self.viewpe = viewpe
        self.feape = feape

        in_features = self.in_mlpC
        out_features = 3

        self.networks = nn.ModuleList([SimpleSplitNet(**k, in_features=in_features, out_features=out_features) for k in model_configs])

    def forward(self, pts, viewdirs, features):
        indata = [features, viewdirs]
        if self.feape > 0:
            indata += [positional_encoding(features, self.feape)]
        if self.viewpe > 0:
            indata += [positional_encoding(viewdirs, self.viewpe)]
        mlp_in = torch.cat(indata, dim=-1)

        rgb = 0

        for net in self.networks:
            rgb = rgb + net(mlp_in).sigmoid()

        return rgb
