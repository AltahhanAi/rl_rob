'''
Abdulrahman Altahhan (c) 2026
Version: 3.8
Educational code for teaching RL (DQN and related methods).
Permission required for redistribution or research/commercial use.
'''

import time
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import rand
from collections import deque
from itertools import islice
import matplotlib.cm as cm
import matplotlib.animation as animation
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from IPython.display import clear_output
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from collections import deque
from random import sample
from math import prod
from torch.distributions import Normal
# ================================== NN Infrastructure ==========================================
class nnModel(nn.Module):
    def __init__(self, inp_dim, trunk=[(8, 4, 2), (4, 4, 4)], nF=32, out_dim=3, α=1e-4, net_str='', final_bias=True, **kw):
        super().__init__()
        self.layers = nn.ModuleList()
        self.final_bias = final_bias
        self.trunk = trunk
        self.CNN = any(isinstance(h, tuple) and len(h) > 1 for h in trunk)
        feat_in = inp_dim[0]
        feat_in = self.append_trunk(feat_in, inp_dim)
        self.inp_dim = inp_dim
        self.layers.append(nn.Linear(feat_in, nF)) if nF else None
        self.layers.append(nn.Linear(nF if nF else feat_in, out_dim, bias=self.final_bias))
        self.α = α
        self.update_msg  = 'update %s network weights...........! at %d'
        self.saving_msg  = 'saving %s network weights to disk...!'
        self.loading_msg = 'loading %s network weights from disk...!'
        self.net_str = net_str

    def append_trunk(self, feat_in, inp_dim):
        for feat_out in self.trunk:
            CNN_layer = isinstance(feat_out, tuple) and len(feat_out) > 1
            (layer, feat_out) = (nn.Conv2d, feat_out) if CNN_layer else (nn.Linear, (feat_out,))
            if feat_out[0]:
                self.layers.append(layer(feat_in, *feat_out))
                feat_in = feat_out[0]
        self.flat_idx = None
        if self.CNN:
            self.flat_idx = len(self.trunk)
            self.layers.append(nn.Flatten())
            feat_in = self.flatten_dim(inp_dim)
        return feat_in

    def forward(self, x):
        for l, layer in enumerate(self.layers[:-1]):
            x = F.relu(layer(x)) if l != self.flat_idx else layer(x)
        return self.layers[-1](x)

    # def forward(self, x):
    #     for l, layer in enumerate(self.layers[:-1]):
    #         x = F.relu(layer(x)) if l != self.flat_idx else layer(x)
    #     out = self.layers[-1](x)
    #     return out.squeeze(-1) if out.shape[-1] == 1 else out
        
    def flatten_dim(self, inp_dim):
        with torch.no_grad():
            x = torch.zeros(1, *inp_dim)
            for layer in self.layers[:self.flat_idx]: x = layer(x)
            return x.view(1, -1).shape[1]

    # def fit(self, vals, targets, exact=True):
    #     self.train()
    #     self.optim.zero_grad()
    #     vals    = vals.squeeze(-1)    # to avoid having to call unsqueeze(1) elsewhere
    #     targets = targets.squeeze(-1) # to avoid having to call unsqueeze(1) elsewhere
    #     if exact: loss = .5 * F.mse_loss(vals, targets, reduction='sum') / len(vals)
    #     else:     loss = .5 * F.mse_loss(vals, targets)
    #     loss.backward()
    #     clip_grad_norm_(self.parameters(), max_norm=1.0) if self.CNN else None
    #     self.optim.step()
    #     return loss.item()
        
    def fit(self, vals, targets, exact=True):
        self.train()
        self.optim.zero_grad()
        if exact: loss = .5 * F.mse_loss(vals, targets, reduction='sum') / len(vals)
        else:     loss = .5 * F.mse_loss(vals, targets)
        loss.backward()
        clip_grad_norm_(self.parameters(), max_norm=1.0) if self.CNN else None
        self.optim.step()
        return loss.item()

    def predict(self, s, state_dim):
        if not isinstance(s, torch.Tensor):
            s = torch.tensor(s, dtype=torch.float32)
        s_batch = s.ndim > len(state_dim)
        if not s_batch: s = s.unsqueeze(0)
        self.eval()
        with torch.no_grad():
            return self(s) if s_batch else self(s)[0]


    def init_weights(self, final_v0=None, skip_from=None):
        gain = init.calculate_gain('relu')
        for i, layer in enumerate(self.layers):
            if isinstance(layer, (nn.Linear, nn.Conv2d)):
                is_final = (i == len(self.layers) - 1)
                if skip_from is not None and i >= skip_from:
                    continue
                if final_v0 is not None and is_final:
                    init.constant_(layer.weight, final_v0)
                else:
                    init.xavier_normal_(layer.weight, gain=gain)
                if layer.bias is not None:
                    init.zeros_(layer.bias)
        if self.CNN: self.optim = optim.Adam(self.parameters(), lr=self.α)
        else:        self.optim = optim.SGD(self.parameters(),  lr=self.α)

    def load_weights(self, net_str):
        print(self.loading_msg % net_str)
        self.load_state_dict(torch.load(net_str))

    def save_weights(self, net_str):
        print(self.saving_msg % net_str)
        torch.save(self.state_dict(), f'{net_str}.weights.pt')

    def set_weights(self, source_model, net_str, t):
        self.load_state_dict(source_model.state_dict())

    def print_model_summary(self, net_str):
        print( "╭───────────────────────────────────────────────────────────────────────────────────────╮")
        print(f"│          Model Architecture: {net_str:<57}│")
        print( "├────┬───────────────────────────┬─────────────────┬─────────────────────────┬──────────┤")
        print( "│ Id │ Layer                     │ Output Shape    │ Parameters              │Trainable │")
        print( "├────┼───────────────────────────┼─────────────────┼─────────────────────────┼──────────┤")
        total_params = 0
        bias_params  = 0
        prev_shape   = tuple(self.inp_dim)
        for i, layer in enumerate(self.layers):
            param_count = sum(p.numel() for p in layer.parameters())
            trainable   = any(p.requires_grad for p in layer.parameters())
            total_params += param_count
            layer_bias   = sum(p.numel() for name, p in layer.named_parameters() if name == 'bias')
            bias_params  += layer_bias
            param_str    = f"{param_count:>10,} ({layer_bias:>3,} bias)"
            if isinstance(layer, nn.Conv2d):
                kH, kW   = layer.kernel_size
                iC, oC   = layer.in_channels, layer.out_channels
                sH, sW   = layer.stride
                oH = (prev_shape[1] - kH) // sH + 1
                oW = (prev_shape[2] - kW) // sW + 1
                prev_shape = (oC, oH, oW)
                detail     = f"Conv2d ({kH}x{kW}x{iC}x{oC}+{layer_bias})"
                shape_str  = f"({oC}, {oH}, {oW})"
            elif isinstance(layer, nn.Flatten):
                flat       = prod(prev_shape)
                detail     = "Flatten"
                shape_str  = f"({flat},)"
                prev_shape = (flat,)
            elif isinstance(layer, nn.Linear):
                prev_shape = (layer.out_features,)
                detail     = f"Linear ({layer.in_features}x{layer.out_features})"
                shape_str  = f"({layer.out_features},)"
            else:
                detail    = type(layer).__name__
                shape_str = ""
            print(f"│ {i:2d} │ {detail:<25} │ {shape_str:<15} │ {param_str:<23} │ {'Yes' if trainable else 'No ':<8} │")
        print("╰───────────────────────────────────────────────────────────────────────────────────────╯")
        print(f"Total parameters: {total_params:,} of which {bias_params:,} are bias")

# ==================== Split head models for Dueling and Actor-Critic ===========================

class nnSplitModel(nnModel):
    def __init__(self, head1_dim, head2_dim, **kw):
        super().__init__(out_dim=head1_dim, **kw)
        feat_in    = self.layers[-1].in_features
        self.layers = self.layers[:-1]                         # remove final layer
        self.head1  = nn.Linear(feat_in, head1_dim, bias=self.final_bias)
        self.head2  = nn.Linear(feat_in, head2_dim, bias=self.final_bias)
        self.layers.append(self.head1)                         # register for summary
        self.layers.append(self.head2)                         # register for summary
        self.head_idx = len(self.layers) - 2                   # index where heads start

    def init_weights(self, final_v0=None, head2_v0=None):
        super().init_weights(skip_from=self.head_idx)  # trunk only
        gain = init.calculate_gain('relu')
        for head, v0 in [(self.head1, final_v0),
                         (self.head2, head2_v0)]:
            init.constant_(head.weight, v0) if v0 is not None else init.xavier_normal_(head.weight, gain=gain)
            if head.bias is not None: init.zeros_(head.bias)
            
    def forward(self, x):
        for l, layer in enumerate(self.layers[:self.head_idx]):
            x = F.relu(layer(x)) if l != self.flat_idx else layer(x)
        self._trunk_out = x
        return self.head1(x), self.head2(x)

# ===============================================================================================
class nnDuelModel(nnSplitModel):
    def __init__(self, out_dim, **kw):
        super().__init__(head1_dim=1, head2_dim=out_dim, **kw)

    def forward(self, x):
        V, A = super().forward(x)
        return V + (A - A.mean(dim=1, keepdim=True))

# ===============================================================================================
class nnACSharedModel(nnSplitModel):
    def __init__(self, out_dim, αv=1e-3, αq=1e-4, **kw):
        super().__init__(head1_dim=1, head2_dim=out_dim, **kw)
        self.optim = optim.Adam([
            {'params': list(self.layers[:self.head_idx].parameters()) +
                       list(self.head1.parameters()), 'lr': αv},  # trunk + V head
            {'params': list(self.head2.parameters()),  'lr': αq}   # π head only
        ])

    def forward(self, x):
        V, logits = super().forward(x)
        return V, F.softmax(logits, dim=-1)

    def logπ(self, s, a):
        V, π = self(s)
        return V, torch.log(π[range(len(a)), a]), π

    def entropy(self, π):
        return -(π * torch.log(π + 1e-8)).sum(dim=-1).mean()

    def fit(self, s, a, Gt):
        self.train()
        self.optim.zero_grad()
        V, logπ, π = self.logπ(s, a)
        # V     = V.squeeze(-1)
        A     = (Gt - V).detach()
        critic_loss   = 0.5 * F.mse_loss(V, Gt)
        actor_loss    = -(logπ * A).mean()
        entropy_bonus = self.entropy(π)
        loss = actor_loss + critic_loss - 0.01 * entropy_bonus
        loss.backward()
        clip_grad_norm_(self.parameters(), max_norm=1.0) if self.CNN else None
        self.optim.step()
        return loss.item()

    def predict(self, s, state_dim, deterministic=False):
        if not isinstance(s, torch.Tensor):
            s = torch.tensor(s, dtype=torch.float32)
        s_batch = s.ndim > len(state_dim)
        if not s_batch: s = s.unsqueeze(0)
        self.eval()
        with torch.no_grad():
            V, π = self(s)
            a = π.argmax(dim=-1) if deterministic else torch.multinomial(π, 1).squeeze(-1)
            if s_batch: return V, π
            return V[0], π[0]

# ===============================================================================================
class nnACcSharedModel(nnACSharedModel):
    def __init__(self, out_dim, **kw):
        super().__init__(out_dim=out_dim, **kw)
        feat_in     = self.head2.in_features
        self.μ_head = self.head2
        self.σ_head = nn.Linear(feat_in, out_dim, bias=self.final_bias)

    def forward(self, x):
        V, μ = nnSplitModel.forward(self, x)
        σ    = F.softplus(self.σ_head(self._trunk_out)) + 1e-6
        return V, μ, σ

    def logπ(self, s, a):
        V, μ, σ = self(s)
        dist = Normal(μ, σ)
        return V, dist.log_prob(a).sum(dim=-1), dist

    def entropy(self, dist):
        return dist.entropy().mean()

    def predict(self, s, state_dim, deterministic=False):
        if not isinstance(s, torch.Tensor):
            s = torch.tensor(s, dtype=torch.float32)
        s_batch = s.ndim > len(state_dim)
        if not s_batch: s = s.unsqueeze(0)
        self.eval()
        with torch.no_grad():
            V, μ, σ = self(s)
            a = μ if deterministic else Normal(μ, σ).sample()
            if s_batch: return V, a
            return V[0], a[0]
