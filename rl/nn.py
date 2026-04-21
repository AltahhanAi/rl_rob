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

# deals with traces addition and multiplication
class Trace(list):
    def __imul__(self, scalar):
        for z in self:
            z.mul_(scalar)
        return self
    
    def __iadd__(self, grads):
        for z, g in zip(self, grads):
            z.add_(g)
        return self
# ================================== NN Infrastructure ==========================================
class nnModel(nn.Module):
    def __init__(self, inp_dim, trunk=[(8, 4, 2), (4, 4, 4)], nF=32, out_dim=3, α=1e-4, τ=1.0, net_str='', optimiser=None,
                 final_bias=True, clipModel=False,  **kw): 
        super().__init__()
        self.layers = nn.ModuleList()
        self.final_bias = final_bias
        self.trunk = trunk
        self.nF = nF
        self.CNN = any(isinstance(h, tuple) and len(h) > 1 for h in trunk)
        self.clipModel = clipModel # clip the weights of the model
        
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
        
        # default optimiser is Adam unless the model is linear, which means we want to test for exact alignment  
        if optimiser is None:
            # self.optimiser = optim.Adam if not self.linear_compatible() else optim.SGD
            self.optimiser = optim.Adam if self.CNN else optim.SGD
            
        else: self.optimiser = optimiser
    
    # useful for testing exact alignment with linear models such as vTD, vQlearn, vActor_Critic, etc.
    def linear_compatible(self):
        return self.trunk==[] and self.nF is None
        
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
    
    def flatten_dim(self, inp_dim):
        with torch.no_grad():
            x = torch.zeros(1, *inp_dim)
            for layer in self.layers[:self.flat_idx]: x = layer(x)
            return x.view(1, -1).shape[1]
        
    def forward(self, x):
        for l, layer in enumerate(self.layers[:-1]):
            x = F.relu(layer(x)) if l != self.flat_idx else layer(x)
        return self.layers[-1](x)    

    def clip_grads(self):
        clip_grad_norm_(self.parameters(), max_norm=1.0) 
        
    def fit(self, vals, targets, exact=True):
        self.train()
        self.optim.zero_grad()
        if exact: loss = .5 * F.mse_loss(vals, targets, reduction='sum') / len(vals)
        else:     loss = .5 * F.mse_loss(vals, targets)
        loss.backward()
        self.clip_grads() if self.clipModel else None
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

    # -------The following functions are useful for eligibility traces update style:------
    
    # gradient of the parameters of the network
    def Δ(self, output):
        self.optim.zero_grad()
        output.sum().backward()
        return [p.grad.clone() for p in self.parameters() if p.grad is not None]
    
    # to update the traces
    def update(self, δ, z):
        with torch.no_grad():
            for z_, p in zip(z, self.parameters()):
                if p.grad is None:
                    p.grad = torch.zeros_like(p)
                p.grad.copy_(-δ.item() * z_)
        self.clip_grads() if self.clipModel else None
        self.optim.step()
        self.optim.zero_grad()
    # --------------------------------------------------------------------------------------
    
    def init_weights(self, head_v0=None, skip_from=None):
        
        gain = init.calculate_gain('relu')
        
        for i, layer in enumerate(self.layers):
            if isinstance(layer, (nn.Linear, nn.Conv2d)):
                is_head = (i == len(self.layers) - 1)
                if skip_from is not None and i >= skip_from:
                    continue
                if head_v0 is not None and is_head:
                    init.constant_(layer.weight, head_v0)
                else:
                    init.xavier_normal_(layer.weight, gain=gain)
                if layer.bias is not None:
                    init.zeros_(layer.bias)
                    
        self.optim = self.optimiser(self.parameters(), lr=self.α)

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
    def __init__(self, head1_dim, head2_dim,  τ=1.0, **kw):
        super().__init__(out_dim=head1_dim, **kw)
        feat_in    = self.layers[-1].in_features
        self.layers = self.layers[:-1]                         # remove final layer
        self.head1  = nn.Linear(feat_in, head1_dim, bias=self.final_bias)
        self.head2  = nn.Linear(feat_in, head2_dim, bias=self.final_bias)
        self.layers.append(self.head1)                         # register for summary
        self.layers.append(self.head2)                         # register for summary
        self.head_idx = len(self.layers) - 2                   # index where heads start

    def init_weights(self, head1_v0=None, head2_q0=None):
        gain_trunk = init.calculate_gain('tanh')    # matches the trunk activation
        gain_head  = init.calculate_gain('linear')  # heads are linear outputs
    
        # --- trunk ---
        for layer in self.layers[:self.head_idx]:
            if isinstance(layer, (nn.Linear, nn.Conv2d)):
                init.xavier_normal_(layer.weight, gain=gain_trunk)
                if layer.bias is not None:
                    init.zeros_(layer.bias)
    
        # --- heads ---
        for head, v0 in [(self.head1, head1_v0), (self.head2, head2_q0)]:
            if v0 is not None:
                init.constant_(head.weight, v0)
            elif head is self.head2 and isinstance(self, nnACSharedModel):
                init.orthogonal_(head.weight, gain=0.01)   # near-uniform initial policy
            else:
                init.xavier_normal_(head.weight, gain=gain_head)
            if head.bias is not None:
                init.zeros_(head.bias)
    
        # --- optimiser ----
        αv = getattr(self, 'αv', None)
        αq = getattr(self, 'αq', None)
        αt = getattr(self, 'αt', αv)   # falls back to αv if αt not set
    
        if αv is not None and αq is not None:
            trunk_params = [p for layer in self.layers[:self.head_idx]
                              for p in layer.parameters()]
            groups = [
                {'params': trunk_params,                  'lr': αt},  # trunk
                {'params': list(self.head1.parameters()), 'lr': αv},  # critic head
                {'params': list(self.head2.parameters()), 'lr': αq},  # actor head
            ]
            self.optim = self.optimiser(groups)
        else: self.optim = self.optimiser(self.parameters(), lr=self.α)
            
                    
    # def init_weights(self, head1_v0=None, head2_q0=None):
    #     super().init_weights(skip_from=self.head_idx)
    #     gain = init.calculate_gain('relu')
    #     for head, v0 in [(self.head1, head1_v0), (self.head2, head2_q0)]:
    #         init.constant_(head.weight, v0) if v0 is not None else init.xavier_normal_(head.weight, gain=gain)
    #         if head.bias is not None: init.zeros_(head.bias)
    #     αv = getattr(self, 'αv', None)
    #     αq = getattr(self, 'αq', None)
    #     αt = getattr(self, 'αt', αv)             # falls back to αv if αt not set
    #     if αv is not None and αq is not None:
    #         trunk_params = [p for layer in self.layers[:self.head_idx] for p in layer.parameters()]
    #         groups = [
    #             {'params': trunk_params,                              'lr': αt},  # trunk
    #             {'params': list(self.head1.parameters()),             'lr': αv},  # critic head
    #             {'params': list(self.head2.parameters()),             'lr': αq},  # actor head
    #         ]
    #         self.optim = self.optimiser(groups)           
            
    def forward(self, x):
        for l, layer in enumerate(self.layers[:self.head_idx]):
            x = F.elu(layer(x)) if l != self.flat_idx else layer(x) 
        self._trunk_out = x
        return self.head1(x), self.head2(x)

    # override to include the trun, head1 and head2 seprately with different max-norms
    def clip_grads(self, trunk_norm=1.0, head1_norm=.5, head2_norm=1.0):
        '''Clip gradients per parameter group (trunk / critic head / actor head)'''
        trunk_params = [p for layer in self.layers[:self.head_idx] for p in layer.parameters()]
        clip_grad_norm_(trunk_params,            max_norm=trunk_norm)  # trunk
        clip_grad_norm_(self.head1.parameters(), max_norm=head1_norm)  # critic head: tighter
        clip_grad_norm_(self.head2.parameters(), max_norm=head2_norm)  # actor head:  full budget

# ===============================================================================================
class nnDuelModel(nnSplitModel):
    def __init__(self, out_dim, **kw):
        super().__init__(head1_dim=1, head2_dim=out_dim, **kw)

    def forward(self, x):
        V, Q = super().forward(x)
        return V + Q - Q.mean(dim=1, keepdim=True)

# ================================= Vanilla Actor-Critic =======================================
class nnACSharedModel(nnSplitModel):
    def __init__(self, out_dim, αv, αq, αt=None, τ=1.0, β_entropy=0.01, **kw):
        super().__init__(head1_dim=1, head2_dim=out_dim, **kw)
        self.β_entropy = β_entropy
        self.αv, self.αq = αv, αq
        self.αt = αt if αt is not None else αv
        self.τ  = τ

    def forward(self, x):
        V, logits = super().forward(x)
        self._scaled_logits = logits / self.τ        # cache for stable log/entropy
        return V, F.softmax(self._scaled_logits, dim=-1)

    def Vπ(self, s):
        V, π = self(s)
        return V.squeeze(-1), π

    def logπ(self, π, a):
        # π kept in signature for backward compatibility; computed from cached logits
        logp = F.log_softmax(self._scaled_logits, dim=-1)
        return logp[range(len(a)), a]

    def entropy(self, π):
        logp = F.log_softmax(self._scaled_logits, dim=-1)
        p    = logp.exp()
        print('entropy', self.β_entropy )
        return -(p * logp).sum(dim=-1)


    
    # def forward(self, x):
    #     V, logits = super().forward(x)
    #     return V, F.softmax(logits / self.τ, dim=-1)

    # def Vπ(self, s):
    #     V, π = self(s)
    #     return V.squeeze(-1), π
    
    # def logπ(self, π, a):
    #     return torch.log(π[range(len(a)), a] + 1e-8)

    # def entropy(self, π):
    #     return -(π * torch.log(π + 1e-8)).sum(dim=-1)

    # ---------- the three update signals (mirror the classical AC algorithm) ----------
    def Δlogπ(self, A, logπ, γt=1.0, **kw):
        """Actor signal — vanilla PG surrogate:  A · log π · γ^t  → maximise."""
        return (logπ * A * γt).mean()

    def ΔV(self, V, Gt, exact=True):
        """Critic signal — .5 · (Gt − V)²  → minimise."""
        return 0.5 * F.mse_loss(V, Gt, reduction='sum') / len(V) if exact else 0.5 * F.mse_loss(V, Gt)

    def ΔH(self, H):
        """Entropy signal — H(π)  → maximise."""
        return H.mean()
    # ----------------------------------------------------------------------------------

    def fit(self, s, a, Gt, A=None, γt=1.0, exact=True, **kw):
        a = a.to(torch.int64) if not torch.is_floating_point(a) else a
        self.train()

        V, π  = self.Vπ(s)
        logπ  = self.logπ(π, a)

        Gt = Gt.squeeze(-1) if Gt.ndim > 1 else Gt
        if A is None: A = (Gt - V).detach()

        L_actor  = self.Δlogπ(A, logπ, γt=γt, **kw)
        L_critic = self.ΔV(V, Gt, exact=exact)
        L_ent    = self.ΔH(self.entropy(π))

        loss = L_critic  -L_actor - self.β_entropy*L_ent

        self.optim.zero_grad()
        loss.backward()
        self.clip_grads() if self.clipModel else None
        self.optim.step()
        return loss.item()

    def predict(self, s, state_dim):
        s = torch.tensor(s, dtype=torch.float32) if not isinstance(s, torch.Tensor) else s
        s_batch = s.ndim > len(state_dim)
        s = s.unsqueeze(0) if not s_batch else s
        self.eval()
        with torch.no_grad():
            V, π = self(s)
            if s_batch: return V, π
            else:       return V.squeeze(0), π.squeeze(0)

# ================================= Continuous (Gaussian) AC ==================================
class nnACcSharedModel(nnACSharedModel):
    def __init__(self, out_dim, σ=1, **kw):
        super().__init__(out_dim=out_dim, **kw)
        self.μ_head = self.head2
        self.σ_head = None # so σ decay in nnPGc does not affect σ_head when it exists
        self.σ = σ         # no σ_head — σ is passed from nnPGc

    def forward(self, x):
        V, μ = nnSplitModel.forward(self, x)
        return V, μ

    def Vπ(self, s):
        V, μ = self(s)
        return V.squeeze(-1), self.π(μ, self.σ)

    def π(self, μ, σ):
        return Normal(μ, σ)
        
    # def logπ(self,  μ, σ, a):
    #     logπ = -((a-μ)**2)/(2*σ**2) - np.log(σ ) - .5*np.log(2*np.pi)
    #     return np.sum(logπ)
        
    def logπ(self, π, a):
        return π.log_prob(a).sum(dim=-1)
        
    def entropy(self, π, a=None):
         # (B,) of zeros, matches discrete shape
        return π.entropy().sum(-1) if self.β_entropy else torch.zeros(π.mean.shape[0]) 
    
    def predict(self, s, state_dim):
        V, μ = super().predict(s, state_dim)
        return V, μ, self.σ

# ================================= PPO — only Δlogπ changes ==================================
def ACEpoch(base):
    class nnACEpochModel_(base):

        def Δlogπ(self, A, logπ, logπ_old=None, ε_clip=None, γt=1.0, **kw):
            """Actor signal — clipped PPO surrogate."""
            r = (logπ - logπ_old).exp()
            return torch.min(r * A * γt, torch.clamp(r, 1 - ε_clip, 1 + ε_clip) * A * γt).mean()

        def fit(self, s, a, A, Gt, logπ_old, epochs, mb_size, ε_clip, γt=1.0):
            a  = a.to(torch.int64) if not torch.is_floating_point(a) else a
            Gt = Gt.detach()
            for _ in range(epochs):
                idx = torch.randperm(len(s))
                for start in range(0, len(s), mb_size):
                    mb = idx[start:start + mb_size]
                    super().fit(s[mb], a[mb], Gt[mb], A=A[mb], γt=γt, exact=False,
                                logπ_old=logπ_old[mb], ε_clip=ε_clip)

    nnACEpochModel_.__name__ = f'nnACEpochModel_{base.__name__}'
    return nnACEpochModel_

# instantiate AFTER both bases are defined
nnACEpochModel   = ACEpoch(nnACSharedModel)     # discrete PPO — softmax
nnACEpochModel_c = ACEpoch(nnACcSharedModel)    # continuous PPO — Gaussian

# ===============================================================================================
