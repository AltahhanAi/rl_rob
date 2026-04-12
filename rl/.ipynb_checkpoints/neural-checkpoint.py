'''
Abdulrahman Altahhan (c) 2026
Version: 3.8
Educational code for teaching RL (DQN and related methods).
Permission required for redistribution or research/commercial use.
'''
from rl.linear import *
from env.grid.neural import *
# ===============================================================================================
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
# ================================== NN Infrastructure ==========================================
class nnModel(nn.Module):
    def __init__(self, inp_dim, trunk=[(8, 4, 2), (4, 4, 4)], nF=32, out_dim=3, α=1e-4, net_str='', final_bias=True):
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

    def flatten_dim(self, inp_dim):
        with torch.no_grad():
            x = torch.zeros(1, *inp_dim)
            for layer in self.layers[:self.flat_idx]: x = layer(x)
            return x.view(1, -1).shape[1]

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

    def init_weights(self, final_zero):
        print(f'training afresh so resetting the weights {self.net_str}')
        gain = init.calculate_gain('relu')
        for layer in self.layers:
            if isinstance(layer, (nn.Linear, nn.Conv2d)):
                init.xavier_normal_(layer.weight, gain=gain)
                if layer.bias is not None:
                    init.zeros_(layer.bias)
        if final_zero and isinstance(self.layers[-1], nn.Linear):
            print('setting final layer weights to 0')
            init.zeros_(self.layers[-1].weight)
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
        feat_in = self.layers[-1].in_features
        self.layers = self.layers[:-1]
        self.head1 = nn.Linear(feat_in, head1_dim, bias=self.final_bias)
        self.head2 = nn.Linear(feat_in, head2_dim, bias=self.final_bias)

    def forward(self, x):
        for l, layer in enumerate(self.layers):
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
class nnACModel(nnSplitModel):
    def __init__(self, out_dim, **kw):
        super().__init__(head1_dim=1, head2_dim=out_dim, **kw)

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
        V     = V.squeeze(1)
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
            if s_batch: return V, a
            return V[0], a[0]

# ===============================================================================================
class nnACcModel(nnACModel):
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
        dist = torch.distributions.Normal(μ, σ)
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
            a = μ if deterministic else torch.distributions.Normal(μ, σ).sample()
            if s_batch: return V, a
            return V[0], a[0]

# ================ nnMRP, nnMDP, PG Classes with Neural Net =====================================
class nnMRP(MRP):
    def __init__(self,
                 trunk=[(32, 8, 4), (64, 4, 2), (64, 3, 1)],
                 final_zero=False,
                 final_bias=False,
                 nF=512, rndbatch=True,
                 nbuffer=10000, nbatch=32, endbatch=1,
                 save_weights=1000, load_weights=False,
                 create_vN=True,
                 create_vNn=False, t_Vn=1000,
                 action_dtype=torch.int64,
                 model_class=nnModel,
                 **kw):
        print(f'------------------- 易  {self.__class__.__name__} is being set up 易 ---------------------')
        super().__init__(**kw)
        self.store        = True
        self.create_vN    = create_vN
        self.create_vNn   = create_vNn
        self.t_Vn         = t_Vn
        self.nF           = nF
        self.final_zero   = final_zero
        self.trunk        = trunk
        self.final_bias   = final_bias
        self.action_dtype = action_dtype
        self.model_class  = model_class
        if endbatch > nbatch: endbatch = nbatch - 1
        self.endbatch     = endbatch
        self.nbuffer      = nbuffer
        self.nbatch       = nbatch
        self.rndbatch     = rndbatch
        self.buffer       = deque(maxlen=self.nbuffer)
        self.load_weights_ = load_weights
        self.save_weights_ = save_weights
        self.t_           = 0
        self.vN  = self.create_model('V',  self.α, self.final_bias) if create_vN  else None
        self.vNn = self.create_model('Vn', self.α, self.final_bias) if create_vNn else None

    def init_(self):
        torch.manual_seed(self.seed)
        self.vN.load_weights('V') if self.load_weights_ else self.vN.init_weights(self.final_zero)
        self.vNn.eval() if self.create_vNn else None
        self.V_ = self.V

    def create_model(self, net_str, α, final_bias):
        self.state_dim  = self.env.reset().shape
        self.action_dim = 1 if net_str == 'V' else self.env.nA
        model = self.model_class(
            inp_dim=self.state_dim, trunk=self.trunk,
            nF=self.nF, out_dim=self.action_dim,
            α=α, net_str=net_str, final_bias=final_bias
        )
        model.print_model_summary(net_str)
        return model

    def V(self, s=None):
        result = self.vN.predict(s if s is not None else self.env.S_(), self.state_dim)
        return result.detach().numpy()

    def Vn(self, sn):
        return self.vNn.predict(sn, self.state_dim) if self.create_vNn else None

    def allocate(self):
        self.buffer = deque(maxlen=self.nbuffer)

    def store_(self, s=None, a=None, rn=None, sn=None, an=None, done=None, t=0):
        self.buffer.append((
            torch.tensor(s,    dtype=torch.float32),
            torch.tensor(a,    dtype=self.action_dtype),
            torch.tensor(rn,   dtype=torch.float32),
            torch.tensor(sn,   dtype=torch.float32),
            torch.tensor(done, dtype=torch.bool)
        ))

    def slice_(self, buffer, nbatch):
        return list(islice(buffer, len(buffer) - nbatch, len(buffer)))

    def batch(self):
        endbatch = self.endbatch
        samples  = sample(self.buffer, self.nbatch - endbatch) if self.rndbatch else self.slice_(self.buffer, self.nbatch)
        samples.extend(self.slice_(self.buffer, self.endbatch))
        s, a, rn, sn, dones = zip(*samples)
        s     = torch.stack(s)
        a     = torch.stack(a)
        rn    = torch.stack(rn)
        sn    = torch.stack(sn)
        dones = torch.stack(dones)
        inds  = torch.arange(len(samples))
        return (s, a, rn, sn, dones), inds

# ===============================================================================================
class nnMDP(MDP(nnMRP)):
    def __init__(self, create_vN=False, create_qNn=True, t_Qn=1000, **kw):
        super().__init__(create_vN=create_vN, **kw)
        self.create_qNn = create_qNn
        self.t_Qn       = t_Qn
        self.qN  = self.create_model('Q',  self.α, self.final_bias)
        self.qNn = self.create_model('Qn', self.α, self.final_bias) if create_qNn else None

    def init_(self):
        torch.manual_seed(self.seed)
        if self.create_vN:
            self.vN.load_weights('V') if self.load_weights_ else self.vN.init_weights(self.final_zero)
        self.qN.load_weights('Q') if self.load_weights_ else self.qN.init_weights(self.final_zero)
        self.qNn.eval() if self.create_qNn else None
        self.V_ = self.V
        self.Q_ = self.Q

    def Q(self, s):
        return self.qN.predict(s, self.state_dim)

    def Qn(self, sn):
        return self.qNn.predict(sn, self.state_dim) if self.create_qNn else None

# ===============================================================================================
class nnPG(PG(nnMDP)):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.ϴ = nnACModel(inp_dim=self.state_dim, trunk=self.trunk,
                            nF=self.nF, out_dim=self.env.nA,
                            α=self.α, net_str='ϴ', final_bias=self.final_bias)
        self.policy = self.softmax

    def init_(self):
        torch.manual_seed(self.seed)
        self.ϴ.load_weights('ϴ') if self.load_weights_ else self.ϴ.init_weights(self.final_zero)
        self.V_ = self.V
        self.Q_ = self.Q
        self.H_ = self.H

    def V(self, s=None):
        V, _ = self.ϴ.predict(s if s is not None else self.env.S_(), self.state_dim)
        return V.detach().numpy()

    def H(self, s=None, a=None):
        _, π = self.ϴ.predict(s if s is not None else self.env.S_(), self.state_dim)
        if a is None: return π.detach().numpy()
        return π[a].detach().numpy()

    def softmax(self, s):
        _, π = self.ϴ.predict(s, self.state_dim)
        π = π.detach().numpy()
        return np.random.choice(self.env.nA, p=π)

    def online(self, s, a, rn, sn, an, done, t):
        if len(self.buffer) < self.nbatch: return
        (s, a, rn, sn, dones), inds = self.batch()
        Vn, _ = self.ϴ(sn)
        Vn    = Vn.squeeze(1).detach()
        Vn[dones] = 0
        Gt = rn + self.γ * Vn
        a  = torch.tensor(np.array(a.tolist()), dtype=torch.int64)
        loss = self.ϴ.fit(s, a, Gt)

# ===============================================================================================
class nnPGc(PG(nnMDP)):
    def __init__(self, σ=1, σmin=.01, dσ=1, Tσ=0, **kw):
        super().__init__(action_dtype=torch.float32, **kw)
        self.σ    = σ
        self.σ0   = σ
        self.dσ   = dσ
        self.Tσ   = Tσ
        self.σmin = σmin
        self.ϴ    = nnACcModel(inp_dim=self.state_dim, trunk=self.trunk,
                                nF=self.nF, out_dim=self.env.action_space.shape[0],
                                α=self.α, net_str='ϴ', final_bias=self.final_bias)
        self.policy = self.Gaussian

    def init_(self):
        torch.manual_seed(self.seed)
        self.ϴ.load_weights('ϴ') if self.load_weights_ else self.ϴ.init_weights(self.final_zero)
        self.V_ = self.V
        self.Q_ = self.Q

    def V(self, s=None):
        V, _, _ = self.ϴ.predict(s if s is not None else self.env.S_(), self.state_dim)
        return V.detach().numpy()

    def μ_π(self, s):
        _, μ, _ = self.ϴ.predict(s, self.state_dim)
        return μ.detach().numpy()

    def σ_π(self, s):
        _, _, σ = self.ϴ.predict(s, self.state_dim)
        return σ.detach().numpy()

    def Gaussian(self, s):
        if self.dσ < 1: self.σ = max(self.σmin, self.σ  * self.dσ)
        if self.Tσ > 0: self.σ = max(self.σmin, self.σ0 * (1 - self.t_ / self.Tσ))
        _, μ, σ = self.ϴ.predict(s, self.state_dim)
        μ = μ.detach().numpy()
        σ = σ.detach().numpy()
        a = np.random.normal(μ, σ)
        a = np.clip(a, self.env.action_space.low, self.env.action_space.high)
        return np.atleast_1d(a)

    def π(self, s, a):
        μ, σ = self.μ_π(s), self.σ_π(s)
        return np.prod((1.0 / (np.sqrt(2 * np.pi) * σ)) * np.exp(-((a - μ) ** 2) / (2 * σ**2)))

    def logπ(self, s, a):
        μ, σ = self.μ_π(s), self.σ_π(s)
        return np.sum(-((a - μ)**2) / (2 * σ**2) - np.log(σ) - .5 * np.log(2 * np.pi))

    def online(self, s, a, rn, sn, an, done, t):
        if len(self.buffer) < self.nbatch: return
        (s, a, rn, sn, dones), inds = self.batch()
        Vn, _, _ = self.ϴ(sn)
        Vn = Vn.squeeze(1).detach()
        Vn[dones] = 0
        Gt = rn + self.γ * Vn
        a  = torch.tensor(np.array(a.tolist()), dtype=torch.float32)
        loss = self.ϴ.fit(s, a, Gt)

# ===============================================================================================
class TDN(nnMRP):
    def online(self, *args):
        if len(self.buffer) < self.nbatch: return
        (s, a, rn, sn, dones), inds = self.batch()
        Vs  = self.vN(s)
        Vn  = self.vNn(sn).detach() if self.create_vNn and self.ep > 2 else self.vN(sn).detach()
        Vn[dones] = 0
        targets = self.γ * Vn + rn.unsqueeze(1)
        loss = self.vN.fit(Vs, targets, exact=True)
        if self.t_Vn and self.t_ % self.t_Vn == 0 and self.create_vNn:
            self.vNn.set_weights(self.vN, 'V', self.t_)

# ===============================================================================================
class DQN(nnMDP):
    def online(self, *args):
        if len(self.buffer) < self.nbatch: return
        (s, a, rn, sn, dones), inds = self.batch()
        Qs  = self.qN(s)
        Qn  = self.qNn(sn).detach() if self.create_qNn and self.ep > 2 else self.qN(sn).detach()
        Qn[dones] = 0
        targets = Qs.clone().detach()
        targets[inds, a] = self.γ * Qn.max(1).values + rn
        loss = self.qN.fit(Qs, targets)
        if self.t_Qn and self.t_ % self.t_Qn == 0 and self.create_qNn:
            self.qNn.set_weights(self.qN, 'Q', self.t_)

# ===============================================================================================
class DDQN(DQN):
    def online(self, *args):
        if len(self.buffer) < self.nbatch: return
        (s, a, rn, sn, dones), inds = self.batch()
        Qs  = self.qN(s)
        an  = self.qN(sn).detach().argmax(1)
        Qn  = self.qNn(sn).detach() if self.create_qNn and self.ep > 2 else self.qN(sn).detach()
        Qn[dones] = 0
        targets = Qs.clone().detach()
        targets[inds, a] = self.γ * Qn[inds, an] + rn
        loss = self.qN.fit(Qs, targets, exact=True)
        if self.t_Qn and self.t_ % self.t_Qn == 0 and self.create_qNn:
            self.qNn.set_weights(self.qN, 'Q', self.t_)

# ===============================================================================================
class DuelDQN(DQN):
    def __init__(self, model_class=nnDuelModel, **kw):
        super().__init__(model_class=model_class, **kw)

# ===============================================================================================
def AC(base, label):
    class nnAC_(base):
        def online(self, s, a, rn, sn, an, done, t):
            if len(self.buffer) < self.nbatch: return
            (s, a, rn, sn, dones), inds = self.batch()
            Vn, *_ = self.ϴ(sn)
            Vn = Vn.squeeze(1).detach()
            Vn[dones] = 0
            Gt = rn + self.γ * Vn
            a  = torch.tensor(np.array(a.tolist()), dtype=torch.float32 if label == 'continuous' else torch.int64)
            loss = self.ϴ.fit(s, a, Gt)
    nnAC_.__name__ = f'AC_{label}'
    return nnAC_

# ===============================================================================================
nnAC  = AC(nnPG,  'discrete')
nnACc = AC(nnPGc, 'continuous')