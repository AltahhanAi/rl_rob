'''
Abdulrahman Altahhan (c) 2026
Version: 3.6

Educational code for teaching RL (DQN and related methods).
Permission required for redistribution or research/commercial use.
'''

'''
    This library implements a nonlinear function approximation for the well-known 
    RL algorithms. It works by inheriting from the classes in the 
    rl.tabular library. We added an nn prefix to the MRP and MDP base classes to 
    differentiate them from their ancestor but we could have kept the same names.
    As usual, we start by defining an MRP class for prediction, then an MDP for control,
    then make other rl algorithms inherit from them as needed.
'''


# from rl.tabular import *
from rl.linear import * # we use this import to ensure that duplicate names such as Qlearn, Sarsa are defined both in tabular.py and in linear.py are imported according to the latest definition from rlln.py
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

# ===============================================================================================
'''
    The nnModel class is just a helper class to create a neural network model 
    that can be used for various tasks, including reinforcement learning. 
    It is built using PyTorch and can handle both convolutional neural networks (CNNs) 
    and multi-layer perceptrons (MLPs).
    The class allows for flexible configuration of the network architecture, including
    the number of layers, the number of filters, and the kernel size for CNNs.
    The model can be used for both feature extraction, Q-learning and actor-critic tasks.
'''

class nnModel(nn.Module):
    def __init__(self, inp_dim, feat_layers=[(16, 5, 2), 32], nF=128, out_dim=3, α=1e-4, net_str='', last_layer_bias=True):
        # register as a subclass of nn.Module and create a list of layers
        super().__init__()
        self.layers = nn.ModuleList()
        self.last_layer_bias = last_layer_bias

        # feat_layer can be a mixture of cnn and fully connected layers
        self.feat_layers = feat_layers
        self.CNN = any(isinstance(h, tuple) and len(h) > 1 for h in feat_layers)

        # feature extractor backbone
        feat_in = inp_dim[0]  # channels if image, feature dim otherwise
        feat_in = self.append_feat_layers(feat_in, inp_dim)

        # Q-learning head, last_layer_bias=False allows us to get exact matching between 1-layer-net and linear class
        self.layers.append(nn.Linear(feat_in, nF)) if nF else None
        self.layers.append(nn.Linear(nF if nF else feat_in, out_dim, bias=self.last_layer_bias))  # Final output layer, no ReLU
        self.α = α
        # done in the reset
        # Initialise the weights and biases of the last fully connected layer (output layer) to 0
        # this is needed in order to give all actions equal chances at least at the start of the agent life!
        # init.zeros_(self.layers[-1].weight)  # Set the weights of the last layer to zero
        # init.zeros_(self.layers[-1].bias)    # Set the bias of the last layer to zero

        self.update_msg = 'update %s network weights...........! at %d'
        self.saving_msg = 'saving %s network weights to disk...!'
        self.loading_msg = 'loading %s network weights from disk...!'
        self.net_str = net_str

    # Append feature extraction layers to the model, either CNN or MLP
    def append_feat_layers(self, feat_in, inp_dim):
        for feat_out in self.feat_layers:
            CNN_layer = isinstance(feat_out, tuple) and len(feat_out) > 1
            (layer, feat_out) = (nn.Conv2d, feat_out) if CNN_layer else (nn.Linear, (feat_out,))
            if feat_out[0]: 
                self.layers.append(layer(feat_in, *feat_out))
                feat_in = feat_out[0]  # Assign feat_in for both CNN and MLP layers
        self.flat_idx = None
        if self.CNN:
            self.flat_idx = len(self.feat_layers)
            self.layers.append(nn.Flatten())
            feat_in = self.flatten_dim(inp_dim)
        return feat_in

    # Apply ReLU activation to all layers except the Flatten and final output layers
    def forward(self, x):
        for l, layer in enumerate(self.layers[:-1]):
            x = F.relu(layer(x)) if l != self.flat_idx else layer(x)
        return self.layers[-1](x)

    # Calculate the flattened dimension size after passing through the CNN layers
    def flatten_dim(self, inp_dim):
        with torch.no_grad():
            x = torch.zeros(1, *inp_dim)  # Create a dummy tensor with the input shape
            for layer in self.layers[:self.flat_idx]: x = layer(x)
            return x.view(1, -1).shape[1]  # Return the flattened dimension size

    # fit a model back, calculating the loss and doing a backprop step
    def fit(self, vals, targets):
        # print('fitting the neural net')
        self.train()
        self.optim.zero_grad()
        loss = .5 * F.mse_loss(vals, targets, reduction='sum')/len(vals)  # to get exact matching between 1-layer-net and linear class
        loss.backward()
        clip_grad_norm_(self.parameters(), max_norm=1.0) if self.CNN else None # only clip for CNN
        self.optim.step()
        return loss.item()

    # gives a single state prediction or a batch prediction and maintains the state dim
    def predict(self, s, state_dim):
        # only convert to tensor when necessary
        if not isinstance(s, torch.Tensor): 
            s = torch.tensor(s, dtype=torch.float32)
        
        # if not a batch bachify
        s_batch = s.ndim > len(state_dim)
        if not s_batch:
            s = s.unsqueeze(0)

        self.eval()
        with torch.no_grad():
            return self(s) if s_batch else self(s)[0]

    def init_weights(self, is_final_layer_zero):
        print(f'training afresh so resetting the weights {self.net_str}')
        gain = init.calculate_gain('relu')
        for layer in self.layers:
            if isinstance(layer, (nn.Linear, nn.Conv2d)):
                init.xavier_normal_(layer.weight, gain=gain) # use init.xavier_normal_, xavier_uniform_,or init.kaiming_uniform_
                if layer.bias is not None:
                    init.zeros_(layer.bias) # we always set the bias to 0
    
        if is_final_layer_zero and isinstance(self.layers[-1], nn.Linear):
            print('setting final layers weights to 0')
            init.zeros_(self.layers[-1].weight)
             # init.zeros_(self.layers[-1].bias)    # already done before 
    
        if self.CNN: self.optim = optim.Adam(self.parameters(), lr=self.α)
        else:        self.optim = optim.SGD(self.parameters(), lr=self.α)

    def load_weights(self, net_str):
        print(self.loading_msg % net_str)
        self.load_state_dict(torch.load(net_str))

    def save_weights(self, net_str):
        print(self.saving_msg % (net_str))
        torch.save(self.state_dict(), f'{net_str}.weights.pt')

    def set_weights(self, source_model, net_str, t):
        # print(self.update_msg % (net_str, t))
        self.load_state_dict(source_model.state_dict())
    
    def print_model_summary(self, net_str):
        print( "╭──────────────────────────────────────────────────────────────────╮")
        print(f"│          Model Architecture: {net_str:<36}│")
        print( "├────┬────────────────────┬─────────────────────────┬──────────────┤")
        print( "│ Id │ Layer              │ Parameters              │ Trainable    │")
        print( "├────┼────────────────────┼─────────────────────────┼──────────────┤")
        total_params = 0
        bias_params = 0
        for i, layer in enumerate(self.layers):
            param_count = sum(p.numel() for p in layer.parameters())
            trainable = any(p.requires_grad for p in layer.parameters())
            total_params += param_count
            layer_bias = sum(p.numel() for name, p in layer.named_parameters() if name == 'bias')
            bias_params += layer_bias
            layer_str = type(layer).__name__
            param_str = f"{param_count:>10,} ({layer_bias:>3,} bias)"
            print(f"│ {i:2d} │ {layer_str:<18} │ {param_str:<23} │ {'Yes' if trainable else 'No ':<12} │")
        print("╰──────────────────────────────────────────────────────────────────╯")
        print(f"Total parameters: {total_params:,} of which {bias_params:,} are bias")

# ===============================================================================================
'''
In neural nets, usually we would want to set the hidden to 0, but not the final, as this will make sets of weights from previous layers identical
'''
class nnMRP(MRP):
    def __init__(self, 
                 feat_layers=[(32, 8, 4), (64, 4, 2), (64, 3, 1)], # (filters/channels, kernel size, stride)
                 is_final_layer_zero=False,                        # useful for setting the default weights of the final layer to 0 
                 nF=512, nbuffer=10000,                            # nF n_feature penultimate layer, nbuffer is the replay buffer size
                 nbatch=32, rndbatch=True, endbatch=1,             # mini batch size, rand batch sampling, non-rand samples at its end
                 save_weights=1000, load_weights=False, create_vN=True, last_layer_bias=False, **kw):

        super().__init__(**kw)
        self.create_vN = create_vN
        self.nF = nF
        self.is_final_layer_zero = is_final_layer_zero
        self.feat_layers = feat_layers
        self.last_layer_bias = last_layer_bias

        if endbatch > nbatch: endbatch=nbatch-1
        self.endbatch = endbatch
        self.nbuffer = nbuffer    # buffer size
        self.nbatch = nbatch      # batch size
        self.rndbatch = rndbatch  # random batch if False, batch is sampled from the end of the buffer
        # endbatch works when rndbatch is True: 
        # sets how many of the latest transitions you want to always add to the end of the sampled batch
        # the count of nbatch includes also endbatch
        
        self.buffer = deque(maxlen=self.nbuffer)

        self.load_weights_ = load_weights
        self.save_weights_ = save_weights

        self.t_ = 0
        if create_vN: self.vN = self.create_model('V', self.α, self.last_layer_bias)

    def init_(self):
        torch.manual_seed(self.seed)
        self.vN.load_weights('V') if self.load_weights_ else self.vN.init_weights(self.is_final_layer_zero)
        self.V_ = self.V

    #--------------------------------------Neural Network model related---------------------------
    ''' create a model for the V or the Q function based on net_str.
        This function creates a customisable neural network model, suitable for tasks like regression 
        or classification. It supports different input dimensions, and accomodate for cnn based 
        architecture, suitable for images, and simpler dense architecture suitable for laser beams.
        It allows for customisation of hidden layers and output units (they can be passed via the constructor).
        You can change the activation functions. The model uses the usual mean squared error loss. 
    '''

    def create_model(self, net_str, α, last_layer_bias):
        self.state_dim = self.env.reset().shape
        self.action_dim = 1 if net_str == 'V' else self.env.nA
        
        # CNN = len(self.inp_dim) == 3  # (C, H, W) → CNN, else MLP
        model = nnModel(
            inp_dim=self.state_dim,
            feat_layers=self.feat_layers,
            nF=self.nF,
            out_dim=self.action_dim,
            α=α,
            net_str=net_str,
            last_layer_bias=last_layer_bias
        )
        # model = model.to(torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'))

        model.print_model_summary(net_str)
        return model

    def V(self, s):
        return self.vN.predict(s, self.state_dim)

    def allocate(self):
        self.buffer = deque(maxlen=self.nbuffer)

    def store_(self, s=None, a=None, rn=None, sn=None, an=None, done=None, t=0):
        self.buffer.append((
            torch.tensor(s,    dtype=torch.float32),
            torch.tensor(a,    dtype=torch.int64),
            torch.tensor(rn,   dtype=torch.float32),
            torch.tensor(sn,   dtype=torch.float32),
            torch.tensor(done, dtype=torch.bool)
        ))

    def slice_(self, buffer, nbatch):
        return list(islice(buffer, len(buffer)-nbatch, len(buffer)))

    def batch(self):
        endbatch = self.endbatch
        samples = sample(self.buffer, self.nbatch-endbatch) if self.rndbatch else self.slice_(self.buffer, self.nbatch)
        samples.extend(self.slice_(self.buffer, self.endbatch)) # always add the latest endbatch items from the buffer, endbatch=0 disable it

        s, a, rn, sn, dones = zip(*samples)
        # Tensors already — just stack
        s = torch.stack(s)
        a = torch.stack(a)
        rn = torch.stack(rn)
        sn = torch.stack(sn)
        dones = torch.stack(dones)
        # inds = torch.arange(self.nbatch)
        inds = torch.arange(len(samples))

        return (s, a, rn, sn, dones), inds

# ===============================================================================================

class nnMDP(MDP(nnMRP)):
    def __init__(self, create_vN=False, create_qNn=True, **kw):
        super().__init__(create_vN=create_vN, **kw)
        self.create_qNn = create_qNn

        self.qN = self.create_model('Q', self.α, self.last_layer_bias)
        self.qNn = self.create_model('Qn', self.α, self.last_layer_bias)  if create_qNn else None # α is not needed target is not trained

    def init_(self):
        torch.manual_seed(self.seed)
        if self.create_vN: # useful for QV-learning
            self.vN.load_weights('V') if self.load_weights_ else self.vN.init_weights(self.is_final_layer_zero)

        self.qN.load_weights('Q') if self.load_weights_ else self.qN.init_weights(self.is_final_layer_zero)
        self.qNn.eval() if self.create_qNn else None

        self.V_ = self.V
        self.Q_ = self.Q

    # This is needed to calculate the Q values for a single state and its policy
    # If we do double Q learning, then we will need it to deal with a batch
    def Q(self, s):
        # if s is None: s = self.env.S_()
        return self.qN.predict(s, self.state_dim)

    # only needed to calculate the targets for a batch of states
    # if we do double Q learning, then we need it to deal with a single state
    def Qn(self, sn):
        return self.qNn.predict(sn, self.state_dim) if self.create_qNn else None


# ===============================================================================================
class DQN(nnMDP):
    def __init__(self, t_Qn=1000, **kw):
        print('------------------- 易  DQN is being set up 易 ---------------------')
        super().__init__(**kw)
        self.store = True
        self.t_Qn = t_Qn

    def online(self, *args):
        if len(self.buffer) < self.nbatch:
            return

        (s, a, rn, sn, dones), inds = self.batch()

        Qs = self.qN(s)
        Qn = self.qNn(sn).detach() if self.create_qNn and self.ep>2 else self.qN(sn).detach() 
        Qn[dones] = 0

        targets = Qs.clone().detach()  #; print('Qs = ', targets.squeeze().numpy().round(3))
        targets[inds, a] = self.γ * Qn.max(1).values + rn # ; print('Qn = ', targets.squeeze().numpy().round(3), ' r = ', rn.numpy() )
        loss = self.qN.fit(Qs, targets)

        if self.t_Qn and self.t_ % self.t_Qn == 0 and self.create_qNn:
            self.qNn.set_weights(self.qN, 'Q', self.t_)

        # print(f'loss = {round(loss,3)}')

# =================================================================================================

# usage example
# nnqlearn = DQN(env=nnenv, \
#                 episodes=300, \
#                 α=1e-4, ε=0.1, γ=.95, 
#                 h1=0, h2=0, nF=32, \
#                 nbuffer=5000, nbatch=32, rndbatch=False,\
#                 self_path='DQN_exp', \
#                 t_Qn=500,
#                 seed=1, **demoGame()).interact(resume=False, save_ep=True)