'''
    imports
'''

import gymnasium as gym
from gymnasium import spaces
from gymnasium.spaces.utils import flatdim, flatten
from ocatari.core import OCAtari # extract features for pong, useful to avoid having to deal with images
import minigrid  # ensures env registration
import itertools


import numpy as np
import time
import io
import os
import random

from IPython.display import clear_output, display, HTML
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from matplotlib import colors

from numpy.random import rand, seed, randint, choice
from random import choices, sample
from tqdm import trange, tqdm
from numbers import Integral


# ======================================set of useful dictionaries===============================================
'''
CartPole / MountainCar / Acrobot
    continuous state
    discrete actions

Pendulum / HalfCheetah
    continuous state
    continuous actions
'''

# -----------------------------------discrete observation environments--------------------------------------
Taxi = {
    'env_id': 'Taxi-v3',

    # state / action related.........................
    'nS': 500,   # number of states
    'nA': 6,     # south, north, east, west, pickup, dropoff
}

CliffWalking = {
    'env_id': 'CliffWalking-v0',

    # env related..................................
    'nS': 48,   # 4 x 12 grid
    'nA': 4,    # up, right, down, left
}

FrozenLake = {
    'env_id': 'FrozenLake-v1',

    # env related..................................
    'nS': 16,   # 4 x 4 grid
    'nA': 4,    # left, down, right, up
}

# -----------------------------------continuous observation environments--------------------------------------
MountainCar = {
    'env_id': 'MountainCar-v0',

    # discretisation related.........................
    'n_bins': (18, 14),                 # (position, velocity)
    'clip_ranges': ((-1.2, 0.6), (-0.07, 0.07)),
    'scale': (1.2, 0.07),
    
    # tile coding related..........................
    'low': (-1.2, -0.07),
    'high': (0.6, 0.07),
    'n_tiles': (16, 16),
    'n_tilings': 8,
    'hash_size': 4096,
}

CartPole = {
    'env_id': 'CartPole-v1',

    # discretisation related.........................
    'n_bins': (8, 8, 20, 20),
    'clip_ranges': ((-2.4, 2.4), (-3.0, 3.0), (-0.209, 0.209), (-3.5, 3.5)),
    'scale': (2.4, 3.0, 0.209, 3.5),

    # tile coding related..........................
    'low': (-2.4, -3.0, -0.209, -3.5),
    'high': (2.4, 3.0, 0.209, 3.5),
    'n_tiles': (8, 8, 8, 8),
    'n_tilings': 8,
    'hash_size': 4096,
}
Acrobot = {
    'env_id': 'Acrobot-v1',

    # discretisation related.........................
    'n_bins': (8, 8, 8, 8, 10, 10),     # (cos1, sin1, cos2, sin2, dtheta1, dtheta2)
    'clip_ranges': ((-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0), (-12.57, 12.57), (-28.27, 28.27)),
    'scale': (1.0, 1.0, 1.0, 1.0, 12.57, 28.27),

    # tile coding related..........................
    'low': (-1.0, -1.0, -1.0, -1.0, -12.57, -28.27),
    'high': (1.0, 1.0, 1.0, 1.0, 12.57, 28.27),
    'n_tiles': (8, 8, 8, 8, 6, 6),
    'n_tilings': 8,
    'hash_size': 16384,
}

LunarLander = {
    'env_id': 'LunarLander-v3',

    # discretisation related.........................
    'n_bins': (12, 12, 12, 12, 10, 2, 2),  
    # (x, y, vx, vy, angle, left_leg, right_leg)

    'clip_ranges': (
        (-1.5, 1.5),   # x position
        (-0.2, 1.5),   # y height
        (-2.0, 2.0),   # horizontal velocity
        (-2.0, 2.0),   # vertical velocity
        (-1.5, 1.5),   # angle
        (0.0, 1.0),    # left leg contact
        (0.0, 1.0),    # right leg contact
    ),

    'scale': (1.5, 1.5, 2.0, 2.0, 1.5, 1.0, 1.0),

    # tile coding related..........................
    'low': (-1.5, -0.2, -2.0, -2.0, -1.5, 0.0, 0.0),
    'high': (1.5, 1.5, 2.0, 2.0, 1.5, 1.0, 1.0),

    'n_tiles': (8, 8, 8, 8, 6, 2, 2),
    'n_tilings': 8,
    'hash_size': 16384,
}
# -----------------------------------continuous actions and observation environments--------------------------------------
# Pendulum has cont actions, so Sarsa and Q-learning will not work directly 
# unless we discretise the action space, but we can run actor-critic
Pendulum = {
    'env_id': 'Pendulum-v1',

    # discretisation related.........................
    'n_bins': (9, 9, 15),               # (cos, sin, dtheta)
    'clip_ranges': ((-1.0, 1.0), (-1.0, 1.0), (-8.0, 8.0)),
    'scale': (1.0, 1.0, 8.0),
    
    # tile coding related..........................
    'low': (-1.0, -1.0, -8.0),
    'high': (1.0, 1.0, 8.0),
    'n_tiles': (8, 8, 8),
    'n_tilings': 8,
    'hash_size': 4096,
    
    # actor critic related..........................
    'σ': 0.5

}

HalfCheetah = {
    'env_id': 'HalfCheetah-v5',
    
    '''
    IMPORTANT:
    For pedagogical reasons, we use a reduced feature vector instead of the default 17-dimensional observation provided by HalfCheetah.
    
    Selected features:
    (rootz, rooty, rootx_dot, rootz_dot, rooty_dot, bthigh, fthigh, bthigh_dot, fthigh_dot)
    
    This reduced representation makes tabular discretisation or tile coding feasible. 
    Using the full 17-dimensional state would lead to an impractically large state space.
    
    For nonlinear function approximation methods (e.g., neural networks), it is generally preferable to use the full observation vector.
    '''

    # discretisation related.........................
    'n_bins': (8, 12, 16, 12, 12, 10, 10, 10, 10),
    'clip_ranges': (
        (0.2, 1.2),      # rootz          : torso height
        (-1.0, 1.0),     # rooty          : torso pitch
        (-6.0, 6.0),     # rootx_dot      : forward velocity
        (-4.0, 4.0),     # rootz_dot      : vertical velocity
        (-8.0, 8.0),     # rooty_dot      : torso angular velocity
        (-1.5, 1.5),     # bthigh         : back thigh angle
        (-1.5, 1.5),     # fthigh         : front thigh angle
        (-10.0, 10.0),   # bthigh_dot     : back thigh angular velocity
        (-10.0, 10.0),   # fthigh_dot     : front thigh angular velocity
    ),
    'scale': (1.2, 1.0, 6.0, 4.0, 8.0, 1.5, 1.5, 10.0, 10.0),

    # tile coding related..........................
    'low':  (0.2, -1.0, -6.0, -4.0, -8.0, -1.5, -1.5, -10.0, -10.0),
    'high': (1.2,  1.0,  6.0,  4.0,  8.0,  1.5,  1.5,  10.0,  10.0),
    'n_tiles': (8, 8, 10, 8, 8, 6, 6, 6, 6),
    'n_tilings': 8,
    'hash_size': 32768,
}

# ==============================================================================================================
class Gym(gym.Wrapper):
    """
    Base wrapper: constructs env and provides generic reset/step/render.
    Subclasses may override:
      - _proc_obs(obs)
      - _proc_action(a)
      - check_env(env_id)
      - nS meaning
    """
    def __init__(self, env_id, make=gym.make, render_mode="rgb_array",
                 discrete_states=False, remap_actions=False, **kw):
        if isinstance(env_id, str):
            env = make(env_id, render_mode=render_mode)
        else: # if env_id is an actual object from OCAtari or other wrappers
            env = env_id
            spec = getattr(env, "spec", None)
            env_id = spec.id if spec is not None else type(env).__name__
        
        super().__init__(env) 
        self.check_env(env_id)

        self.discrete_states = discrete_states
        self.remap_actions = remap_actions

        self.nA = flatdim(self.action_space)

        # Default: "input dim" of observations (works for Box/Dict/Tuple/Discrete)
        self.nS = flatdim(self.observation_space)

        # If requested, for truly Discrete obs spaces nS means number of states
        if self.discrete_states and isinstance(self.observation_space, spaces.Discrete):
            self.nS = self.observation_space.n

    def check_env(self, env_id):
        pass

    def _proc_obs(self, obs):
        return obs

    def _proc_action(self, a):
        # gym [up, right, down, left] vs yours [left, right, down, up]
        if self.remap_actions:
            return (3 - a) if (a == 0 or a == 3) else a
        return a

    def reset(self, **kw):
        obs, info = super().reset(**kw)
        return self._proc_obs(obs)

    def step(self, a):
        obs, r, terminated, truncated, info = super().step(self._proc_action(a))
        return self._proc_obs(obs), r, terminated, truncated, info

    def render(self, visible=True, pause=0, subplot=131, animate=True, **kw):
        if not visible: return
        self.ax0 = plt.subplot(subplot)
        plt.gcf().set_size_inches(12, 2)
        plt.imshow(super().render())
        plt.axis("off")
        if animate:
            clear_output(wait=True)
            plt.show()
            time.sleep(pause)

#----------------------------------------- now continuous envs -----------------------------------------

class GymCont(Gym):
    """
    Suitable for continuous / structured observation spaces (Box, Dict, Tuple).
    """
    def __init__(self, env_id="CartPole-v1", make=gym.make, render_mode="rgb_array", **kw):
        super().__init__(env_id=env_id, make=make, render_mode=render_mode)

        self.obs_space = self.observation_space

        # For continuous/structured observations, nS should be the flattened dim
        self.nS = flatdim(self.obs_space)

        # For actions, keep base attributes:
        self.act_space = self.action_space

    def check_env(self, env_id):
        common = ["CartPole-v1", "MountainCar-v0", "Acrobot-v1", "Pendulum-v1"]
        if env_id not in common:
            print("note: env not in common list; wrapper still works if observation_space is Box/Dict/Tuple.")
        if isinstance(self.observation_space, spaces.Discrete):
            print("warning: observation_space is Discrete; you may want gymenv_discrete instead.")

    def _proc_obs(self, obs):
        return flatten(self.obs_space, obs)
        
    def _proc_action(self, a):
        # IMPORTANT: continuous-class environments should NOT inherit gridworld action remapping.
        return a