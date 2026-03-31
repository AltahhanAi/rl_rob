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
    'n_tiles': (8, 8),
    'n_tilings': 8,
    'hash_size': 8*8*8,
}



CartPole = {
    'env_id': 'CartPole-v1',
    # 'n_bins': (3, 6, 8, 12),   # 3×3×6×12 = 648 states — still manageable
    'n_bins': (3, 3, 10, 12),    # pos vel angle angvel
    'clip_ranges': ((-2.4, 2.4), (-4.0, 4.0), (-0.209, 0.209), (-4.0, 4.0)),
    'scale': (2.4, 4.0, 0.209, 4.0),
    'low': (-2.4, -4.0, -0.209, -4.0),
    'high': (2.4, 4.0, 0.209, 4.0),
    'n_tiles': (6, 6, 8, 12),  # more tiles for position
    'n_tilings': 8,
    'hash_size': 32768, #4096,
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
    'n_bins': (12, 12, 12, 12, 10, 10, 2, 2),
    # (x, y, vx, vy, angle, angular_velocity, left_leg, right_leg)

    'clip_ranges': (
        (-1.5, 1.5),   # x position
        (-0.2, 1.5),   # y height
        (-2.0, 2.0),   # horizontal velocity
        (-2.0, 2.0),   # vertical velocity
        (-1.5, 1.5),   # angle
        (-3.0, 3.0),   # angular velocity
        (0.0, 1.0),    # left leg contact
        (0.0, 1.0),    # right leg contact
    ),

    'scale': (1.5, 1.5, 2.0, 2.0, 1.5, 3.0, 1.0, 1.0),

    # tile coding related..........................
    'low': (-1.5, -0.2, -2.0, -2.0, -1.5, -3.0, 0.0, 0.0),
    'high': (1.5, 1.5, 2.0, 2.0, 1.5, 3.0, 1.0, 1.0),

    'n_tiles': (6, 6, 6, 6, 6, 6, 2, 2),
    'n_tilings': 8,
    'hash_size': 32768,
}
# -----------------------------------continuous actions and observation environments--------------------------------------

MountainCarContinuous = {
    'env_id': 'MountainCarContinuous-v0',

    # discretisation related.........................
    'n_bins': (18, 14),                 # (position, velocity)
    'clip_ranges': ((-1.2, 0.6), (-0.07, 0.07)),
    'scale': (1.2, 0.07),
    
    # tile coding related..........................
    'low': (-1.2, -0.07),
    'high': (0.6, 0.07),
    'n_tiles': (12, 12),
    'n_tilings': 8,
    'hash_size': 4096,
    
    # actor-critic related..........................
    'σ': 0.3
}

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

LunarLanderContinuous = {
    'env_id': 'LunarLanderContinuous-v3',

    'n_bins': (10, 10, 10, 10, 8, 8, 2, 2),
    # (x, y, vx, vy, angle, angular_velocity, left_leg, right_leg)
    'clip_ranges': (
        (-1.5, 1.5),   # x position
        (-0.2, 1.5),   # y height
        (-2.0, 2.0),   # horizontal velocity
        (-2.0, 2.0),   # vertical velocity
        (-1.5, 1.5),   # angle
        (-2.5, 2.5),   # angular velocity
        (0.0, 1.0),    # left leg contact
        (0.0, 1.0),    # right leg contact
    ),

    'scale': (1.5, 1.5, 2.0, 2.0, 1.5, 2.5, 1.0, 1.0),

    'low': (-1.5, -0.2, -2.0, -2.0, -1.5, -2.5, 0.0, 0.0),
    'high': (1.5, 1.5, 2.0, 2.0, 1.5, 2.5, 1.0, 1.0),

    'n_tiles': (5, 5, 5, 5, 5, 5, 2, 2),
    'n_tilings': 8,
    'hash_size': 32768,

    # actor critic related..........................
    'σ': 0.15
}

    
'''
IMPORTANT:
For pedagogical reasons, we use a reduced feature vector instead of the default 17-dimensional observation provided by HalfCheetah.

Selected features:
(rootz, rooty, rootx_dot, rootz_dot, rooty_dot, bthigh, fthigh, bthigh_dot, fthigh_dot)

This reduced representation makes tabular discretisation or tile coding feasible. 
Using the full 17-dimensional state would lead to an impractically large state space.

For nonlinear function approximation methods (e.g., neural networks), it is generally preferable to use the full observation vector.
'''
HalfCheetah = {
    'env_id': 'HalfCheetah-v5',

    # Raw obs is 17-dim. We select 9 features for tabular feasibility.
    # Full layout: [rootx(0), rootz(1), rooty(2), bthigh(3), bshin(4), bfoot(5),
    #               fthigh(6), fshin(7), ffoot(8), rootx_dot(9), rootz_dot(10),
    #               rooty_dot(11), bthigh_dot(12), bshin_dot(13), bfoot_dot(14),
    #               fthigh_dot(15), fshin_dot(16), ffoot_dot(17)]
    # Excluded: rootx (unbounded), bshin, bfoot, fshin, ffoot (less informative)
    'feature_indices': np.array([1, 2, 3, 6, 9, 10, 11, 12, 15]),
    # selected:                  rootz, rooty, bthigh, fthigh, rootx_dot, rootz_dot, rooty_dot, bthigh_dot, fthigh_dot
  
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

    # actor critic related..........................
    'σ': 0.2
}

# ==============================================================================================================
'''
# A little utility to create a modified copy of any environment dictionary.
# Changes are temporary (per-experiment) by default — the original dict is never mutated.
# To permanently change a config, edit the source dictionary directly and document the change in your notebook.
'''
def envDict(base, **kw):
    return {**base, **kw}

# Examples:
# envDict(CartPole, n_bins=(1, 1, 6, 12))                          # focus bins on pole only
# envDict(CartPole, n_tiles=(6, 6, 8, 12), hash_size=2**15)        # tile coding tweaks
# envDict(MountainCar, n_tilings=16, hash_size=2**15)              # more tilings
# ========================================= useful to quick try and env =======================================
def play(env, steps=5, cont_actions=False):
    obs, obses = env.reset(), []
    for _ in range(steps):
        action = randint(env.nA) if not cont_actions else env.action_space.sample()   # works for LunarLander and Pendulum
        obs, reward, done, _, _ = env.step(action)
        obses.append(obs)                                                         # store rendered frame
        env.render()
        if done: obs = env.reset()
    return np.array(obses)

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
        
        self.figsize0 = (3,2) # for compatibility
    
    def S_(self): 
        pass # for compatibility and graceful failure when we call demoQ
        
    def check_env(self, env_id):
        pass

    def _proc_obs(self, obs):
        return obs
    
    def s_(self):
        return self._proc_obs(self.obs) # for compatibility
        
    def _proc_action(self, a):
        # gym [up, right, down, left] vs yours [left, right, down, up]
        if self.remap_actions:
            return (3 - a) if (a == 0 or a == 3) else a
        return a

    def reset(self, **kw):
        self.obs, info = super().reset(**kw)
        return self._proc_obs(self.obs)

    def step(self, a):
        self.obs, r, terminated, truncated, info = super().step(self._proc_action(a))
        return self._proc_obs(self.obs), r, terminated, truncated, info

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

#======================================= Continuous State Space envs =========================================

class GymContS(Gym):
    """
    Suitable for continuous/structured observation spaces (Box, Dict, Tuple).
    This class flatten the observation and hence is **not suitable for games which uses pixles**
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

GymCont = GymContS