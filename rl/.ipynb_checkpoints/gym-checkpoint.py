import gymnasium as gym
from gymnasium import spaces
from gymnasium.spaces.utils import flatdim, flatten
import numpy as np

import minigrid  # ensures env registration

class Gym(gym.Wrapper):
    """
    Base wrapper: constructs env and provides generic reset/step/render.
    Subclasses may override:
      - _proc_obs(obs)
      - _proc_action(a)
      - check_env(env_id)
      - nS meaning
    """
    def __init__(self, env_id, make=gym.make, render_mode="rgb_array"):
        super().__init__(make(env_id, render_mode=render_mode))
        self.check_env(env_id)

        self.nA = flatdim(self.action_space)

        # Default: "input dim" of observations (works for Box/Dict/Tuple/Discrete)
        self.nS = flatdim(self.observation_space)

    def check_env(self, env_id):
        pass

    def _proc_obs(self, obs):
        return obs

    def _proc_action(self, a):
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
            
#----------------------------------------- now discrete envs -----------------------------------------
class GymDisc(Gym):
    """
    Suitable for discrete-observation envs: CliffWalking-v0, Taxi-v3, FrozenLake-v1, MiniGrid, etc.
    """
    def __init__(self, env_id="CliffWalking-v0", make=gym.make, render_mode="rgb_array"):
        super().__init__(env_id=env_id, make=make, render_mode=render_mode)

        # For truly Discrete obs spaces, nS is number of states
        if isinstance(self.observation_space, spaces.Discrete):
            self.nS = self.observation_space.n
        else:
            # fall back to flatdim for structured spaces
            self.nS = flatdim(self.observation_space)

    def check_env(self, env_id):
        if env_id not in ["CliffWalking-v0", "Taxi-v3", "FrozenLake-v1"]:
            print("warning: this class might not work appropriately for the intended environment")

    def _proc_action(self, a):
        # gym [up, right, down, left] vs yours [left, right, down, up]
        return (3 - a) if (a == 0 or a == 3) else a

#----------------------------------------- now continuous envs -----------------------------------------

class GymCont(Gym):
    """
    Suitable for continuous / structured observation spaces (Box, Dict, Tuple).
    """
    def __init__(self, env_id="CartPole-v1", make=gym.make, render_mode="rgb_array"):
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