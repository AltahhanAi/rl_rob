from env.gym.tiled import *

class GymTiledDiscAct(GymTiled):
    """
    Tile-coded state + discretised continuous actions.
    """
    def __init__(self, env_id, make=gym.make, action_vals=(-1.0, 0.0, 1.0), **kw):
        GymTiled.__init__(self, env_id=env_id, make=make, **kw)

        assert isinstance(self.action_space, spaces.Box), \
            "GymTiledDiscAct only works for continuous Box action spaces."

        self.act_space = self.action_space
        self.act_shape = self.act_space.shape
        self.act_dim = flatdim(self.act_space)

        # one list of allowed values per action dimension
        vals = np.asarray(action_vals, dtype=np.float32)
        self.disc_actions = np.array(list(itertools.product(vals, repeat=self.act_dim)), dtype=np.float32)

        # clip actions to env bounds
        low = self.act_space.low.reshape(-1)
        high = self.act_space.high.reshape(-1)
        self.disc_actions = np.clip(self.disc_actions, low, high)

        # now actions are discrete indices
        self.nA = len(self.disc_actions)
        self.action_space = spaces.Discrete(self.nA)

    def _proc_action(self, a):
        return self.disc_actions[int(a)].reshape(self.act_shape)