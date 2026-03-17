from env.gym import *

# ========================================= discretised env setup ==========================================
CartPole = {
    'env_id': 'CartPole-v1',
    'n_bins': (8, 8, 20, 20),
    'clip_ranges': ((-2.4, 2.4), (-3.0, 3.0), (-0.209, 0.209), (-3.5, 3.5)),
    'scale': (2.4, 3.0, 0.209, 3.5),
}

MountainCar = {
    'env_id': 'MountainCar-v0',
    'n_bins': (18, 14),                 # (position, velocity)
    'clip_ranges': ((-1.2, 0.6), (-0.07, 0.07)),
    'scale': (1.2, 0.07),
}

Acrobot = {
    'env_id': 'Acrobot-v1',
    'n_bins': (8, 8, 8, 8, 10, 10),     # (cos1, sin1, cos2, sin2, dtheta1, dtheta2)
    'clip_ranges': ((-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0), (-12.57, 12.57), (-28.27, 28.27)),
    'scale': (1.0, 1.0, 1.0, 1.0, 12.57, 28.27),
}

Pendulum = {
    'env_id': 'Pendulum-v1',
    'n_bins': (9, 9, 15),               # (cos, sin, dtheta)
    'clip_ranges': ((-1.0, 1.0), (-1.0, 1.0), (-8.0, 8.0)),
    'scale': (1.0, 1.0, 8.0),
}

# ========================= Discretisaiton class that can be used with GymCont ====================================

# ------------------------------------------
class Discretise:
    """
    Gym Env: flatten -> clip -> normalise -> discretise -> int state id
    
    - Keeps generality via configurable bins_per_dim and clip_ranges.
    - Assumes normalisation is always on.
    - Overrides nS to be the number of discrete states (product of bins), suitable for tabular Q/SARSA.
    n_bins is the number of bins per dimensions
    """
    def __init__(self, n_bins, clip_ranges, scale):

        self.n_bins = tuple(b for b in n_bins)
        self.clip_ranges = tuple((lo, hi) for (lo, hi) in clip_ranges)

        # Normalisation scales (chosen to match typical CartPole working ranges)
        self.scale = np.array(scale)

        # Bin edges in NORMALISED space. Given the default clip_ranges above, each dim maps ~to [-1, 1].
        self._edges = [
            np.linspace((lo / sc), (hi / sc), b + 1)[1:-1]
            for (b, (lo, hi)), sc in zip(zip(self.n_bins, self.clip_ranges), self.scale)
        ]

        # Map 4D bin index -> single integer id
        self._strides = np.cumprod((1,) + self.n_bins[:-1])
    
    def discretise(self, x):
        # clip in raw space
        for i, (lo, hi) in enumerate(self.clip_ranges):
            x[i] = np.clip(x[i], lo, hi)

        # normalise
        x = x / self.scale

        # bin indices (each 0..bins-1)
        idx = [int(np.digitize(x[i], self._edges[i])) for i in range(len(self.scale))]

        # single integer state id
        return int(np.dot(idx, self._strides))

class GymDiscretised(GymCont, Discretise):

    def __init__(self, env_id, make=gym.make, **kw):
        # force flattening: we want a predictable (4,) vector before discretising
        GymCont.__init__(self, env_id=env_id, make=make)
        Discretise.__init__(self, **kw)
        # Override nS: discrete state count for tabular algorithms
        self.nS = int(np.prod(self.n_bins))
        
    def _proc_obs(self, obs):
        """
        raw obs -> flat float vector -> clip -> normalise -> discretise -> int state id
        """
        return self.discretise(obs)  # shape (4,)
