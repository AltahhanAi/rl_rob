from env.gym.base import *

# ========================= Discretisation class that can be used with GymCont ====================================
class Discretise:
    """
    Continuous vector <-> discrete integer id
    """
    def __init__(self, n_bins, clip_ranges, scale, **kw):

        self.n_bins = tuple(b for b in n_bins)
        self.clip_ranges = tuple((lo, hi) for (lo, hi) in clip_ranges)
        self.scale = np.array(scale)

        self._edges = [
            np.linspace((lo / sc), (hi / sc), b + 1)[1:-1]
            for (b, (lo, hi)), sc in zip(zip(self.n_bins, self.clip_ranges), self.scale)
        ]

        self._strides = np.cumprod((1,) + self.n_bins[:-1])

        # representative values in raw space: bin centres
        self._centres = []
        for b, (lo, hi) in zip(self.n_bins, self.clip_ranges):
            edges = np.linspace(lo, hi, b + 1)
            centres = 0.5 * (edges[:-1] + edges[1:])
            self._centres.append(centres)

    def discretise(self, x):
        x = np.array(x, dtype=np.float32).copy()

        for i, (lo, hi) in enumerate(self.clip_ranges):
            x[i] = np.clip(x[i], lo, hi)

        x = x / self.scale
        idx = [int(np.digitize(x[i], self._edges[i])) for i in range(len(self.scale))]
        return int(np.dot(idx, self._strides))

    def undiscretise(self, i):
        """
        integer id -> representative continuous vector (bin centres)
        """
        idx = []
        i = int(i)

        for b, stride in zip(self.n_bins, self._strides):
            idx.append((i // stride) % b)

        x = np.array([self._centres[d][idx[d]] for d in range(len(idx))], dtype=np.float32)
        return x

# =============================== Discretised State Space Gym Env ===========================================

class GymDiscreteS(GymContS, Discretise):
    def __init__(self, env_id, make=gym.make, feature_indices=None, **kw):
        GymContS.__init__(self, env_id=env_id, make=make)
        Discretise.__init__(self, **kw)  # feature_indices already consumed, won't leak
        self.nS = int(np.prod(self.n_bins))
        self.feature_indices = feature_indices

    def _proc_obs(self, obs):
        if self.feature_indices is not None:
            obs = np.asarray(obs)[self.feature_indices]  # safe slice
        return self.discretise(obs)

# =============================== Vectorised Gym Env ===========================================
'''
This class returns a vector for each state observation.
'''
class vGymDiscreteS(GymDiscreteS):

    def __init__(self, env_id, make=gym.make, **kw):
        super().__init__(env_id=env_id, make=make, **kw)
        self.nF = self.nS
        
#     def _proc_obs(self, obs):
#         s = self.discretise(obs)  # scalar index
#         φ = np.zeros(self.nF)
#         φ[s] = 1
#         return φ                  # return feature vector, not scalar

    def _proc_obs(self, obs):
        if self.feature_indices is not None:   # ADD this guard
            obs = np.asarray(obs)[self.feature_indices]
        s = self.discretise(obs)
        φ = np.zeros(self.nF)
        φ[s] = 1
        return φ
    
