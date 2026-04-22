from env.gym.base import *

# ========================================= Tile Coder Class ==========================================

# class TileCoder:
#     """
#     Sparse tile coding with hashing.
#     Returns a list of active feature indices (length = n_tilings).
#     """
#     def __init__(self, low, high, n_tiles=(8,8,8,8), n_tilings=8, hash_size=4096, seed=0, **kw):
#         self.low = np.asarray(low)
#         self.high = np.asarray(high)
#         self.n_tiles = np.asarray(n_tiles, dtype=np.int32) # tiles per dim
#         self.n_tilings = int(n_tilings)                    # how many active features per state
#         self.hash_size = int(hash_size)                    # number of features

#         assert self.low.shape == self.high.shape
#         self.d = self.low.size

#         # tile width per dimension
#         self.width = (self.high - self.low) / self.n_tiles

#         rng = np.random.default_rng(seed)
#         # deterministic-ish offsets: evenly spaced fractions + tiny jitter
#         base = (np.arange(self.n_tilings) / self.n_tilings)[:, None]
#         self.offsets = (base * self.width)[..., :self.d]  # (n_tilings, d)
#         self.offsets += (rng.uniform(-1e-3, 1e-3, size=self.offsets.shape)).astype(np.float32)

#     def _hash(self, tiling, coords):
#         # coords is int array of length d
#         h = tiling
#         for c in coords:
#             h = (h * 1315423911) ^ (int(c) + 0x9e3779b9 + (h << 6) + (h >> 2))
#         return int(h % self.hash_size)

#     # returns active_features indexes
#     def idx(self, obs): 
#         # clip to bounds
#         obs = np.clip(obs, self.low, self.high)
#         idx = [] # features indexes
#         for t in range(self.n_tilings):
#             coords = (obs + self.offsets[t] - self.low) // self.width
#             idx.append(self._hash(t, coords))
#         return np.array(idx, dtype=np.uint32)

#     def tilecode(self, obs):            
#         φ = np.zeros(self.hash_size)      # dense feature vector
#         φ[self.idx(obs) ] = 1.0           # multi-hot, self.idx(obs) length n_tilings
#         return φ

class TileCoder:
    """
    Sparse tile coding with hashing — vectorised across tilings.
    Returns a list of active feature indices (length = n_tilings).
    """
    def __init__(self, low, high, n_tiles=(8,8,8,8), n_tilings=8, hash_size=4096, seed=0, **kw):
        self.low       = np.asarray(low,     dtype=np.float32)
        self.high      = np.asarray(high,    dtype=np.float32)
        self.n_tiles   = np.asarray(n_tiles, dtype=np.int32)
        self.n_tilings = int(n_tilings)
        self.hash_size = int(hash_size)
        assert self.low.shape == self.high.shape
        self.d = self.low.size

        self.width = (self.high - self.low) / self.n_tiles  # (d,)

        # rng  = np.random.default_rng(seed)
        base = (np.arange(self.n_tilings, dtype=np.float32) / self.n_tilings)[:, None]
        self.offsets = (base * self.width).astype(np.float32)          # (n_tilings, d)
        # self.offsets += rng.uniform(-1e-3, 1e-3, size=self.offsets.shape).astype(np.float32)
        self.offsets += np.random.uniform(-1e-3, 1e-3, size=self.offsets.shape).astype(np.float32)

        # Precompute per-tiling seeds for the vectorised hash (uint32, wraps naturally)
        self._tiling_seeds = np.arange(self.n_tilings, dtype=np.uint32)

    def _hash_all(self, coords_all):
        """
        Vectorised hash across all tilings at once.
        coords_all : (n_tilings, d)  int32 tile coordinates
        returns    : (n_tilings,)    uint32 feature indices
        """
        h = self._tiling_seeds.copy()                   # (n_tilings,)  uint32
        for dim in range(self.d):                       # loop over dims, not tilings
            c = coords_all[:, dim].astype(np.uint32)
            h = np.bitwise_xor(
                    h * np.uint32(1315423911),
                    c + np.uint32(0x9e3779b9) + (h << np.uint32(6)) + (h >> np.uint32(2))
                )
        return h % np.uint32(self.hash_size)

    def idx(self, obs):
        obs = np.clip(obs, self.low, self.high).astype(np.float32)
        # All tilings at once: (n_tilings, d)
        shifted = obs[None, :] + self.offsets - self.low[None, :]
        coords  = (shifted // self.width).astype(np.int32)
        return self._hash_all(coords).astype(np.uint32)

    def tilecode(self, obs):
        phi = np.zeros(self.hash_size, dtype=np.float32)
        phi[self.idx(obs)] = 1.0
        return phi
        
# ========================================= Gym Env Tile Coded Class ==========================================
class GymTiled(GymContS, TileCoder):
    def __init__(self, env_id, make=gym.make, **kw):
        GymContS.__init__(self, env_id=env_id, make=make)
        if 'σ' in kw: self.σ = kw['σ']
        
        TileCoder.__init__(self, **kw)
        

        self.nS = self.nF = self.hash_size   # length of weight vector per action
        if 'Freeway' in env_id:     self._proc_obs_ = self._proc_obs_Freeway
        if 'HalfCheetah' in env_id: self._proc_obs_ = self._proc_obs_HalfCheetah

    def _proc_obs_(self, obs): 
        if len(obs.shape)>1: obs = obs[-1]  # print('warning obs has multiple steps, particularly for Atari')
        return obs
        
    def _proc_obs(self, obs):
        obs = self._proc_obs_(obs)
        return self.tilecode(obs)
    
    def s_(self):
        return self._proc_obs(self.obs) # for compatibility


    # Freeway needs a specific processing
    def _proc_obs_Freeway(self, obs):
        """
        obs: shape (24) from reset()/step()
        returns: (3,) feature vector [ay, nearest_car_x, nearest_car_y] in [0,1] if normalise else pixels
        """
        pairs = obs[-1].reshape(-1, 2)             # last of the 4 stacked steps (12,2) => [x,y] per object
        agent = pairs[0]                           # assume object 0 is the agent/chicken
        cars  = pairs[1:]                          # remaining objects
        
        # keep only detected cars (x != -3)
        cars = cars[cars[:, 0] != -3]
        car = agent # fallback if no cars detected
    
        if cars.shape[0] > 0: 
            # choose car closest in vertical distance to the agent
            idx = np.argmin(np.abs(cars[:, 1] - agent[1]))
            car = cars[idx]
    
        W, H = 160.0, 210.0
        return np.array([agent[1]/ H, car[0]/ W, car[1]/ H])
    
    def _proc_obs_HalfCheetah(self, obs):
        obs = np.asarray(obs).reshape(-1)
        return np.array([
            obs[0],   # rootz
            obs[1],   # rooty
            obs[8],   # rootx_dot
            obs[9],   # rootz_dot
            obs[10],  # rooty_dot
            obs[2],   # bthigh
            obs[5],   # fthigh
            obs[11],  # bthigh_dot
            obs[14],  # fthigh_dot
        ], dtype=np.float32)