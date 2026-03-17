from env.gym.base import *

# ========================================= Tile Coder Class ==========================================

class TileCoder:
    """
    Sparse tile coding with hashing.
    Returns a list of active feature indices (length = n_tilings).
    """
    def __init__(self, low, high, n_tiles=(8,8,8,8), n_tilings=8, hash_size=4096, seed=0, **kw):
        self.low = np.asarray(low)
        self.high = np.asarray(high)
        self.n_tiles = np.asarray(n_tiles, dtype=np.int32) # tiles per dim
        self.n_tilings = int(n_tilings)                    # how many active features per state
        self.hash_size = int(hash_size)                    # number of features

        assert self.low.shape == self.high.shape
        self.d = self.low.size

        # tile width per dimension
        self.width = (self.high - self.low) / self.n_tiles

        rng = np.random.default_rng(seed)
        # deterministic-ish offsets: evenly spaced fractions + tiny jitter
        base = (np.arange(self.n_tilings) / self.n_tilings)[:, None]
        self.offsets = (base * self.width)[..., :self.d]  # (n_tilings, d)
        self.offsets += (rng.uniform(-1e-3, 1e-3, size=self.offsets.shape)).astype(np.float32)

    def _hash(self, tiling, coords):
        # coords is int array of length d
        h = tiling
        for c in coords:
            h = (h * 1315423911) ^ (int(c) + 0x9e3779b9 + (h << 6) + (h >> 2))
        return int(h % self.hash_size)

    # returns active_features indexes
    def idx(self, obs): 
        # clip to bounds
        obs = np.clip(obs, self.low, self.high)
        idx = [] # features indexes
        for t in range(self.n_tilings):
            coords = (obs + self.offsets[t] - self.low) // self.width
            idx.append(self._hash(t, coords))
        return np.array(idx, dtype=np.uint32)

    def tilecode(self, obs):            
        φ = np.zeros(self.hash_size)      # dense feature vector
        φ[self.idx(obs) ] = 1.0           # multi-hot, self.idx(obs) length n_tilings
        return φ

# ========================================= Gym Env Tile Coded Class ==========================================
class GymTiled(GymCont, TileCoder):
    def __init__(self, env_id, make=gym.make, **kw):
        GymCont.__init__(self, env_id=env_id, make=make)
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