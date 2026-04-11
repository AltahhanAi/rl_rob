from env.gym.base import *
from numpy import floor

def primes(n):
    ps, k = [], 2
    while len(ps) < n:
        if all(k % p != 0 for p in ps): ps.append(k)
        k += 1
    return np.array(ps)

# ========================================= Tile Coder Class ==========================================
class TileCoder:
    def __init__(self, X0=None, Xn=None, n_tiles=8, n_tilings=1, low=None, high=None, **kw):
        self.X0        = np.asarray(X0 if X0 is not None else low)     # accept both X0/Xn and low/high
        self.Xn        = np.asarray(Xn if Xn is not None else high)
        self.n_tiles   = np.asarray([n_tiles]*len(self.X0)) if np.isscalar(n_tiles) else np.asarray(n_tiles)  # (d,) per-dim tiles
        self.n_tilings = n_tilings
        self.primes    = primes(len(self.X0))                          # [2,3,5,...] ensures tilings never overlap
        self.shape     = tuple([self.n_tilings] + [self.n_tiles[i] + p for i, p in enumerate(self.primes)])  # (n_tilings, n_tiles[0]+2, n_tiles[1]+3, ...)
        self.nF        = int(np.prod(self.shape))
    
    def inds(self, x):
        X0, Xn, n_tiles, n_tilings, primes = self.X0, self.Xn, self.n_tiles, self.n_tilings, self.primes[None,:]
        tilings  = np.arange(n_tilings)[:,None]
        s_tiling = floor(n_tilings * n_tiles * (x - X0) / (Xn - X0))
        s_all    = ((s_tiling + primes*tilings) // n_tilings).astype(int)            # cast to int for indexing
        return   [(t, *s_all[t]) for t in range(n_tilings)]
        
    def s_(self, x):
        φ = np.zeros(self.shape)
        for ind in self.inds(x): φ[ind] = 1
        return φ.flatten()


# ================================== Hashed Tile Coder =====================================
class HashedTileCoder(TileCoder):
    def __init__(self, nF=1024, hash_size=None, **kw):
        super().__init__(**kw)
        self.nF = nF if hash_size is None else hash_size       # accept both nF and hash_size

    def s_(self, x):
        φ = np.zeros(self.nF)
        φ[[hash(ind) % self.nF for ind in self.inds(x)]] = 1
        return φ


# ================================ IHT Tile Coder ===============================
class IHTTileCoder(TileCoder):
    def __init__(self, nF=1024, hash_size=None, **kw):
        super().__init__(**kw)
        self.nF = nF if hash_size is None else hash_size        # accept both nF and hash_size

    def s_(self, x):
        φ = np.zeros(self.nF)
        inds = np.where(super().s_(x) != 0)[0]
        φ[inds % self.nF] = 1
        return φ


# ========================================= Shared Gym Mixin ==========================================
class GymTiledMixin:
    """
    Obs processing pipeline shared by all GymTiled* classes.
    Handles two cases:
      1. stacked frames (Atari): obs shape (n_steps, obs_dim) → take last frame
      2. dim selection (HalfCheetah): select subset of raw obs dims
    X0/Xn passed to TileCoder must match the processed obs dims, not the raw obs.
    """
    def _init_mixin(self, env_id, **kw):
        if 'σ' in kw: self.σ = kw['σ']
        if 'Freeway'     in env_id: self._get_obs = self._get_obs_Freeway        # override for stacked RAM features
        if 'HalfCheetah' in env_id: self._get_obs = self._get_obs_HalfCheetah   # override for dim selection

    def _get_obs(self, x):
        # default: handle stacked frames by taking the most recent step
        if len(x.shape) > 1: x = x[-1]
        return x

    def _proc_obs(self, x):                                                      # overrides GymCont._proc_obs
        return self.s_(self._get_obs(x))

    def s_(self, x=None):
        if x is None: x = self.obs                                               # no args → use current obs
        return super().s_(self._get_obs(x))

    def _get_obs_Freeway(self, x):
        # obs is (n_steps, 24) stacked RAM features, each step has 12 (x,y) object pairs
        # returns [agent_y, nearest_car_x, nearest_car_y] normalised to [0,1] → X0=[0,0,0], Xn=[1,1,1]
        pairs = x[-1].reshape(-1, 2)                                             # last step: (12,2) → [x,y] per object
        agent = pairs[0]
        cars  = pairs[1:]
        cars  = cars[cars[:, 0] != -3]                                           # filter undetected cars (sentinel x==-3)
        car   = agent
        if cars.shape[0] > 0:
            idx = np.argmin(np.abs(cars[:, 1] - agent[1]))                       # nearest car by vertical distance
            car = cars[idx]
        W, H  = 160.0, 210.0
        return np.array([agent[1]/H, car[0]/W, car[1]/H])

    def _get_obs_HalfCheetah(self, x):
        # raw obs is 17-dim, select 9 most informative dims
        # X0/Xn must be 9-dim to match
        x = np.asarray(x).reshape(-1)
        return np.array([
            x[0], x[1], x[8], x[9], x[10],                                      # rootz, rooty, rootx_dot, rootz_dot, rooty_dot
            x[2], x[5], x[11], x[14],                                            # bthigh, fthigh, bthigh_dot, fthigh_dot
        ], dtype=np.float32)
# ========================================= Gym Tiled Classes ==========================================
class GymTiled(GymTiledMixin, TileCoder, GymCont):
    """exact tile coding — zero collision, nF grows with n_tilings × ∏(n_tiles+p)"""
    def __init__(self, env_id, make=gym.make, **kw):
        GymCont.__init__(self, env_id=env_id, make=make)
        TileCoder.__init__(self, **kw)
        self._init_mixin(env_id, **kw)
        self.nS = self.nF


class GymTiledHashed(GymTiledMixin, HashedTileCoder, GymCont):
    """hashed tile coding — fixed nF, rare collisions, slowest (Python hash loop)"""
    def __init__(self, env_id, make=gym.make, **kw):
        GymCont.__init__(self, env_id=env_id, make=make)
        HashedTileCoder.__init__(self, **kw)
        self._init_mixin(env_id, **kw)
        self.nS = self.nF


class GymTiledIHT(GymTiledMixin, IHTTileCoder, GymCont):
    """IHT tile coding — fixed nF, fully vectorised, best for high-dim envs like HalfCheetah"""
    def __init__(self, env_id, make=gym.make, **kw):
        GymCont.__init__(self, env_id=env_id, make=make)
        IHTTileCoder.__init__(self, **kw)
        self._init_mixin(env_id, **kw)
        self.nS = self.nF


