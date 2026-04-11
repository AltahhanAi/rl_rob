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
    def __init__(self, X0=None, Xn=None, n_tiles=8, n_tilings=1, low=None, high=None, nF=None, **kw):
        self.X0        = np.asarray(X0 if X0 is not None else low)             # accept both X0/Xn and low/high
        self.Xn        = np.asarray(Xn if Xn is not None else high)
        self.n_tiles   = np.asarray([n_tiles]*len(self.X0)) if np.isscalar(n_tiles) else np.asarray(n_tiles)
        self.n_tilings = n_tilings
        self.primes    = primes(len(self.X0))                                   # [2,3,5,...] ensures tilings never overlap
        self.shape     = tuple([self.n_tilings] + [self.n_tiles[i] + p for i, p in enumerate(self.primes)])
        nF_exact       = int(np.prod(self.shape))
        self.nF        = nF if nF is not None else nF_exact                     # overridden by subclasses via nF arg
        self._φex      = np.zeros(nF_exact,  dtype=np.float32)                 # exact buffer — always exact size
        self._φ        = self._φex if nF is None else np.zeros(self.nF, dtype=np.float32)  # same as _φex if exact

    def inds(self, x):
        X0, Xn, n_tiles, n_tilings, primes = self.X0, self.Xn, self.n_tiles, self.n_tilings, self.primes[None,:]
        tilings  = np.arange(n_tilings)[:,None]
        s_tiling = floor(n_tilings * n_tiles * (x - X0) / (Xn - X0))
        s_all    = ((s_tiling + primes*tilings) // n_tilings).astype(int)
        return   [(t, *s_all[t]) for t in range(n_tilings)]

    def s_(self, x):
        self._φex[:] = 0                                                        # zero exact buffer
        φ = self._φex.reshape(self.shape)                                       # free view, no copy
        for ind in self.inds(x): φ[ind] = 1
        return self._φex                                                        # already flat


# ================================== Hashed Tile Coder =====================================
class HashedTileCoder(TileCoder):
    def __init__(self, nF=1024, hash_size=None, **kw):
        super().__init__(nF=nF if hash_size is None else hash_size, **kw)       # pass nF up — no repeat alloc

    def s_(self, x):
        self._φ[:] = 0
        self._φ[[hash(ind) % self.nF for ind in self.inds(x)]] = 1
        return self._φ


# ================================ IHT Tile Coder ===============================
class IHTTileCoder(TileCoder):
    def __init__(self, nF=1024, hash_size=None, **kw):
        super().__init__(nF=nF if hash_size is None else hash_size, **kw)       # pass nF up — no repeat alloc

    def s_(self, x):
        self._φ[:] = 0
        inds = np.where(super().s_(x) != 0)[0]                                 # uses _φex safely — no conflict
        self._φ[inds % self.nF] = 1
        return self._φ


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
        if 'Freeway'     in env_id: self._get_obs = self._get_obs_Freeway       # override for stacked RAM features
        if 'HalfCheetah' in env_id: self._get_obs = self._get_obs_HalfCheetah  # override for dim selection

    def _get_obs(self, x):
        if len(x.shape) > 1: x = x[-1]                                         # stacked frames → last frame only
        return x

    def _proc_obs(self, x):                                                     # overrides GymCont._proc_obs
        return self.s_(self._get_obs(x))

    def s_(self, x=None):
        if x is None: x = self.obs                                              # no args → use current obs
        return super().s_(self._get_obs(x))

    def _get_obs_Freeway(self, x):
        # obs is (n_steps, 24) stacked RAM features, each step has 12 (x,y) object pairs
        # returns [agent_y, nearest_car_x, nearest_car_y] normalised to [0,1] → X0=[0,0,0], Xn=[1,1,1]
        pairs = x[-1].reshape(-1, 2)                                            # last step: (12,2) → [x,y] per object
        agent = pairs[0]
        cars  = pairs[1:]
        cars  = cars[cars[:, 0] != -3]                                          # filter undetected cars (sentinel x==-3)
        car   = agent
        if cars.shape[0] > 0:
            idx = np.argmin(np.abs(cars[:, 1] - agent[1]))                      # nearest car by vertical distance
            car = cars[idx]
        W, H  = 160.0, 210.0
        return np.array([agent[1]/H, car[0]/W, car[1]/H])

    def _get_obs_HalfCheetah(self, x):
        # raw obs is 17-dim, select 9 most informative dims
        # X0/Xn must be 9-dim to match
        x = np.asarray(x).reshape(-1)
        return np.array([
            x[0], x[1], x[8], x[9], x[10],                                     # rootz, rooty, rootx_dot, rootz_dot, rooty_dot
            x[2], x[5], x[11], x[14],                                           # bthigh, fthigh, bthigh_dot, fthigh_dot
        ], dtype=np.float32)


# ========================================= Gym Tiled Factory ==========================================
def GymTiledFactory(Coder=TileCoder):
    class GymTiled(GymTiledMixin, Coder, GymCont):
        def __init__(self, env_id, make=gym.make, **kw):
            GymCont.__init__(self, env_id=env_id, make=make)
            Coder.__init__(self, **kw)
            self._init_mixin(env_id, **kw)
            self.nS = self.nF
    return GymTiled

GymTiledExact  = GymTiledFactory(TileCoder)          # exact  — zero collision, nF = n_tilings × ∏(n_tiles+p)
GymTiledIHT    = GymTiledFactory(IHTTileCoder)        # IHT    — fixed nF, best for high-dim envs
GymTiledHashed = GymTiledFactory(HashedTileCoder)     # hashed — fixed nF, rare collisions
GymTiled       = GymTiledHashed                       # backward compat — GymTiled means hashed