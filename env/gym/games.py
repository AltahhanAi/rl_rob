from env.gym.base import *


Freeway = {
    'env_id': 'ALE/Freeway-v5',
    'low': (0.0, 0.0, 0.0),                 # (agent_y, nearest_car_x, nearest_car_y) in [0,1]
    'high': (1.0, 1.0, 1.0),
    'n_tiles': (24, 24, 24),
    'n_tilings': 8,
    'hash_size': 16384,
}

Breakout = {
    'env_id': 'ALE/Breakout-v5',
    'low': (0.0, 0.0, -1.0, -1.0, 0.0),     # (ball_x, ball_y, ball_vx, ball_vy, paddle_x)
    'high': (1.0, 1.0,  1.0,  1.0, 1.0),
    'n_tiles': (18, 18, 3, 3, 18),
    'n_tilings': 8,
    'hash_size': 16384,
}

SpaceInvaders = {
    'env_id': 'ALE/SpaceInvaders-v5',
    'low': (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),  # (ship_x, ship_y, nearest_invader_x, nearest_invader_y, nearest_bullet_x, nearest_bullet_y)
    'high': (1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
    'n_tiles': (16, 8, 16, 8, 16, 8),
    'n_tilings': 8,
    'hash_size': 32768,
}

Pong = {
    'env_id': 'ALE/Pong-v5',
    'low': (0.0, 0.0, -1.0, -1.0, 0.0, 0.0),    # (ball_x, ball_y, ball_vx, ball_vy, paddle_y_self, paddle_y_opp)
    'high': (1.0, 1.0,  1.0,  1.0, 1.0, 1.0),
    'n_tiles': (16, 16, 3, 3, 16, 16),          # 3 bins is enough for vx/vy in {-1,0,1}
    'n_tilings': 8,
    'hash_size': 16384,
}

# ================================================================================================================
class GymGames(Gym):
    """
    For image-based game environments.
    Keeps image shape, normalises pixels, and converts HWC -> CHW for PyTorch.
    """
    def __init__(self, env_id, make=gym.make, render_mode="rgb_array", **kw):
        super().__init__(env_id=env_id, make=make, render_mode=render_mode, **kw)

        self.obs_space = self.observation_space
        self.act_space = self.action_space

        self.nA = flatdim(self.action_space)
        self.nS = self.reset().shape

    def check_env(self, env_id):
        if not isinstance(self.observation_space, spaces.Box):
            print("warning: GymGames expects Box observation space")

    def _proc_obs(self, obs):
        obs = np.asarray(obs, dtype=np.float32) / 255.0 # normalize
        # convert (H, W, C) -> (C, H, W)
        if obs.ndim == 3 and obs.shape[-1] in (1, 3, 4):
            obs = np.transpose(obs, (2, 0, 1))

        return obs

    def _proc_action(self, a):
        return a