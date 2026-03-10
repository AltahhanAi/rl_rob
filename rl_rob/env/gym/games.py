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

