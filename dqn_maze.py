"""
Minimal rl_rob example: Deep Q-Network on a sparse-reward maze.

Run:
    python examples/dqn_maze.py

Shows the same construct-and-run pattern as the tabular and linear examples,
now with a neural value function, a replay buffer, and a target network.
Swap `DQN` for `DDQN`, `nnSarsa`, or `nnQlearn` to change the algorithm and
keep everything else the same.

Assumes rl_rob is installed (or run from the repo root). Adjust import paths
if your package layout differs.
"""

from rl.linear import *  # shared infrastructure
from rl.neural import *  # DQN, DDQN, nnSarsa, nnQlearn, demo configs (demoGame)
import matplotlib.pyplot as plt


def imaze(**kw):
    # image-grid maze wrapper used in the neural worksheets
    return maze(iGrid, **kw)


def main():
    env = imaze(reward='sparse')
    dqn = DQN(
        env=env, ε=.1, γ=.98, seed=1, episodes=30, q0=0,
        α=1e-4,
        create_Wn=True,                 # build the target network
        trunk=[(8, 4, 2), (4, 4, 4)],   # convolutional trunk
        nF=env.nS,
        nbuffer=1000, nbatch=64,        # replay buffer
        t_Qn=100,                       # target-network sync period
        clipModel=True,
        **demoGame,
    ).interact()
    plt.show()
    return dqn


if __name__ == "__main__":
    main()
