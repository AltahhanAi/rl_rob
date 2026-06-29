"""
Minimal rl_rob example: linear control on Mountain Car.

Run:
    python examples/tilecoded_mountaincar.py

This uses a discretised state representation. For genuine multi-tiling tile
coding, import from `env.gym.tiled` instead of `env.gym.discretised` and use
the tiled environment wrapper (see worksheet 4.2 for the tilings study).

Assumes rl_rob is installed (or run from the repo root). Adjust import paths
if your package layout differs.
"""

from rl.linear import *           # vSarsa, vQlearn, demo configs (demoGym, ...)
from env.gym.base import *        # gym environment base
from env.gym.discretised import *  # vGymDiscreteS, MountainCar
import matplotlib.pyplot as plt


def main():
    vsarsa = vSarsa(
        env=vGymDiscreteS(**MountainCar),
        α=.1, ε=0, episodes=500, seed=1,
        **demoGym,
    ).interact()
    plt.show()
    return vsarsa


if __name__ == "__main__":
    main()
