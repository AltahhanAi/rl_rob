"""
Minimal rl_rob example: tabular Sarsa on a gridworld.

Run:
    python examples/sarsa_gridworld.py

Assumes rl_rob is installed (or you are running from the repo root so that
the `rl` and `env` packages are importable). Adjust the import paths if your
package layout differs.
"""

from rl.tabular import *        # Sarsa, demo configs (demoπ, demoQ, ...)
from env.grid.tabular import *  # grid, maze, windy, cliffwalk, ...
import matplotlib.pyplot as plt


def main():
    # One construct-and-run call. `.interact()` trains the agent and produces
    # the trajectory view and the steps-per-episode curve.
    sarsa = Sarsa(env=grid(), α=.8, episodes=50, seed=10, **demoπ).interact()
    plt.show()
    return sarsa


if __name__ == "__main__":
    main()
