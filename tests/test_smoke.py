"""
Smoke tests for rl_rob: each stage (tabular, linear, neural) builds an agent,
trains it briefly and returns without error.

Run from the repository root:
    pip install -e ".[test]"
    pytest
"""
import matplotlib
matplotlib.use("Agg")  # no display needed

import numpy as np


def test_tabular_sarsa_gridworld():
    from rl.tabular import Sarsa
    from env.grid.tabular import grid
    agent = Sarsa(env=grid(), α=.8, episodes=5, seed=1).interact()
    assert np.isfinite(agent.Q).all()


def test_tabular_qlearning_gridworld():
    from rl.tabular import Qlearn
    from env.grid.tabular import grid
    agent = Qlearn(env=grid(), α=.5, episodes=5, seed=1).interact()
    assert np.isfinite(agent.Q).all()


def test_linear_sarsa_gridworld():
    from rl.linear import vSarsa
    from env.grid.linear import vgrid
    agent = vSarsa(env=vgrid(), α=.1, episodes=5, seed=1).interact()
    assert np.isfinite(agent.W).all()


def test_linear_td_lambda_randomwalk():
    from rl.linear import vTDλ
    from env.grid.linear import vrandwalk
    agent = vTDλ(env=vrandwalk(), α=.05, λ=.8, episodes=5, seed=1).interact()
    assert np.isfinite(agent.w).all()


def test_neural_dqn_maze():
    from rl.neural import DQN
    from env.grid.neural import imaze
    agent = DQN(env=imaze(reward='sparse'), α=1e-4, γ=.98, episodes=2,
                trunk=[(8, 4, 2), (4, 4, 4)], nbuffer=100, nbatch=16,
                rndbatch=True, model_summary=False).interact()
    assert agent is not None
