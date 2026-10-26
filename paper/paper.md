---
title: 'rl_rob: a transparent, from-scratch library for teaching reinforcement learning from tabular methods to deep policy gradients'
tags:
  - Python
  - reinforcement learning
  - machine learning education
  - deep learning
  - Jupyter
authors:
  - name: Abdulrahman Altahhan
    orcid: 0000-0003-1133-7744
    affiliation: 1
affiliations:
  - name: School of Computer Science, University of Leeds, United Kingdom
    index: 1
date: 22 October 2026
bibliography: paper.bib
---

# Summary

`rl_rob` is an open-source Python library for teaching reinforcement learning (RL). It implements the full arc of a modern RL course, from tabular temporal-difference prediction to deep actor-critic and proximal policy optimisation (PPO) [@schulman2017ppo]. The tabular and linear stages are written in NumPy [@harris2020numpy] with Matplotlib [@hunter2007matplotlib] for visualisation; the neural stage uses PyTorch [@paszke2019pytorch] only for network layers and gradients. Every algorithm is written from first principles so that its update rule is visible in a few lines of code, and every algorithm is used in the same way: it is constructed with an environment and hyperparameters, and then trained with a single `.interact()` call that also produces learning curves, value maps and inline animations of the agent. The library ships with gridworld, maze, random-walk and cliff-walking environments, wrappers for Gymnasium control tasks [@towers2024gymnasium], and state-representation utilities such as discretisation and hashed tile coding.

# Statement of need

Students of RL face a gap between textbook pseudocode [@sutton2018reinforcement] and the production libraries used in research. Libraries such as Stable-Baselines3 [@raffin2021sb3] and CleanRL [@huang2022cleanrl] are excellent for applying algorithms, but they hide the update rules behind optimisers, automatic differentiation and engineering layers that are not the subject of an introductory course. Educational resources such as Spinning Up [@achiam2018spinningup] focus on deep RL and assume the tabular and linear foundations. In practice, a lecturer either asks students to implement every algorithm and its environment from scratch, which consumes the course in plumbing, or gives them a framework in which the learning itself is invisible.

`rl_rob` fills this gap. It covers the tabular, linear and neural stages of the Sutton and Barto progression within one consistent interface, so that moving from Q-learning to a deep Q-network [@mnih2015dqn] changes one class name rather than the whole program. In the neural stage, PyTorch supplies only the network and its gradients: the temporal-difference target, replay buffer, eligibility traces and policy-gradient updates are written out explicitly, so students can trace a deep RL update from the temporal-difference error to the weight change. This design lets instructors ask students to implement or modify the core update of an algorithm while the library supplies the environment, experiment loop and visualisation. It also supports a form of assessment that remains meaningful when generative AI tools are widely available: students are asked to reason about, alter and explain the update rules, rather than to produce working code alone.

The library is suitable for postgraduate and advanced undergraduate courses in RL, for self-study alongside @sutton2018reinforcement, and for instructors who want a single code base across a whole module.

# Functionality

The library is organised in three stages that mirror a typical course.

- **Tabular methods** (`rl.tabular`): Monte Carlo and TD(0) prediction, n-step methods, Sarsa, Q-learning, expected Sarsa, double Q-learning, Dyna-Q and prioritised sweeping.
- **Linear function approximation** (`rl.linear`): prediction and control with features, n-step Sarsa, eligibility traces including true online TD(λ) [@vanseijen2014trueonline], REINFORCE and actor-critic with softmax and Gaussian policies, applied to Mountain Car, Acrobot, Pendulum and CartPole through discretisation and tile coding.
- **Neural function approximation** (`rl.neural`): neural Sarsa and Q-learning, DQN, double and duelling DQN with experience replay and target networks, neural TD(λ) and Sarsa(λ), REINFORCE, n-step actor-critic and PPO.

All agents share one base class, so hyperparameters, seeding, logging and plotting behave identically across stages. A configuration dictionary passed to `.interact()` switches on the visual mode, which renders the agent moving through the environment inside the notebook as it learns, alongside steps-per-episode and return curves. Utilities for repeated runs and comparisons support the classic experiments of the textbook, such as the error of TD(λ) as a function of step size and λ.

A minimal example trains tabular Sarsa on a gridworld:

```python
from rl.tabular import *
from env.grid.tabular import *

sarsa = Sarsa(env=grid(), α=.8, episodes=50, seed=10).interact()
```

and the same pattern trains a deep Q-network on a sparse-reward maze:

```python
from rl.neural import *

dqn = DQN(env=imaze(reward='sparse'), α=1e-4, γ=.98, episodes=30,
          trunk=[(8, 4, 2), (4, 4, 4)], nbuffer=1000, nbatch=64).interact()
```

The repository includes runnable scripts and notebooks that open directly in Google Colab, and automated tests that train a short run of each stage.

# Teaching experience

`rl_rob` grew out of the Reinforcement Learning module (originally Robotics and Reinforcement Learning) of the online MSc in Artificial Intelligence at the University of Leeds, which I designed and have taught since [ADD: first year]. Over five years it has been used by several hundred postgraduate students, most of them part-time working professionals. A companion set of Jupyter worksheets takes students from one-step TD learning to PPO, with explicit learning outcomes for each worksheet; in each one, students implement or adapt the update rule of an algorithm and then study its behaviour with the library's experiment and plotting tools. To remove installation barriers for part-time learners, the module also runs the library in a cloud laboratory.

Two lessons shaped the design. First, students engage more deeply with an algorithm when the code they write is the update rule itself, and the rest of the experiment is handled for them. Second, a single interface across all three stages lets students compare methods directly, for example tabular and neural Sarsa on the same maze, which makes the effect of function approximation concrete. The library has also served as the starting point for doctoral research on RL in my group.

# References
