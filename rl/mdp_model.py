
from env.grid.tabular import *

# ================================ dynamics p(s', r| s, a)=p[sn,rn,s,a] ==================================
# p[sn, rn, s, a] = P(S_{t+1}=sn, R_{t+1}=reward_values[rn] | S_t=s, A_t=a)

def dynamics(env=None, seed=0, stoch=False, show=False, repeat=1000):
    
    np.random.seed(seed)
    rewards = env.rewards_set()
    nS, nA, nR = env.nS, env.nA, rewards.shape[0]
    # p[sn, rn | s, a]
    p = np.zeros((nS, nR, nS, nA))

    reward_to_idx = {r: i for i, r in enumerate(rewards)}

    randjump = env.randjump
    old_jump = env.jump
    env.randjump = False

    jumps = range(1, old_jump + 1) if randjump else [old_jump]

    for i in trange(repeat if stoch else 1):
        for s in range(nS):
            if s in env.goals:
                continue
            for a in range(nA):
                for jump in jumps:
                    if not i and show:
                        env.render()
                    env.s = s
                    env.jump = jump
                    rn = env.step(a)[1]
                    sn = env.s
                    rn_idx = reward_to_idx[rn]
                    p[sn, rn_idx, s, a] += 1

    env.randjump = randjump
    env.jump = old_jump
                
    sums = p.sum(axis=(0, 1))   # p[:, :, s, a].sum() for all s, a
    for s in range(nS):
        for a in range(nA):
            if sums[s, a] != 0:
                p[:, :, s, a] /= sums[s, a]
    return p # p[s', r | s, a]

# =========================================================================================================================
'''
Important: although we defined the full dynamics as p(sn,r, s,a) for consistency with the textbook definition of p(s',r|s,a).
However the transition p(s'|s,a) and expected transition rewards(normalised and unnormalised) all use the shape (s,a,sn) for
easy for computation in dynamic programming, particularly so that we can do Q = R + γ* P@V without having to transpose inside a loop.

'''
# ========================= p(s'|s,a) = p[s,a,sn] ============================
# from p[sn, rn, s, a] compute p(sn | s, a), returns shape (s, a, sn) for computational reasons

def ssa(p):
    # sum over reward axis rn
    return np.transpose(p.sum(axis=1), (1, 2, 0)) # transpose -> (s, a, sn) 

# ================================ r(s,a) =====================================
# expected immediate reward:
# r[s,a] = E[r | s,a] = Σ_sn ( Σ_r (r * p[sn,r|s,a]))

def rsa(p, rewards):
    # rewards[None, :, None, None] has shape (1, nR, 1, 1) so it broadcasts over p[sn, rn, s, a]
    return (p * rewards[None, :, None, None]).sum(axis=(0, 1))

# ================================ r'(s,a,s') ==================================
# expected transition reward (unnormalised)
# r'[s,a,sn] = E[r | s,a,sn]* p(sn|s,a) = Σ_r (r * p[sn,r|s,a])

def rpssa(p, rewards):
    # we use rewards[None, :, None, None] so that broadacsting of * can work
    r = (p * rewards[None, :, None, None]).sum(axis=1)
    # return shape (s,a,sn)
    return np.transpose(r, (1, 2, 0))
    
# ================================ r[s,a,s'] ==================================
# expected transition reward (normalised): proper expectation
# r[s,a,sn] = E[r | s,a,sn] = Σ_r (r * p[sn,r|s,a]) / p(sn|s,a)

def rssa(p, rewards):

    # r'[sn,s,a] = sum_r reward[r] * p[sn,r,s,a]
    r = rpssa(p, rewards)
    
    # p(sn|s,a)
    p = ssa(p)

    # normalisation step, with explicit safe division
    not0 = p!=0
    r[not0] /= p[not0]

    # return shape (s,a,sn)
    return r
