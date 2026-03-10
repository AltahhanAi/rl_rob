
from env.grid.tabular import *


# =====================================================================================================
def dynamics(env=None, seed=0, stoch=False, show=False, repeat=1000):
    
    np.random.seed(seed)                           # change the seed to get different dynamics
    rewards = env.rewards_set()
    nS, nA, nR = env.nS, env.nA, rewards.shape[0]
    p  = np.zeros((nS,nR,  nS,nA))
    randjump = env.randjump
    env.randjump = False                           # so that probabilities of all intermediate jumps are correctly calculated
    for i in trange(repeat if stoch else 1):       # in case the env is stochastic (non-deterministic)
        for s in range(nS):
            if s in env.goals: continue            # uncomment to explicitly make pr of terminal states=0
            for a in range(nA):
                for jump in (range(1,env.jump+1) if randjump else [env.jump]):
                    if not i and show: env.render() # render the first repetition only
                    env.s = s
                    env.jump = jump
                    rn = env.step(a)[1]
                    sn = env.s
                    rn_= np.where(rewards==rn)[0][0] # get reward index we need to update
                    p[sn,rn_, s,a] +=1
                    
    env.randjump = randjump
    # making sure that it is a conditional probability that satisfies Bayes rule
    for s in range(nS):
        for a in range(nA):
            sm=p[:,:, s,a].sum()
            if sm: p[:,:, s,a] /= sm
            
    return p

# ------------------------------------------- P[s'|s,a] ------------------------------------------------------
# P[sn| s,a]: state-transition probability: induced from the dynamics p[sn,rn | s,a]
def ssa(p): 
    # states dim, action dim
    nS, nA = p.shape[0], p.shape[3]
    P = np.zeros((nS, nS, nA))
    for s in range(nS):
        for a in range(nA):
            for sn in range(nS):
                P[sn, s,a] = p[sn,:,s,a].sum()
    return P.transpose(1, 2, 0)          # transforms P[sn,s,a]->P[s,a,sn] for easier tensor manipulation 
                                         # particularly to do P[s,a,sn]@V[sn] instead of (P[sn,s,a].T@V[sn]).T

# ------------------------------------------ R[s,a] -----------------------------------------------------------
# reward function R[s,a]: induced from the dynamics p[sn,rn | s,a]
def rsa(p, rewards):
    # state dim, reward dim
    nS, nA, nR = p.shape[0], p.shape[3], p.shape[1]
    R = np.zeros((nS,nA))
    for s in range(nS):
        for a in range(nA):
             for rn_, rn in enumerate(rewards):        # get the reward rn and its index rn_
                # print(rn_, rn)
                R[s,a] += rn*p[:,rn_, s,a].sum()
    return R
# -------------------------------------------- R[s'|s,a] -------------------------------------------------------
# expected reward given transition: induced from the dynamics
def rssa(p, rewards):
    # state dim, reward dim
    nS, nA, nR = p.shape[0], p.shape[3], p.shape[1]
    R = np.zeros((nS,nS,nA)) 
    for s in range(nS):
        for a in range(nA):
            for sn in range(nS):
                for rn_, rn in enumerate(rewards):        # get the reward rn and its index rn_
                    R[sn, s,a] += rn*p[sn,rn_, s,a]
    return R
