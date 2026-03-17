
'''
    This is a Dynamic Programming Library, mainly 
    1. Policy evaluation for prediction
    2. Policy iteration and value iteration for control
'''
from env.grid import *

# =====================================================================================================
def argmaxes(Q, atol=1e-4):
    ''' -------**useful for random tie breaking**---------
    returns multiple argmaxes, instead of the first argmax of np.armax()
       1-if Q is 1d array (individual states, returns indexes) 
       2-if Q is 2d array (multiple states, returns a mask)
    '''
    argmaxes = np.isclose(Q, Q.max(axis=-1, keepdims=True), atol=atol, rtol=0.0)
    return np.where(argmaxes)[-1] if Q.ndim==1 else argmaxes*1 
    
def π_argmaxes(Q, atol=1e-4):
    argmaxes = np.isclose(Q, Q.max(axis=-1, keepdims=True), atol=atol, rtol=0.0)
    argmaxes_sum = argmaxes.sum() if Q.ndim==1 else argmaxes.sum(-1)[:,np.newaxis] 
    return argmaxes/argmaxes_sum

np.argmaxes = argmaxes
np.π_argmaxes = π_argmaxes

# =====================================================================================================

def dynamics(env=randwalk(), stoch=False, show=False, repeat=1000): # , maxjump=1

    rewards = env.rewards_set()
    nS, nA, nR = env.nS, env.nA, rewards.shape[0]
    p  = np.zeros((nS,nR,  nS,nA))
    randjump = env.randjump
    env.randjump = False # so that probabilities of all intermediate jumps are correctly calculated
    for i in trange(repeat if stoch else 1): # in case the env is stochastic (non-deterministic)
        for s in range(nS):
            if s in env.goals: continue # uncomment to explicitly make pr of terminal states=0
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


# P[sn| s,a]: state-transition probability: induced from the dynamics p[sn,rn | s,a]
def ssa(p): 
    # states dim, action dim
    nS, nA = p.shape[0], p.shape[3]
    P = np.zeros((nS, nS, nA))
    for s in range(nS):
        for a in range(nA):
            for sn in range(nS):
                P[sn, s,a] = p[sn,:,s,a].sum()
    return P.transpose(1, 2, 0) # transforms P[sn,s,a]->P[s,a,sn] for easier tensor manipulation 
                                # particularly to do P[s,a,sn]@V[sn] instead of (P[sn,s,a].T@V[sn]).T

# reward function R[s,a]: induced from the dynamics p[sn,rn | s,a]
def rsa(p, rewards):
    # state dim, reward dim
    nS, nA, nR = p.shape[0], p.shape[3], p.shape[1]
    R = np.zeros((nS,nA))
    for s in range(nS):
        for a in range(nA):
             for rn_, rn in enumerate(rewards): # get the reward rn and its index rn_
                # print(rn_, rn)
                R[s,a] += rn*p[:,rn_, s,a].sum()
    return R

# expected reward given transition: induced from the dynamics
def rssa(p, rewards):
    # state dim, reward dim
    nS, nA, nR = p.shape[0], p.shape[3], p.shape[1]
    R = np.zeros((nS,nS,nA)) 
    for s in range(nS):
        for a in range(nA):
            for sn in range(nS):
                for rn_, rn in enumerate(rewards): # get the reward rn and its index rn_
                    R[sn, s,a] += rn*p[sn,rn_, s,a]
    return R

# =====================================================================================================
# stochastic and deterministic policy evaluation
def policy_evaluation(env, p=None, V0=None, π=None, γ=.99, θ=1e-3, max_t=np.inf, show=False):
    # env parameters
    nS, nA, nR, rewards = env.nS, env.nA, env.nR, env.rewards_set()

    p = dynamics(env) if p is None else np.array(p)
    R = rsa(p,rewards)
    P = ssa(p)
    
    # # policy to be evaluated **stochastic/deterministic**
    π = np.zeros(nS,int) if π  is None else np.array(π); π = π if π.ndim==1 else π/π.sum(1)[:,None]
    V = np.zeros(nS)     if V0 is None else np.array(V0); V[env.goals] = 0 # initial state values

    i = 0
    # policy evaluation --------------------------------------------------------------
    while i < max_t: # when max_t=np.inf, this is effectively equivalent to (while True)
        Δ = 0
        i+= 1
        V_ = V.copy()
        S_ = np.arange(nS)
        
        # P[s,a, sn]@V[sn] sn over S+
        if π.ndim == 1: V = R[S_,π] + γ*P[S_,π]@V               # deterministic policy
        else:           V =((R      + γ*P      @V)*π).sum(1)    # stochastic policy
            
        V[env.goals] = 0
        Δ = np.abs((V-V_)).max()
        if Δ < θ: break
        
    if show: 
        env.render(underhood='V', V=V)
        print('policy evaluation stopped @ iteration %d'%i)

    return V
# =====================================================================================================
# stochastic and deterministic policy iteration
def policy_iteration(env=randwalk(), p=None, V0=None, π0=None, 
                           γ=.99, ε=.1, θ=1e-4, show=False, max_t=np.inf, epochs=np.inf):
    # env parameters
    nS, nA, nR, rewards = env.nS, env.nA, env.nR, env.rewards_set()
    
    # dynamics related
    p = dynamics(env) if p is None else np.array(p)
    R = rsa(p,rewards) # gives reward function     R[s,a]
    P = ssa(p)         # gives transition funciton P[s,a, sn]

    # policy to be improved **stochastic**
    π = np.ones ((nS, nA))  if π0 is None else np.array(π0); π = π if π.ndim==1 else π/π.sum(1)[:,None]
    V = np.zeros(nS)        if V0 is None else np.array(V0); V[env.goals] = 0 # initial state values
    Q = np.zeros((nS, nA))      

    j=0
    while j < epochs:
        j+=1
    # 1. Policy evaluation---------------------------------------------------
        i=0
        while i < max_t:
            i += 1
            V_ = V.copy()
            S_ = np.arange(nS)
            # P[s,a, sn]@V[sn] sn over S+
            if π.ndim == 1: V = (R[S_,π] + γ*P[S_,π]@V)               # deterministic policy
            else:           V = ((R      + γ*P      @V)*π).sum(1)     # stochastic policy
            V[env.goals] = 0
            Δ = np.abs((V-V_)).max()
            if Δ < θ: print('policy evaluation stopped @ iteration: %d'%i); break

    # 2. Policy improvement----------------------------------------------------
        # update Q based on V
        Q = R + γ * P@V                                # R[s,a]+γP[s,a, sn]@V[sn] sn over S+

        # update π based on Q
        π_= π.copy()
        if π.ndim==1: π = Q.argmax(1)                  # deterministic improvement (greedy action per state)
        else:         π = ε/nA + (1-ε)*π_argmaxes(Q,θ) # stochastic    improvement (ε-greedy over argmax ties)
                                                       # pr[argmax a] += (1-ε)/|argmax a| greedy step across S, even ties breaking
        # check for stability
        π_stable = np.allclose(π, π_, atol=θ)
        if π_stable: print('policy improvement stopped @ iteration %d:'%j); break
    
        if show: env.render(π=π)
    return π, Q, V
    
# =====================================================================================================
    
def value_iteration(env, p=None, V0=None, γ=.99, θ=1e-4, epochs=np.inf, show=False): 
    # env parameters
    nS, nA, nR, rewards = env.nS, env.nA, env.nR, env.rewards_set()

    p = dynamics(env) if p is None else np.array(p)
    R = rsa(p,rewards)     # gives expected rewards R[s,a]
    P = ssa(p)             # gives transition probabilities P[s,a, sn]= P[sn| s,a]

    # policy parameters
    V = np.zeros(nS) if V0 is None else np.array(V0); V[env.goals] = 0 # initial state values
    Q = np.zeros((nS,nA))  # state action values storage
    
    j = 0
    while j < epochs:
        j += 1
        V_= V.copy()        
        Q = R + γ* P@V      # R[s,a] + γP[s,a, sn].V[sn] = Q[s,a]
                             
        V = Q.max(1)        # step which made the algorithm more concise
        V[env.goals] = 0    # since the pev step assigned V[goals]=Q[goals].max
        
        Δ = abs(V-V_).max()    
        if Δ < θ: print(f'loop stopped @ iteration: {j}, Δ = {Δ:.3e}'); break

        if show: env.render(π=Q)        
    return Q
# =====================================================================================================
