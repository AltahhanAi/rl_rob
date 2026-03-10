
'''
    This is a Dynamic Programming Library, mainly 
    1. Policy evaluation for prediction
    2. Policy iteration and value iteration for control
'''
from env.grid.tabular import *
from rl.mdp_model import *

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
def policy_evaluation(env, P=None, R=None, V=None, π=None, γ=.99, θ=1e-3, max_t=np.inf, inplace=False, seed=0, show=False): 
    # env parameters
    nS, nA, nR, rewards = env.nS, env.nA, env.nR, env.rewards_set()
    
    p = dynamics(env, seed) if P is None and R is None else None
    R = rsa(p,rewards)      if R is None else np.array(R) # gives reward function     R[s,a]
    P = ssa(p)              if P is None else np.array(P) # gives transition funciton P[s,a, sn]

    # policy to be evaluated **stochastic/deterministic**
    π = np.zeros(nS,int)  if π  is None else np.array(π); 
    V = np.zeros(nS)      if V  is None else np.array(V); V[env.goals] = 0 # initial state values
    Q = np.zeros((nS,nA)) if inplace else None                             # state action values storage
    
    πdeterm = π.ndim==1    # policy is deterministic or stochastic
    if not πdeterm: π = π/π.sum(1)[:,None]
    S_ = np.arange(nS)
    i = 0
    # policy evaluation --------------------------------------------------------------
    while i < max_t: # when max_t=np.inf, this is effectively equivalent to (while True)
        Δ = 0
        i+= 1
        V_= V.copy()
        if inplace:
            for s in env.S:
                Q[s] = R[s] + γ* P[s]@V                        # P[s,a, sn]@V[sn] sn over S+
                V[s] = Q[s,π[s]] if πdeterm else Q[s]@π[s,:]   # deterministic/stochastic policy
        else:
            Q = R + γ* P@V                                     # P[s,a, sn]@V[sn] sn over S+
            V = Q[S_,π]          if πdeterm else (Q*π).sum(1)  # deterministic/stochastic policy
            V[env.goals] = 0
        
        Δ = np.abs((V-V_)).max()
        if Δ < θ: break
    if show: 
        env.render(underhood='V', V=V)
        print(f'policy evaluation stopped @ iteration {i}')

    return V
# ============================================================================================================================

def policy_iteration_naive(env=randwalk(), p=None, V0=None, π0=None, γ=.99, θ=1e-3, inplace=False, seed=0,
                                    show=False, max_t=np.inf, epochs=np.inf): 
    
    # env parameters
    nS, nA, nR, rewards = env.nS, env.nA, env.nR, env.rewards_set()
    p = dynamics(env,seed) if p is None else np.array(p)

    # policy parameters 
    V = np.zeros(nS)     if V0 is None else np.array(V0); V[env.goals] = 0 # initial state values
    π = np.zeros(nS,int) if π0 is None else np.array(π0); # initial **deterministic** policy 
    Q = np.zeros((nS,nA))  # state action values storage
    
    j=0
    while j < epochs:
        j+=1
        # 1. Policy evaluation---------------------------------------------------
        i=0
        while i < max_t:
            Δ = 0
            i+= 1
            V_ = V if inplace else V.copy()
            for s in range(nS): 
                if s in env.goals: continue # S not S+
                v, V[s] = V[s], 0
                for sn in range(nS): # S+
                    for rn_, rn in enumerate(rewards): # get the reward rn and its index rn_
                        V[s] += p[sn,rn_,  s, π[s]]*(rn + γ*V_[sn])

                Δ = max(Δ, abs(v-V[s]))
            if Δ<θ: print('policy evaluation stopped @ iteration %d:'%i); break
        
        # 2. Policy improvement----------------------------------------------------
        policy_stable=True
        for s in range(nS):
            if s in env.goals: continue # S not S+
            πs = π[s].copy() # we did not need to do the same for V because V[s] is a scalar, not a vector
            for a in range(nA):
                Q[s,a]=0
                for sn in range(nS): # S+
                    for rn_, rn in enumerate(rewards): # get the reward rn and its index rn_
                        Q[s,a] += p[sn,rn_,  s,a]*(rn + γ*V[sn]) 
            
            π[s] = Q[s].argmax() # simple greedy step
            if π[s]!=πs: policy_stable=False
           
        if policy_stable: print(f'policy improvement stopped @ iteration: {j}'); break
        if show: env.render(π=π)
        
    return π
# ---------------------------------------------------------------------------------------------------------------------------

# stochastic and deterministic policy iteration
def policy_iteration(env, V=None, π=None, γ=.99, ε=.1, θ=1e-4, max_t=np.inf, epochs=np.inf, inplace=False, seed=0, show=False):
    
    def sum(π): return π.sum(-1)[:,None]
    # dynamics and env related parameters
    nS, nA, rewards = env.nS, env.nA, env.rewards_set()
    p = dynamics(env,seed)     
    R = rsa(p, rewards)                           # gives reward function     R[s,a]
    P = ssa(p)                                    # gives transition funciton P[s,a, sn]

    # policy to be improved, deterministic or stochastic
    π = np.ones ((nS, nA))  if π is None else np.array(π)    
    πdeterm = π.ndim == 1                         # policy is deterministic or stochastic
    if not πdeterm: π/sum(π)                      # ensures π is a probability
    
    j = 0
    while j < epochs:
        j += 1
        # 1. Policy evaluation---------------------------------------------------
        V = policy_evaluation(env=env, P=P, R=R, V=V, π=π, γ=γ, θ=θ, max_t=max_t, inplace=inplace)

        # 2. Policy improvement----------------------------------------------------
        # update Q based on V
        Q = R + γ * P@V                                           # R[s,a]+γP[s,a, sn]@V[sn] sn over S+

        # improve π based on Q
        π_= π.copy()
        if πdeterm: π = np.argmax(Q,1)                            # deterministic(greedy action)
        else:       π = np.argmaxes(Q); π = ε/nA + (1-ε)*π/sum(π) # stochastic(ε-greedy action breaks ties even) pr[a_maxQ] += (1-ε)/|a_maxQ|
        
        # check for stability
        if np.allclose(π, π_, atol=θ): print(f'policy improvement stopped @ iteration {j}'); break
    
        if show: env.render(π=π)
    return π #, Q, V
# ====================================================================================================================================

def value_iteration_naive(env=randwalk(), p=None, V0=None, γ=.99, θ=1e-4, epochs=np.inf, inplace=False, show=False, seed=0): 

    # env parameters
    nS, nA, nR, rewards = env.nS, env.nA, env.nR, env.rewards_set()
    p = dynamics(env,seed) if p is None else np.array(p)

    # policy parameters
    V = np.zeros(nS) if V0 is None else np.array(V0); V[env.goals] = 0 # initial state values
    Q = np.zeros((nS,nA)) # state action values storage
    
    j = 0
    while j < epochs:
        Δ = 0
        j+= 1
        V_ = V if inplace else V.copy()
        for s in range(nS):
            if s in env.goals: continue
            v, Q[s] = V[s], 0
            for a in range(nA):
                for sn in range(nS):
                    for rn_, rn in enumerate(rewards):             # get the reward rn and its index rn_
                        Q[s,a] += p[sn,rn_,  s,a]*(rn + γ*V_[sn])  # max operation is embedded now in the evaluation
                        
            V[s] = Q[s].max()                                      # step which made the algorithm more concise 
            Δ = max(Δ, abs(v-V[s]))
            
        if Δ < θ: print(f'loop stopped @ iteration: {j}, Δ = {Δ:.3e}'); break 
        if show: env.render(π=Q)
        
    return Q

# ------------------------------------------------------------------------------------------------------------------
# Q estimation, we obtain the policy indirectly from Q
def value_iteration(env=randwalk(), p=None, V=None, γ=.99, θ=1e-4, epochs=np.inf, inplace=False, seed=0, show=False):
    # env parameters
    nS, nA, nR, rewards = env.nS, env.nA, env.nR, env.rewards_set()

    p = dynamics(env, seed) if p is None else np.array(p)
    R = rsa(p,rewards)                   # gives R[s,a]
    P = ssa(p)                           # gives P[s,a, sn] = pr[sn| s,a]
    
    # policy parameters
    V = np.zeros(nS) if V is None else np.array(V); V[env.goals] = 0 # initial state values
    Q = np.zeros((nS,nA))                                            # state action values storage
    j = 0
    while j < epochs:
        j += 1
        Δ = 0
        V_= V.copy()
        if inplace:
            for s in env.S:                 # S not S+ 
                Q[s] = R[s] + γ*P[s]@V      # P[s,a, sn]@V[sn] sn over S+
                V[s] = Q[s].max()           # greedy step
        else:   
            Q = R + γ*P@V                   # R[s,a] + γP[s,a, sn].V[sn] = Q[s,a];
            V = Q.max(1)                    # greedy step
            V[env.goals] = 0                # need to reset V[goals] for the not inplace     
        Δ = abs(V-V_).max()
        if Δ < θ: print(f'loop stopped @ iteration: {j}, Δ = {Δ:.3e}'); break     
        if show: env.render(π=Q)
    
    return Q #, V
# =====================================================================================================
