'''
    this library implement a *linear function approximation* for well-known 
    RL algorithms. It works by inheriting from the classes in the 
    rl.base library. We added a v prefix to the MRP and MDP base classes to 
    differentiate them from their ancestor but we could have kept the same names.
    We start by defining an MRP class for prediction, then MDP for control,
    then make other rl algorithms inherit from them as needed.
'''

from rl.tabular import *
from env.grid.linear import *
from env.gym.tiled import *
from env.gym.discretised import *

from math import floor
# ======================================= prediction master class==========================================
class vMRP(MRP):
    # set up the weights, must be done whenever we train
    def init_(self):
        self.V_ = self.V
        self.w = np.ones(self.env.nF)*self.v0
    
    #-------------------------------------------buffer related-------------------------------------
    # allocate a suitable buffer
    def allocate(self): 
        super().allocate()
        self.s = np.ones ((self.max_t, self.env.nF), dtype=np.uint32) *(self.env.nS+10)    
    
    #---------------------------------------- retrieve Vs ------------------------------------------
    def V(self, s=None):
        return self.w.dot(s) if s is not None else self.w.dot(self.env.S_()) 
        
    def ΔV(self,s): # gradient: we should have used ∇ , but Jupyter does not like it
        return s

# ======================================= prediction algorithms==========================================
class vMC(vMRP):
    def init(self):
        self.store = True 
        
    # ----------------------------- 🌘 offline, MC learning: end-of-episode learning ----------------------    
    def offline(self):
        # obtain the return for the latest episode
        Gt = 0
        for t in range(self.t, -1, -1):
            s = self.s[t]
            rn = self.r[t+1]
            
            Gt = self.γ*Gt + rn
            self.w += self.α*(Gt - self.V(s))*self.ΔV(s)

class vTDf(vMRP):
    def init(self):
        self.store = True
        
    # ----------------------------- 🌘 offline TD learning ----------------------------   
    def offline(self):
        for t in range(self.t, -1, -1):
            s = self.s[t]
            sn = self.s[t+1]
            rn = self.r[t+1]
            done = self.done[t+1]
            
            self.w += self.α*(rn + (1-done)*self.γ*self.V(sn) - self.V(s))*self.ΔV(s)

class vTD(vMRP):
    # ----------------------------- 🌖 online learning ----------------------    
    def online(self, s, rn,sn, done, *args): 
        self.w += self.α*(rn + (1-done)*self.γ*self.V(sn) - self.V(s))*self.ΔV(s)

class vTDn(vMRP):
    def init(self):
        self.store = True # there is a way to save storage by using t%(self.n+1) but we left it for clarity

    # ----------------------------- 🌖 online learning ----------------------    
    def online(self,*args):
        τ = self.t - (self.n-1);  n=self.n
        if τ<0: return
        
        # we take the min so that we do not exceed the episode limit (last step+1)
        τn = τ+n ; τn=min(τn, self.t+1 - self.skipstep)
        τ1 = τ+1
        
        sτ = self.s[τ ]
        sn = self.s[τn]
        done = self.done[τn]
        
        # n steps τ+1,..., τ+n inclusive of both ends
        self.w += self.α*(self.G(τ1,τn)+ (1-done)*self.γ**n *self.V(sn) - self.V(sτ))*self.ΔV(sτ)
        
class vTDnf(vMRP):
    def init(self):
        self.store = True # offline method we need to store anyway

    # ----------------------------- 🌘 offline TD learning ----------------------------   
    def offline(self):
        n = self.n        
        for t in range(self.t+n): # T+n to reach T+n-1
            τ  = t - (n-1)
            if τ<0: continue
        
            # we take the min so that we do not exceed the episode limit (last step+1)
            τ1 = τ+1
            τn = τ+n ; τn=min(τn, self.t+1)
            
            sτ = self.s[τ ]
            sn = self.s[τn]
            done = self.done[τn]
            
            # n steps τ+1,..., τ+n inclusive of both ends
            self.w += self.α*(self.G(τ1,τn)+ (1-done)*self.γ**n *self.V(sn) - self.V(sτ))*self.ΔV(sτ)

# ======================================= control master class==========================================

class vMDP(MDP(vMRP)):
    def init_(self):
        self.w = np.ones(self.env.nF)*self.v0
        self.W = np.ones((self.env.nA, self.env.nF))*self.q0
        
        self.V_ = self.V
        self.Q_ = self.Q

    def Q(self, s=None, a=None):
        W = self.W if a is None else self.W[a]
        return W@s if s is not None else np.matmul(W, self.env.S_()).T 

    # we should have used ∇ , but Python does not like it
    def ΔQ (self, s): 
        return s

# ======================================= control algorithms===================================
class vMCC(vMDP):
    def init(self):
        self.store = True
    # ---------------------------- 🌘 offline, MC learning: end-of-episode learning-------------    
    def offline(self):  
        # obtain the return for the latest episode
        Gt = 0
        for t in range(self.t, -1, -1):
            s = self.s[t]
            a = self.a[t]
            rn = self.r[t+1]
            
            Gt = self.γ*Gt + rn
            self.W[a] += self.α*(Gt - self.Q(s,a))*self.ΔQ(s)

# -------------------------------------🌖 Sarsa online learning ----------------------------------
class vSarsa(vMDP):
    def init(self): #α=.8
        self.step = self.step_an # for Sarsa we want to decide the next action in time step t

    def online(self, s, rn,sn, done, a,an):
        self.W[a] += self.α*(rn + (1-done)*self.γ*self.Q(sn,an) - self.Q(s,a))*self.ΔQ(s)
 
#--------------------------------------🌖 Q-learning online --------------------------------------
class vQlearn(vMDP):
    def online(self, s, rn,sn, done, a,_):
        self.W[a] += self.α*(rn + (1-done)*self.γ*self.Q(sn).max() - self.Q(s,a))*self.ΔQ(s)
    
# --------------------- 🌖 XSarsa (value function) online learning ------------------------------------
class vXSarsa(vMDP):
    def online(self, s, rn,sn, done, a,_):      
        # obtain the ε-greedy policy probabilities, then obtain the expecation via a dot product for efficiency
        π = self.π(sn)
        v = self.Q(sn).dot(π)
        self.W[a] += self.α*(rn + (1-done)*self.γ*v - self.Q(s,a))*self.ΔQ(s)

# ------------------------ 🌖 multi-step (value function) online learning -----------------------------      
class vSarsan(vMDP):
    def init(self):
        self.store = True        # although online but we need to access *some* of earlier steps,
        self.step = self.step_an # for Sarsa we want to decide the next action in time step t
      
    def online(self, *args):
        τ = self.t - (self.n-1);  n=self.n
        if τ<0: return
        
        # we take the min so that we do not exceed the episode limit (last step+1)
        τ1 = τ+1
        τn = τ+n ; τn=min(τn, self.t+1 - self.skipstep)
        
        sτ = self.s[τ];  aτ = self.a[τ]
        sn = self.s[τn]; an = self.a[τn]
        done = self.done[τn]
        
        # n steps τ+1,..., τ+n inclusive of both ends
        self.W[aτ] += self.α*(self.G(τ1,τn) + (1-done)*self.γ**n *self.Q(sn,an) - self.Q(sτ,aτ))*self.ΔQ(sτ)

# ========================
# ------------------------ 🌖 multi-step (value function prediction) online learning -----------------------   
class vTDλ(vMRP):
    def __init__(self, λ=.5, **kw):
        super().__init__(**kw)
        self.λ = λ
    def step0(self):
        self.z = self.w*0
    
    def online(self, s, rn,sn, done, *args): 
        α, γ, λ = self.α, self.γ, self.λ
        self.z = λ*γ*self.z + self.ΔV(s)
        self.w += α*(rn + (1-done)*γ*self.V(sn) - self.V(s))*self.z
    
# ------------------------ 🌖 multi-step (value function prediction) online learning -----------------------
class vtrueTDλ(vMRP):
    def __init__(self, λ=.5, **kw):
        super().__init__(**kw)
        self.λ = λ

    def step0(self):
        self.z = self.w*0
        self.vo = 0
  
    def online(self, s, rn,sn, done, *args): 
        α, γ, λ = self.α, self.γ, self.λ
        
        self.v = self.V(s)
        self.vn= self.V(sn)*(1-done)
        δ = rn + γ*self.vn - self.v
        self.z = λ*γ*self.z + (1-α*λ*γ*self.z.dot(s))*s
        
        self.w += α*(δ + self.v - self.vo )*self.z - α*(self.v - self.vo)*s
        self.vo = self.vn
    
# ------------------------ 🌖 multi-step (value function control) online learning -----------------------
class vSarsaλ(vMDP):
    def __init__(self, λ=.5, **kw):
        super().__init__(**kw)
        self.λ = λ
        self.step = self.step_an # for Sarsa, we want to decide the next action in time step t
    
    def step0(self):
        self.Z = self.W*0

    def online(self, s, rn,sn, done, a,an):
        # decay eligibility traces for all actions first, then update the trace for the selected action
        self.Z    *= self.λ * self.γ
        self.Z[a] += self.ΔQ(s)
        self.W    += self.α*(rn + (1-done)*self.γ*self.Q(sn,an)- self.Q(s,a))*self.Z

# ------------------------ 🌖 multi-step, value function control, online learning -----------------------
class vtrueSarsaλ(vMDP):
    def __init__(self, λ=.5, **kw):
        super().__init__(**kw)
        self.λ = λ
        self.step = self.step_an # for Sarsa we want to decide the next action in time step t
    
    def step0(self):
        self.Z = self.W*0
        self.qo = 0
        
    def online(self, s, rn,sn, done, a,an):
        
        α, γ, λ = self.α, self.γ, self.λ
        
        self.q  = self.Q(s,a)
        self.qn = self.Q(sn,an)*(1-done)
        δ = rn + γ*self.qn - self.q
        self.Z    *= λ*γ
        self.Z[a] += (1 - α*λ*γ*self.Z[a]@s)*s

        self.W    += α *(self.q - self.qo + δ) * self.Z
        self.W[a] -= α *(self.q - self.qo    ) * s

        self.qo = self.qn
# ------------------------ 🌖 multi-step, value function control, online learning -----------------------
'''
This method combines both online and offline learning; it is similar to Sarsaλ but cleaner, as it separates the online stage from the delayed credit backpropagation stage. Therefore, it might be more desirable, particularly for sparse rewards
'''
class vSarsa_online_offline(vMDP):

    def init(self):
        self.step = self.step_an # for Sarsa, we want to decide the next action in time step t
        self.store = True

    def online(self, s, rn,sn, done, a,an):
        self.W[a] += self.α*(rn + (1-done)*self.γ*self.Q(sn,an) - self.Q(s,a))*self.ΔQ(s)

    def offline(self):  
        for t in range(self.t - self.ep, -1, -1):
            s = self.s[t]
            a = self.a[t]
            sn = self.s[t+1]
            an = self.a[t+1]
            rn = self.r[t+1]
            
            self.W[a] += (self.α/4)*(self.Q(sn,an) - self.Q(s,a))*self.ΔQ(s)
# ===================all *continuous-action policy-gradient control algorithms* must inherit this class============================

'''
    overriding  π() in parent class MDP: 
    in MDP  π() returns probabilities according to an εgreedy     [ not used in discrete action update]
    in vPG  π() returns probabilities according to a τsoftmax     [(1-π) used in discrete action update]
    in vPGc π() returns probabilities according to a Gaussian     [ not used in continuous action update]
    
    Continuous action may comprise multiple components, each with a continuous value.
    The mean is calculated via μ_() via W, which assigns to each action component a separate 
    weight W[a], then perform W@s
'''
class vPG(PG(vMDP)):
    def __init__(self, αv=None, αq=None, **kw):
        super().__init__(**kw)
        self.αv = αv if αv is not None else self.α*10
        self.αq = αq if αq is not None else self.α
        self.policy = self.τsoftmax

    def init_(self):
        self.w = np.ones(self.env.nF)*self.v0
        self.W = np.ones((self.env.nA, self.env.nF))*self.q0
        self.Θ = np.ones((self.env.nA, self.env.nF))*self.h0
        
        self.V_ = self.V
        self.Q_ = self.Q
        self.H_ = self.H  # for softmax, not for Gaussian
    
    def H(self, s=None, a=None):
        Θ = self.Θ if a is None else self.Θ[a]
        return Θ@s if s is not None else np.matmul(Θ, self.env.S_()).T 

    # we should have used ∇ , but Python does not like it
    def ΔH (self, s): 
        return s
        
    # This function is for the softmax; it extends the softmax Δlogπ to a linear approximation. 
    def Δlogπ(self, s, a):  # ∇ log π(s,a)
        return super().Δlogπ(s,a)[:, None] @ s[None, :] 

    # The rest are defined with the arent PG class

# =========================================================================================================

class vPGc(PG(vMDP)):
    def __init__(self, αv=None, αq=None, μ0=0, σ=1, σmin=.01, dσ=1, Tσ=0, **kw):
        super().__init__(**kw)
        self.μ0 = μ0
        self.σ = σ
        self.σ0 = σ
        self.dσ = dσ
        self.Tσ = Tσ
        self.σmin = σmin

        self.ϴ = np.ones((self.env.nA, self.env.nF))*self.μ0

        self.αv = αv if αv is not None else self.α*10
        self.αq = αq if αq is not None else self.α
        
        # Gaussian is the default policy to sample an action from for Policy Gradient methods
        self.policy = self.Gaussian
    
    def init_(self):
        self.w = np.ones(self.env.nF)*self.v0
        self.W = np.ones((self.env.nA, self.env.nF))*self.q0
        self.Θ = np.ones((self.env.nA, self.env.nF))*self.h0
        
        self.V_ = self.V
        self.Q_ = self.Q
        # self.H_ = self.H  # for softmax, not for Gaussian
        
    # a here represents the action component index, not an action value or an action index
    # As we can see, this parametrisation for the μ means we are parametrising the policy.
    def μ_π(self, s, a=None):
        # ϴ @ s
        ϴ = self.ϴ if a is None else self.ϴ[a]
        return  np.atleast_1d(ϴ@s)
        
    def σ_π(self, s, a=None):
        # W @ s
        return self.σ # fixed σ here, passed by user, not learned for simplicity, the function defined for extensibility
        
    #------------------------------------- continuous policy 🧠------------------------------------
    # samples a Gaussian policy π to obtain a continuous action value, or a vector of Gaussian action component values
    def Gaussian(self, s):

        if self.dσ < 1: self.σ = max(self.σmin, self.σ  *self.dσ)                  # exponential decay
        if self.Tσ > 0: self.σ = max(self.σmin, self.σ0 * (1 - self.t_ / self.Tσ)) # linear      decay

        μ = self.μ_π(s) # ϴ @ s
        σ = self.σ      # passed by user, not learned for simplicity
        
        # sample an action value from the Gaussian
        a = np.random.normal(μ, σ) # a = μ + σ * randn(*μ.shape)
        a = np.clip(a, self.env.action_space.low, self.env.action_space.high)  # clip for safety          
        return np.atleast_1d(a)

    # action probability as per the Gaussian formula: mainly for reference and will not be called directly.
    def π(self, s, a):     # pr(a|s)
        μ = self.μ_π(s)    # ϴ @ s
        σ = self.σ_π(s)    # W @ s
        
        p = (1.0 / (np.sqrt(2 * np.pi) * σ)) * np.exp(-((a - μ) ** 2) / (2 * σ**2))
        return np.prod(p)
    
    # returns the log of π: mainly for reference and will not be called directly, instead we need ∇logπ 
    def logπ(self, s, a):   # gaussian logπ vector: log pr(a|s)

        μ = self.μ_π(s)     # is     ϴ @ s
        σ = self.σ_π(s)     # can be W @ s

        logπ = -((a-μ)**2)/(2*σ**2) - np.log(σ ) - .5*np.log(2*np.pi)
        return np.sum(logπ)

    # we should have used ∇ , but Python does not like it
    # gradient of the log of the policy π that appears in the **policy gradient theorem**
    def Δlogπ(self, s, a):  # ∇ log π(s,a)
        μ = self.μ_π(s)     # ϴ @ s
        σ = self.σ_π(s)     # W @ s
        a = np.atleast_1d(a)
        
        Δlogπ = ((a - μ ) / (σ**2))[:, None] @ s[None, :] # each component of μ has to be multiplied by vector s
        return Δlogπ

# --------------------  🌘  offline REINFORCE: policy gradient 🧠 control learning continuos actions------------
def ENFORCE(_PG_=vPG):
    class vREINFORCE(_PG_):
        def init(self): 
            self.store = True
    
        def offline(self):
            # Δlogπ vPG performs outer prodcut Δlogπ(s,a)@ΔQ(s)
            Δlogπ, ΔV, γ, αv, αq, τ = self.Δlogπ, self.ΔV, self.τ, self.γ, self.αv, self.αq, getattr(self, 'τ', 1)
            
            # obtain the return for the latest episode
            Gt = 0
            γt = γ**self.t                  # efficient way to calculate powers of γ backwards
            for t in range(self.t, -1, -1): # reversed to make it easier to calculate Gt
                s = self.s[t]
                a = self.a[t]
                rn = self.r[t+1]
                
                Gt = γ*Gt + rn
                δ = Gt - self.V(s)
    
                self.w += αv*δ*ΔV(s)
                self.Θ += αq*δ*Δlogπ(s,a)*(γt/τ) # @ outer product: update all actions; ∇π involves all actions
                γt /= γ
    
    return vREINFORCE

vREINFORCE  = ENFORCE(vPG)    # discrete  — softmax
vREINFORCEc = ENFORCE(vPGc)   # continuous — Gaussian 

# -------------- 🌖 online Actor-Critic: policy gradient 🧠 control learning continuos actions--------------
def AC(_PG_=vPG):
    class vActor_Critic(_PG_):
        def step0(self):
            self.γt = 1 # powers of γ
            
        def online(self, s, rn,sn, done, a,_): 
            # Δlogπ vPG performs outer prodcut Δlogπ(s,a)@ΔQ(s)
            Δlogπ, ΔV, γ, γt, αv, αq, τ = self.Δlogπ, self.ΔV, self.γ, self.γt, self.αv, self.αq, getattr(self, 'τ', 1)
            δ = (1- done)*γ*self.V(sn) + rn - self.V(s)    # TD error is based on the critic estimate
            self.w  += αv*δ*ΔV(s)                 # critic
            self.Θ  += αq*δ*Δlogπ(s,a)*γt/τ       # actor  
            self.γt *= γ
    
    return vActor_Critic
    
vActor_Critic  = AC(vPG)    # discrete  — softmax
vActor_c_Critic = AC(vPGc)  # continuous — Gaussian 


def AC(_PG_=vPG):
    class vActor_Critic(_PG_):
        def __init__(self, λ=0, **kw):
            super().__init__(**kw)
            self.λ = λ

        def step0(self):
            self.γt = 1
            self.z  = self.w * 0  # critic eligibility trace

        def online(self, s, rn, sn, done, a, _):
            Δlogπ, ΔV, γ, γt, αv, αq, λ = self.Δlogπ, self.ΔV, self.γ, self.γt, self.αv, self.αq, self.λ
            τ = getattr(self, 'τ', 1)

            δ = (1-done)*γ*self.V(sn) + rn - self.V(s)

            self.z   = λ*γ*self.z + ΔV(s)          # critic trace — accumulate
            self.w  += αv*δ*self.z                  # critic — use trace
            self.Θ  += αq*δ*Δlogπ(s,a)*γt/τ        # actor  — no trace for now
            self.γt *= γ

    return vActor_Critic

vActor_Critic   = AC(vPG)    # discrete  — softmax
vActor_c_Critic = AC(vPGc)   # continuous — Gaussian
# =========================a set of useful prediction comparisons =========================================

def TDtiledwalk(ntilings):
    env=tiledrandwalk_(nS=20, tilesize=4, offset=1, ntilings=ntilings)
    vTD(env=env, α=.02, episodes=200, **demoE()).interact(label='TD learning, %d tilings'%ntilings)

def TDλ_MC_Walk_Compare_αλ(algorithm=vTDλ, label='TD(λ)', runs=10):
    
    steps0 = list(np.arange(.001,.01,.001))
    steps1 = list(np.arange(.011,.2,.02))
    steps2 = list(np.arange(.25,1.,.05))

    αs = np.round(steps0 +steps1 + steps2, 2)
    #αs = np.arange(0,1.05,.1) # quick testing
    
    plt.xlim(-.02, 1)
    plt.ylim(.24, .56)
    plt.title('%s RMS error Average over 19 states and first 10 episodes'%label)
    for λ in [0, .1, .4, .8, .9, .95, .975, .99, 1]:
        end=34 if λ<.975 else (-3 if λ<.99 else -10)
        compare = Compare(algorithm=algorithm(env=vrandwalk_(), v0=0, λ=λ, episodes=10), 
                                  runs=runs, 
                                  hyper={'α':αs[:end]}, 
                                  plotE=True).compare(label='λ=%.3f'%λ)
    if algorithm==vtrueTDλ:
        compare = Compare(algorithm=vMC(env=vrandwalk_(), v0=0, episodes=10), 
                                  runs=runs, 
                                  hyper={'α':αs}, 
                                  plotE=True).compare(label='MC ≡ TD(λ=1)', frmt='-.')

# =========================a set of useful control comparisons =========================================
def Gym_runs(algo=vSarsa, env=vGymDiscreteS(**CartPole), runs=10,  αs=[.1, .2, .5], αscale=1.0, ε=0.05, episodes=200,
                label='discretised', ylog=True, ylim=None):  
    for α in αs:                                                                     
        sarsaRuns = Runs(algorithm=algo(env=env, α=α/αscale, episodes=episodes, ε=ε),
                         runs=runs, seed=1, plotT=True).interact(label=f'α = {α}/{αscale}')
    if ylog: plt.yscale('log')
    if ylim: plt.ylim(ylim)
    plt.title('Semi Gradient ' + algo.__name__ + ' on ' + env.spec.id + ' ' + label)


def nSarsa_GymTiled_αn_runs(runs=10, Env=CartPole, ε=0.05, episodes=100):
    env = GymTiled(**Env)
    n_tilings = Env['n_tilings']
    plt.title(f"n-step Sarsa with Tiled Coding on {Env['env_id']}: comparison of n with {n_tilings} tilings")
    for n, α in zip([1, 8], [.1, .3]):
        algoRuns = Runs(algorithm=vSarsan(env=env, n=n, α=α/n_tilings, episodes=episodes, ε=ε), 
                         runs=runs, seed=1, plotT=True).interact(label=f'{n} step-Sarsa, α={α}/{n_tilings}')
    # plt.ylim((10**2,10**3))
    plt.yscale('log')
    plt.show()

import math
def GymTiled_n_tilings_runs(algo=vSarsa, runs=10, α=.3, Env=CartPole, tilings=[2, 4, 8, 16, 32]):
    plt.title(f"{algo.__name__} on {Env['env_id']}: comparison of different tilings with α={α}/n_tilings")
    for n_tilings in tilings:
        total_tiles = CartPole['n_tilings'] * np.prod(CartPole['n_tiles'] ) # n_tilings x ntiles1 x ntiles2...
        hash_size = 2 ** math.ceil(math.log2(total_tiles))                  # get the nearest 2**n
        env = GymTiled(**envDict(Env, hash_size=hash_size, n_tilings=n_tilings))
        algoRuns = Runs(algorithm=algo(env=env,α=α/n_tilings, episodes=500, ε=0), 
                         runs=runs, seed=1, plotT=True).interact(label='%d tilings'%n_tilings)
    # plt.ylim((10**2, 10**3))
    plt.yscale('log')
    plt.show()

def GymTiled_Compare_αn(algo=vSarsan, Env=CartPole, ε=.05, episodes=50, runs=3): 
    env = GymTiled(**Env)
    n_tilings = Env['n_tilings']
    
    plt.title(f'Steps per episode averaged over first {episodes} episodes')
    
    for n in range(5):
        if n==0: αs = np.arange(.4,  1.8,  .1)
        if n==1: αs = np.arange(.2,  1.8,  .1)
        if n==2: αs = np.arange(.1,  1.8,  .1)
        if n==3: αs = np.arange(.1,  1.2,  .07)
        if n==4: αs = np.arange(.1,  1.0,  .07)
    
        Compare(algorithm=algo(env=env, n=2**n, episodes=episodes, ε=ε), runs=runs, 
                                  hyper={'α':αs/n_tilings}, 
                                  plotT=True).compare(label=f'{2**n}-step Sarsa')
    plt.show()
    
def GymTiled_Compare_αλ(algo=vtrueSarsaλ, Env=MountainCar, ε=.05, episodes=50, runs=3):
    
    env = GymTiled(**Env)
    n_tilings = Env['n_tilings']
    plt.title(f'Steps per episode averaged over first {episodes} episodes for {algo.__name__}')

    for λ in [0, .68, .84, .92]:#, .96, .98, .99]:
        if λ>=.0: αs = np.arange(.1,  1.8,  .1)
        if λ>=.6: αs = np.arange(.1,  1.8,  .1)
        if λ>=.8: αs = np.arange(.1,  1.8,  .1)
        if λ>=.9: αs = np.arange(.1,  1.8,  .15)
        if λ>=.98: αs = np.arange(.1,  .7,  .15)
        if λ>=.98: αs = np.arange(.1,  .7,  .07)
    
        Compare(algorithm=algo(env=env, λ=λ, episodes=episodes, ε=ε), runs=runs, 
                                   hyper={'α':αs/n_tilings}, 
                                  plotT=True).compare(label=f'λ={λ}')
    plt.show()
    