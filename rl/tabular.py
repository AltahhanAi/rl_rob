'''
    This library implements tabular methods for well-known 
    RL algorithms. It works by inheriting from the classes in the 
    rl_base library.
'''

from rl.base import *
from rl.select import *
from env.grid.tabular import *


from math import floor
# ======================================================================================
'''
    offline here in the sense of end-of-episode learning, 
    not a pure offline, where there are no in-between episodes learning
'''
# ------------------ 🌘 offline Monte Carlo value function prediction learning -----------------------
class MC(MRP):
    def init(self):
        self.store = True
       
    def offline(self):
        # obtain the return for the latest episode
        Gt = 0
        for t in range(self.t, -1, -1):
            s = self.s[t]
            rn = self.r[t+1]
            
            Gt = self.γ*Gt + rn
            self.V[s] += self.α*(Gt - self.V[s])

# ------------------- 🌘 offline Monte Carlo value function control learning 🧑🏻‍🏫 -----------------------
class MCC(MDP()):
    def init(self):
        self.store = True
        
    def offline(self):  
        # obtain the return for the latest episode
        Gt = 0
        for t in range(self.t, -1, -1):
            s = self.s[t]
            a = self.a[t]
            rn = self.r[t+1]

            Gt = self.γ*Gt + rn
            self.Q[s,a] += self.α*(Gt - self.Q[s,a])

# ------------------- 🌘 offline, REINFORCE: MC for policy gradient 🧠 control methods ----------------
# In the tabular case, there is no point in using PG_cont as actions cannot be indexed
class REINFORCE(PG()):
    def init(self):
        self.store = True
    
    def offline(self):
        π, γ, α, τ = self.π, self.γ, self.α, self.τ
        # obtain the return for the latest episode
        Gt = 0
        γt = γ**self.t                  # efficient way to calculate powers of γ backwards
        for t in range(self.t, -1, -1): # reversed to make it easier to calculate Gt
            s = self.s[t]
            a = self.a[t]
            rn = self.r[t+1]
            
            Gt = γ*Gt + rn
            δ = Gt - self.V[s]
            
            self.V[s]   += α*δ
            self.Q[s,a] += α*δ*(1 - π(s,a))*γt/τ
            γt /= γ

# -------------------- 🌖 online Temporal Difference: value prediction learning ------------------------
class TD(MRP):  
    def online(self, s, rn,sn, done, *args): 
        self.V[s] += self.α*(rn + (1- done)*self.γ*self.V[sn] - self.V[s])

# -------------------- 🌘 offline Temporal Difference(TD): value prediction learning ----------------------
class TDf(MRP):
    def init(self):
        self.store = True
  
    def offline(self):
        #for t in range(self.t, -1, -1):
        for t in range(self.t+1):
            s = self.s[t]
            sn = self.s[t+1]
            rn = self.r[t+1]
            done = self.done[t+1]
            
            self.V[s] += self.α*(rn + (1- done)*self.γ*self.V[sn]- self.V[s])

# -------------------- 🌖 online multi-step TD: value prediction learning ---------------------------------
class TDn(MRP):          
    def init(self):
        self.store = True # there is a way to save storage by using t%(self.n+1) but we left it for clarity
       
    def online(self,*args):
        τ = self.t - (self.n-1);  n = self.n 
        if τ<0: return

        # we may be able to deal with varying n using this implementation
        # we take the min so that we do not exceed the episode limit (last step+1)
        τn = τ+n ; τn = min(τn, self.t+1 - self.skipstep)
        τ1 = τ+1
        
        sτ = self.s[τ ]
        sn = self.s[τn]
        done = self.done[τn]
        
        # n steps τ+1,..., τ+n inclusive of both ends
        self.V[sτ] += self.α*(self.G(τ1,τn) + (1- done)*self.γ**n *self.V[sn] - self.V[sτ])
    
# -------------------- 🌘 offline multi-step TD: value prediction learning ---------------------------
class TDnf(MRP):
    def init(self):
        self.store = True # must store because it is offline
     
    def offline(self):
        n=self.n        
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
            self.V[sτ] += self.α*(self.G(τ1,τn)+ (1- done)*self.γ**n *self.V[sn] - self.V[sτ])

# -------------------- 🌖 online Sarsa: value control learning -----------------------------------------
class Sarsa(MDP()):
    def init(self): #α=.8
        self.step = self.step_an # for Sarsa we want to decide the next action in time step t
    
    def online(self, s, rn,sn, done, a,an):
        self.Q[s,a] += self.α*(rn + (1- done)*self.γ*self.Q[sn,an] - self.Q[s,a])

# -------------------- 🌖 online multi-step Sarsa: value control learning -------------------------------
class Sarsan(MDP()):
    def init(self):
        self.store = True        # although online but we need to access *some* of earlier steps,
        self.step = self.step_an # for Sarsa we want to decide the next action in time step t
    
    # ----------------------------- 🌖 online learning ----------------------    
    def online(self,*args):
        τ = self.t - (self.n-1);  n=self.n
        if τ<0: return
        
        # we take the min so that we do not exceed the episode limit (last step+1)
        τ1 = τ+1
        τn = τ+n ; τn=min(τn, self.t+1 - self.skipstep)
        
        sτ = self.s[τ];  aτ = self.a[τ]
        sn = self.s[τn]; an = self.a[τn]
        done = self.done[τn]
        
        # n steps τ+1,..., τ+n inclusive of both ends
        self.Q[sτ,aτ] += self.α*(self.G(τ1,τn) + (1- done)*self.γ**n *self.Q[sn,an] - self.Q[sτ,aτ])

# -------------------- 🌖 online Q-learning: value control learning ------------------------------------
class Qlearn(MDP()):
    def online(self, s, rn,sn, done, a,_):
        self.Q[s,a] += self.α*(rn + (1- done)*self.γ*self.Q[sn].max() - self.Q[s,a])

# -------------------- 🌖 online Expected Sarsa: value control learning --------------------------------
class XSarsa(MDP()):
    def online(self, s, rn,sn, done, a,_):      
        # obtain the ε-greedy policy probabilities, 
        # then obtain the expectation via a dot product for efficiency
        π = self.π(sn)
        v = self.Q[sn].dot(π)
        self.Q[s,a] += self.α*(rn + (1- done)*self.γ*v - self.Q[s,a])

# -------------------- 🌖 online double Q-learning: value control learning -------------------------------
class DQlearn(MDP()):
    def init(self):
        self.Q1 = self.Q
        self.Q2 = self.Q.copy()
        
    # We need to override the action-value function in our εgreedy policy
    def Q_(self, s=None, a=None):
            return self.Q1[s] + self.Q2[s] if s is not None else self.Q1 + self.Q2

    def online(self, s, rn,sn, done, a,_): 
        p = np.random.binomial(1, p=0.5)
        if p:    self.Q1[s,a] += self.α*(rn + (1- done)*self.γ*self.Q2[sn].max() - self.Q1[s,a])
        else:    self.Q2[s,a] += self.α*(rn + (1- done)*self.γ*self.Q1[sn].max() - self.Q2[s,a])

# -------------------- 🌖 online QV-learning: value control learning ------------------------------------
class QVlearn(MDP()):# suitable for dense reward (intermediate steps rewards)
    def online(self, s, rn,sn, done, a,an):
        self.V[s]   += self.α*(rn + (1- done)*self.γ*self.V[sn] - self.V[s])
        self.Q[s,a] += self.α*(rn + (1- done)*self.γ*self.V[sn] - self.Q[s,a])

# -------------------- 🌖 online QV-learning with Eligibility Traces: value control learning ------------   
class QVλlearn(MDP()):# suitable for dense reward (intermediate steps rewards)
    def __init__(self, λ=.8, **kw):
        super().__init__(**kw)
        self.λ = λ

    def step0(self):
        self.z = self.V*0

    def online(self, s, rn,sn, done, a,an):
        self.z[s] += 1
        self.z = self.λ*self.γ*self.z
        
        self.V[s]   += self.α*(rn + (1- done)*self.γ*self.V[sn] - self.V[s])*self.z[s]
        self.Q[s,a] += self.α*(rn + (1- done)*self.γ*self.V[sn] - self.Q[s,a])

# -------------------- 🌖 online Actor-Critic: policy gradient 🧠 control learning ------------------------
# In the tabular case, there actions are usually discrete
class Actor_Critic(PG()):
    def step0(self):
        self.γt = 1 # powers of γ, must be reset at the start of each episode
    
    def online(self, s, rn,sn, done, a,an): 
        π, γ, γt, α, τ, t = self.π, self.γ, self.γt, self.α, self.τ, self.t
        δ = rn + (1- done)*γ*self.V[sn] - self.V[s]  # TD error is based on the critic estimate

        self.V[s]   += α*δ                           # critic
        self.Q[s,a] += α*δ*(1- π(s,a))*γt/τ          # actor
        self.γt *= γ

# ===================adding a few functions that will serve us well for comparing famous algorithms=====================
def TD_MC_randwalk(env=randwalk(), alg1=TDf, alg2=MC):
    plt.xlim(0, 100)
    plt.ylim(0, .25)
    plt.title('Empirical RMS error, averaged over states')
    
    for α in [.05, .1, .15]:
        TDαs = Runs(algorithm=alg1(env=env, α=α, v0=.5), runs=100, plotE=True).interact(label='TD α= %.2f'%α, frmt='-')

    for α in [.01, .02, .03, .04]:
        MCs = Runs(algorithm=alg2(env=env, α=α, v0=.5), runs=100, plotE=True).interact(label='MC α= %.2f'%α, frmt='--')

def example_6_2(**kw): return TD_MC_randwalk(**kw)

# ----------------------------------------------------------------------------------------------------------------------
def figure_6_2():
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.xlim(0,100)
    plt.ylim(0, .25)
    plt.title('Batch Training')

    α=.001
    TDB = Runs(algorithm=TD_batch(v0=-1, α=α, episodes=100), runs=100, plotE=True).interact(label= 'Batch TD, α= %.3f'%α)
    MCB = Runs(algorithm=MC_batch(v0=-1, α=α, episodes=100), runs=100, plotE=True).interact(label='Batch MC, α= %.3f'%α)
# ----------------------------------------------------------------------------------------------------------------------
def Sarsa_windy():
    return Sarsa(env=windy(reward='reward1'), α=.5, seed=1, **demoQ, episodes=170).interact(label='TD on Windy')
    
example_6_5 = Sarsa_windy

# ----------------------------------------------------------------------------------------------------------------------
def Sarsa_Qlearn_cliffwalk(runs=200, α=.5, env=cliffwalk(), alg1=Sarsa, alg2=Qlearn):
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)    
    plt.yticks([-100, -75, -50, -25])
    plt.ylim(-100, -10)
    
    SarsaCliff = Runs(algorithm=alg1(env=env, α=α, episodes=500), runs=runs, plotR=True).interact(label='Sarsa')
    QlearnCliff = Runs(algorithm=alg2(env=env, α=α, episodes=500), runs=runs, plotR=True).interact(label='Q-learning')
    return SarsaCliff, QlearnCliff

def example_6_6(**kw): 
    return Sarsa_Qlearn_cliffwalk(**kw)

# ----------------------------------------------------------------------------------------------------------------------
def XSarsaDQlearnCliff(runs=300, α=.5):
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)    
    plt.yticks([-100, -75, -50, -25])
    plt.ylim(-100, -10)
    env = cliffwalk()

    XSarsaCliff = Runs(algorithm=XSarsa(env=env, α=α, episodes=500), runs=runs, plotR=True).interact(label='XSarsa')
    DQlearnCliff = Runs(algorithm=DQlearn(env=env, α=α, episodes=500), runs=runs, plotR=True).interact(label='Double Q-learning')

    return XSarsaCliff, DQlearnCliff

# ----------------------------------------------------------------------------------------------------------------------
def compareonMaze(runs=100, α=.5):
    
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    
    env=Grid(gridsize=[10,20], style='maze', s0=80, reward='reward1') # this is bit bigger than the defualt maze
    env.render()
    
    SarsaMaze = Runs(algorithm=Sarsa(env=env, α=α, episodes=30), runs=runs, plotT=True).interact(label='Sarsa')
    XSarsaMaze = Runs(algorithm=XSarsa(env=env, α=α, episodes=30), runs=runs, plotT=True).interact(label='XSarsa')
    
    QlearnMaze = Runs(algorithm=Qlearn(env=env, α=α, episodes=30), runs=runs, plotT=True).interact(label='Q-learning')
    DQlearnMaze = Runs(algorithm=DQlearn(env=env, α=α, episodes=30), runs=runs, plotT=True).interact(label='Double Q-learning')

    return SarsaMaze, XSarsaMaze, QlearnMaze, DQlearnMaze

# ----------------------------------------------------------------------------------------------------------------------
def figure_6_3(runs=10, Interim=True, Asymptotic=True, episodes=100,  label=''): #100
    #plt.ylim(-150, -10)
    plt.xlim(.1,1)
    plt.title('Interim and Asymptotic performance')
    αs = np.arange(.1,1.05,.05)

    algors = [ XSarsa,   Sarsa,   Qlearn]#,      DQlearn]
    labels = ['XSarsa', 'Sarsa', 'Qlearning']#, 'Double Q learning']
    frmts  = ['x',      '^',     's']#,         'd']
    
    env = cliffwalk()
    Interim_, Asymptotic_ = [], []
    # Interim perfromance......
    if Interim:
        for g, algo in enumerate(algors):
            compare = Compare(algorithm=algo(env=env, episodes=episodes), runs=runs, hyper={'α':αs},
                             plotR=True).compare(label=labels[g]+' Interim'+label, frmt=frmts[g]+'--')
            Interim_.append(compare)
    
    # Asymptotic perfromance......
    if Asymptotic:
        for g, algo in enumerate(algors):
            compare = Compare(algorithm=algo(env=env, episodes=episodes*10), runs=runs, hyper={'α':αs}, 
                             plotR=True).compare(label=labels[g]+' Asymptotic'+label, frmt=frmts[g]+'-')
            Asymptotic_.append(compare)
    
    plt.gcf().set_size_inches(10, 7)
    return Interim_, Asymptotic_
    

# ----------------------------------------------------------------------------------------------------------------------
def nstepTD_MC_randwalk(env=randwalk(), algorithm=TDn, alglabel='TD'):
    plt.xlim(0, 100)
    plt.ylim(0, .25)
    plt.title('Empirical RMS error, averaged over states')
    n=5
    
    for α in [.05, .1, .15]:
        TDαs = Runs(algorithm=algorithm(env=env, n=1,α=α, v0=.5),  runs=100, plotE=True).interact(label='%s α= %.2f'%(alglabel,α), frmt='.-')
    
    for α in [.05, .1, .15]:
        TDαs = Runs(algorithm=algorithm(env=env,n=n,α=α, v0=.5),  runs=100, plotE=True).interact(label= '%s α= %.2f n=%d'%(alglabel,α,n), frmt='-')

    for α in [.01, .02, .03, .04]:
        MCs = Runs(algorithm=MC(env=env,α=α, v0=.5),  runs=100, plotE=True).interact(label='MC α= %.2f'%α, frmt='--')

# ----------------------------------------------------------------------------------------------------------------------
def nstepTD_MC_randwalk_αcompare(env=randwalk_(), algorithm=TDn, Vstar=None, runs=10, envlabel='19', 
                                 MCshow=True, alglabel='online TD'):
    
    steps0 = list(np.arange(.001,.01,.001))
    steps1 = list(np.arange(.011,.2,.025))
    steps2 = list(np.arange(.25,1.,.05))

    αs = np.round(steps0 +steps1 + steps2, 2)
    #αs = np.arange(0,1.05,.1) # quick testing
    
    plt.xlim(-.02, 1)
    plt.ylim(.24, .56)
    plt.title('n-steps %s RMS error averaged over %s states and first 10 episodes'%(alglabel,envlabel))
    for n in [2**_ for _ in range(10)]:
        Compare(algorithm=algorithm(env=env, v0=0, n=n, episodes=10, Vstar=Vstar), 
                              runs=runs, 
                              hyper={'α':αs}, 
                              plotE=True).compare(label='n=%d'%n)
    if MCshow:
        compare = Compare(algorithm=MC(env=env, v0=0, episodes=10), 
                                  runs=runs, 
                                  hyper={'α':αs}, 
                                  plotE=True).compare(label='MC ≡ TDn(n=$\\infty$)', frmt='-.')
# ----------------------------------------------------------------------------------------------------------------------
def figure_7_4(n=5,seed=16): 
    
    # draw the path(trace) that the agent took to reach the goal
    nsarsa = Sarsan(env=grid(), α=.4, seed=seed, episodes=1).interact()
    nsarsa.env.render(underhood='trace', subplot=131, animate=False, label='path of agent')

    # now draw the effect of learning to estimate the Q action-value function for n=1
    nsarsa = Sarsan(env=grid(), α=.4, seed=seed, episodes=1, underhood='maxQ').interact() 
    nsarsa.render(subplot=132, animate=False, label='action-value increassed by 1-steps Sarsa\n')
    
    #n=5 # try 10
    # now draw the effect of learning to estimate the Q action-value function for n=10
    nsarsa = Sarsan(env=grid(), n=n, α=.4, seed=seed, episodes=1, underhood='maxQ').interact()    
    nsarsa.render(subplot=133, animate=False, label='action-value increassed by %d-steps Sarsa\n'%n)
# ----------------------------------------------------------------------------------------------------------------------