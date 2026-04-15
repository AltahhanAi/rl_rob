'''
Abdulrahman Altahhan (c) 2026
Version: 3.8
Educational code for teaching RL (DQN and related methods).
Permission required for redistribution or research/commercial use.
'''
from rl.linear import *
from rl.nn import *
from env.grid.neural import *

# ===============================================================================================

# ========= This is where RL enters, the above will be data members of the below classes ========
# ================ nnMRP, nnMDP, PG Classes with Neural Net =====================================
#(32, 8, 4), (64, 4, 2), (64, 3, 1)],
class nnMRP(MRP):
    def __init__(self,
                 trunk=[],
                 final_bias=False,
                 nF=None, rndbatch=True,
                 nbuffer=1, nbatch=1, endbatch=1,
                 save_weights=1000, load_weights=False,
                 create_w=True,
                 create_wn=False, 
                 t_Vn=0,
                 action_dtype=torch.int64,
                 model_summary=True,
                 model_class=nnModel, # which type of neural network model from nn.py to create
                 clipCNN=True,
                 β_entropy=0.01,
                 **kw):
        self.model_summary = model_summary
        print(f'------------------- 易  {self.__class__.__name__} is being set up 易 ---------------------') if model_summary else None
        super().__init__(**kw)
        self.store     = True
        self.create_w  = create_w
        self.create_wn = create_wn
        self.t_Vn      = t_Vn
        
        self.trunk      = trunk
        self.nF         = nF
        self.final_bias = final_bias
        
        self.action_dtype = action_dtype
        self.model_class  = model_class
        
        if endbatch > nbatch: endbatch = nbatch - 1
        self.endbatch = endbatch
        self.nbuffer  = nbuffer
        self.nbatch   = nbatch
        self.rndbatch = rndbatch
        self.buffer   = deque(maxlen=self.nbuffer)
        
        self.load_weights_ = load_weights
        self.save_weights_ = save_weights
        self.t_ = 0
        self.β_entropy = β_entropy
        self.clipCNN = clipCNN
        self.w  = self.create_model('V',  self.model_class) if create_w  else None
        self.wn = self.create_model('Vn', self.model_class) if create_wn else None
        

    def init_(self):
        torch.manual_seed(self.seed)
        self.w.load_weights('V') if self.load_weights_ else self.w.init_weights(head_v0=self.v0)
        self.wn.eval() if self.create_wn else None
        self.V_ = self.V
    
    def create_model(self, net_str, model_class):
        self.state_dim  = self.env.reset().shape
        self.action_dim = 1 if net_str == 'V' else self.env.nA
        model = model_class(
            inp_dim=self.state_dim, trunk=self.trunk,
            nF=self.nF, out_dim=self.action_dim,
            α=self.α, αv=getattr(self, 'αv', None), αq=getattr(self, 'αq', None),
            τ=getattr(self, 'τ', 1.0),                    # pass τ, default to 1.0 if not set
            β_entropy=getattr(self, 'β_entropy', 0.0),
            net_str=net_str, final_bias=self.final_bias,
            clipCNN=self.clipCNN,  
        )
        if self.model_summary: model.print_model_summary(net_str)
        return model
        
    # def create_model(self, net_str, model_class):
    #     self.state_dim  = self.env.reset().shape
    #     self.action_dim = 1 if net_str == 'V' else self.env.nA
    #     model = model_class(
    #         inp_dim=self.state_dim, trunk=self.trunk,
    #         nF=self.nF, out_dim=self.action_dim,
    #         α=self.α, αv=getattr(self, 'αv', None), αq=getattr(self, 'αq', None), 
    #         net_str=net_str, final_bias=self.final_bias
    #     )
    #     if self.model_summary: model.print_model_summary(net_str)
    #     return model

    # def V(self, s=None):
    #     result = self.w.predict(s if s is not None else self.env.S_(), self.state_dim)
    #     return result.detach().numpy()

    # def V(self, s=None):
    #     result = self.w.predict(s if s is not None else self.env.S_(), self.state_dim)
    #     return result.detach().numpy().squeeze(-1)  # (nS,1) → (nS,)

    def V(self, s=None):
        result = self.w.predict(s if s is not None else self.env.S_(), self.state_dim)
        return np.squeeze(result.detach().numpy())
        
    # def Vn(self, sn):
    #     return self.wn.predict(sn, self.state_dim) if self.create_wn else None

    def Vn(self, sn):
        if not self.create_wn: return None
        return np.squeeze(self.wn.predict(sn, self.state_dim).detach().numpy())

    # must not use the buffer directly in learning, as it will give unsuitable shaped tensor
    def allocate(self):
        self.buffer = deque(maxlen=self.nbuffer)

        
    def slice(self, nbatch):
        buffer = self.buffer
        return list(islice(buffer, len(buffer) - nbatch, len(buffer)))


    def store_(self, s=None, a=None, rn=None, sn=None, an=None, done=None, t=0):
        self.buffer.append((
            torch.tensor(s,    dtype=torch.float32),
            torch.tensor(a,    dtype=self.action_dtype),
            torch.tensor(rn,   dtype=torch.float32).unsqueeze(0),
            torch.tensor(sn,   dtype=torch.float32),
            torch.tensor(done, dtype=torch.bool)
        ))

    # for the life stream to work but it is bit vulnerable
    # def type_convert(self, s,rn,sn, a,an, done ):
    #     return self.batch(t=-1)[0]
    
    # alias for batch, since batch guarantees that the items returned have the right shape
    def trajectory(self, t=-1):
        return self.batch(t=t)[0]
    
    def batch(self, nbatch=None, endbatch=None, t=None):
        nbatch   = self.nbatch   if nbatch   is None else nbatch
        endbatch = self.endbatch if endbatch is None else endbatch

        if t is not None:
            samples = [self.buffer[t]]    # get item t from the buffer and convert it
            
        elif self.rndbatch:
            samples = sample(self.buffer, nbatch - endbatch)
            if endbatch:
                samples.extend(self.slice(endbatch))
        else:
            samples = self.slice(nbatch)

        s, a, rn, sn, dones = zip(*samples)

        s     = torch.stack(s)
        a     = torch.stack(a)
        rn    = torch.stack(rn)
        sn    = torch.stack(sn)
        dones = torch.stack(dones)
        inds  = torch.arange(len(samples))
        return (s, a, rn, sn, dones), inds

# ===============================================================================================
class nnMDP(MDP(nnMRP)):
    def __init__(self, create_w=False, create_W=True, create_Wn=False, t_Qn=1000, **kw):
        super().__init__(create_w=create_w, **kw)
        self.create_Wn = create_Wn
        self.t_Qn = t_Qn
        self.W  = self.create_model('Q',  self.model_class) if create_W else None
        self.Wn = self.create_model('Qn', self.model_class) if create_Wn else None

    def init_(self):
        torch.manual_seed(self.seed)
        if self.create_w:
            self.w.load_weights('V') if self.load_weights_ else self.w.init_weights(head_v0=self.v0)
        self.W.load_weights('Q') if self.load_weights_ else self.W.init_weights(head_v0=self.q0)
        self.Wn.eval() if self.create_Wn else None
        self.V_ = self.V
        self.Q_ = self.Q

    def Q(self, s):
        return self.W.predict(s, self.state_dim)

    def Qn(self, sn):
        return self.Wn.predict(sn, self.state_dim) if self.create_Wn else None

# ===============================================================================================
class nnPG(PG(nnMDP)):
    def __init__(self,  **kw):
        super().__init__(create_w=False, create_W=False, create_Wn=False, **kw)
        # nnAC_SharedModel returns two part the V for the critic and Mu and sigma for the Actor
        # no need to initilaise the w independently unless we do nto want to share the same 
        # trunk between the actor and the critic
        
        self.wϴ = self.create_model(net_str='wϴ',  model_class=nnACSharedModel) # discrete actions
        self.policy = self.softmax

    def init_(self):
        torch.manual_seed(self.seed)
        self.wϴ.load_weights('ϴ') if self.load_weights_ else self.wϴ.init_weights(head1_v0=self.v0, head2_q0=self.q0)
        
        self.V_ = self.V
        # self.Q_ = self.Q
        self.H_ = self.H

    def V(self, s=None):
        V, _ = self.wϴ.predict(s if s is not None else self.env.S_(), self.state_dim)
        return V.detach().numpy().squeeze(-1)

    def H(self, s=None, a=None):
        _, π = self.wϴ.predict(s if s is not None else self.env.S_(), self.state_dim)
        if a is None: return π.detach().numpy()
        return π[a].detach().numpy()

    def softmax(self, s):
        _, π = self.wϴ.predict(s, self.state_dim)
        π = π.detach().numpy().flatten()              # flatten to 1-d
        a = choices(range(self.env.nA), weights=π, k=1)[0]
        return a
        # return np.random.choice(self.env.nA, p=π)
        
        
# ===============================================================================================
class nnPGc(PG(nnMDP)):
    def __init__(self, σ=1, σmin=.01, dσ=1, Tσ=0, **kw):
        super().__init__(action_dtype=torch.float32, **kw)
        self.σ    = σ
        self.σ0   = σ
        self.dσ   = dσ
        self.Tσ   = Tσ
        self.σmin = σmin
        self.wϴ = self.create_model(net_str='wϴ',  model_class=nnACcSharedModel) # continuous actions
        self.policy = self.Gaussian

    def init_(self):
        torch.manual_seed(self.seed)
        self.wϴ.load_weights('ϴ') if self.load_weights_ else self.wϴ.init_weights(head1_v0=self.v0, head2_q0=self.q0)
        self.V_ = self.V
        # self.Q_ = self.Q

    def V(self, s=None):
        V, _, _ = self.wϴ.predict(s if s is not None else self.env.S_(), self.state_dim)
        return V.detach().numpy().squeeze(-1)

    def μ_π(self, s):
        _, μ, _ = self.wϴ.predict(s, self.state_dim)
        return μ.detach().numpy()

    def σ_π(self, s):
        _, _, σ = self.wϴ.predict(s, self.state_dim)
        return σ.detach().numpy()

    def Gaussian(self, s):
        if self.dσ < 1: self.σ = max(self.σmin, self.σ  * self.dσ)
        if self.Tσ > 0: self.σ = max(self.σmin, self.σ0 * (1 - self.t_ / self.Tσ))
        _, μ, σ = self.wϴ.predict(s, self.state_dim)
        μ = μ.detach().numpy()
        σ = σ.detach().numpy()
        a = np.random.normal(μ, σ)
        a = np.clip(a, self.env.action_space.low, self.env.action_space.high)
        return np.atleast_1d(a)

    def π(self, s, a):
        μ, σ = self.μ_π(s), self.σ_π(s)
        return np.prod((1.0 / (np.sqrt(2 * np.pi) * σ)) * np.exp(-((a - μ) ** 2) / (2 * σ**2)))

    def logπ(self, s, a):
        μ, σ = self.μ_π(s), self.σ_π(s)
        return np.sum(-((a - μ)**2) / (2 * σ**2) - np.log(σ) - .5 * np.log(2 * np.pi))

# ===============================================================================================
class nnMC(nnMRP):
    
    # at the start of the run
    def init(self):
        self.nbuffer = self.max_t # nnMRP stores in a buffer always
        
    # at the start of the episode
    def step0(self):
        self.buffer.clear() # clear the buffer after each episode
    # --------------- 🌘 offline, MC learning: end-of-episode learning ----------------  
    def offline(self):
        # obtain the return for the latest episode
        Gt = 0
        for t in range(self.t, -1, -1):
            s, _, rn, _, _ = self.trajectory(t=t)
            Gt = self.γ*Gt + rn
            Vs  = self.w(s)
            self.w.fit(Vs, Gt)           # backprop handles multilayer learning
            
# ===============================================================================================
class nnTDf(nnMRP):
    
    def init(self):
        self.nbuffer = self.max_t                # all nnMRP and subclasses stores 
        
    def step0(self):
        self.buffer.clear()                      # clear the buffer after each episode
    
    # --------------- 🌘 offline TD learning: end-of-episode learning ----------------  
    def offline(self):
        for t in range(self.t, -1, -1):
            s, _, rn, sn, done = self.trajectory(t=t)
            Vs  = self.w(s)
            Vn  = self.w(sn).detach()                           # semi gradient Vn must be detached
            self.w.fit(Vs, (1 - done.float())*self.γ*Vn + rn)   # backprop handles multilayer learning

# =============================================================================================== 
class nnTD(nnMRP):
    # ----------------------------- 🌖 online learning ------------------------------
    def online(self, *args): 
        s, _, rn, sn, done = self.trajectory(-1) # obtain the latest trajectory
        Vs  = self.w(s)
        Vn  = self.w(sn).detach()                              # detach ensures semi-gradient        
        self.w.fit(Vs, (1-done.float())*self.γ * Vn  + rn)     # backprop handels multi-layer learning

# ===============================================================================================
class TDN_(nnMRP):# without a target
    # ----------------------------- 🌖 online learning ----------------------  
    def online(self, *args):
        if len(self.buffer) < self.nbatch: return  # wait until we have nbatch entries in the buffer
        (s, _, rn, sn, dones), _ = self.batch() # note that we are taking a batch now instead of one buffer item
    
        Vs  = self.w(s)
        Vn  = self.w(sn).detach()
        Vn[dones] = 0 # equivalent to (1 - dones) but works for a batch
        self.w.fit(Vs, self.γ * Vn + rn)
        
# ===============================================================================================
class TDN(nnMRP):
    # ----------------------------- 🌖 online learning ----------------------  
    def online(self, *args):
        if len(self.buffer) < self.nbatch: return  # wait until we have nbatch entries in the buffer
        (s, _, rn, sn, dones), _ = self.batch() # note that we are taking a batch now instead of one buffer item
        
        Vs = self.w(s)
        Vn = self.Vn(sn) if self.create_wn and self.ep > 2 else self.w(sn).detach().squeeze(-1)
        Vn[dones] = 0 # equivalent to (1 - dones) but works for a batch
 
        self.w.fit(Vs, self.γ * Vn + rn)
        if self.t_Vn and self.t_ % self.t_Vn == 0 and self.create_wn:
            self.wn.set_weights(self.w, 'V', self.t_)
# ===============================================================================================

class nnMCC(nnMDP):
    def init(self):
        self.nbuffer = self.max_t                # all nnMRP and subclasses stores 
        
    def step0(self):
        self.buffer.clear()                      # clear the buffer after each episode
        
    # ---------------------------- 🌘 offline, MC learning: end-of-episode learning-----------------------    
    def offline(self):          
        Gt = 0
        for t in range(self.t, -1, -1):
            s, a, rn, sn, done = self.trajectory(t=t)
            
            Gt = self.γ*Gt + rn
            Qs  = self.W(s)
            
            target = Qs.clone().detach() 
            target[:,a] = Gt
            
            self.W.fit(Qs, target)
# ===============================================================================================
class nnSarsa(nnMDP):

    def init(self):
        self.step = self.step_an # for Sarsa, we want to decide the next action in time step t

    # ----------------------------------------🌖 online learning ----------------------------------------
    def online(self, s, rn,sn, done, a, an):
        s, a, rn, sn, done = self.trajectory(-1) # obtain the latest trajectory
        
        Qs  = self.W(s)
        Qn  = self.W(sn).detach() 

        target = Qs.clone().detach() 
        target[:,a] = (1-done.float())*self.γ * Qn[:,an] + rn

        self.W.fit(Qs, target)
# ===============================================================================================
class nnQlearn(nnMDP):
    # ---------------------------- 🌖 online control learning ----------------------------
    def online(self, *args):
        s, a, rn, sn, done = self.trajectory(-1) # obtain the latest trajectory
        
        Qs  = self.W(s)
        Qn  = self.W(sn).detach() 

        target = Qs.clone().detach() 
        target[:,a] = (1-done.float())*self.γ * Qn.max(1).values + rn

        self.W.fit(Qs, target)
# ===============================================================================================
class DQN_(nnMDP): # there is no target network for this one
    # ----------------------------- 🌖 online learning ---------------------- 
    def online(self, *args):
        if len(self.buffer) < self.nbatch: return
        (s, a, rn, sn, dones), inds = self.batch()
    
        Qs  = self.W(s)
        Qn  = self.W(sn).detach() 
        Qn[dones] = 0

        # Copy Qs so when we fit identical Qs, they cancel out, leaving the max ones that we want to update
        targets = Qs.clone().detach() 
        targets[inds, a] = self.γ * Qn.max(1).values + rn.squeeze(1)
        
        self.W.fit(Qs, targets)
# ===============================================================================================
class DQN(nnMDP):
    # ----------------------------- 🌖 online learning ---------------------- 
    def online(self, *args):
        if len(self.buffer) < self.nbatch: return
        (s, a, rn, sn, dones), inds = self.batch()
        
        Qs  = self.W(s)
        Qn  = self.Wn(sn).detach() if self.create_Wn and self.ep > 2 else self.W(sn).detach()
        Qn[dones] = 0
        
        targets = Qs.clone().detach() 
        targets[inds, a] = self.γ * Qn.max(1).values + rn.squeeze(1)
        
        self.W.fit(Qs, targets)
        
        # copy the Q network weights W to the target Qn network weights Wn every t_Qn steps
        if self.t_Qn and self.t_ % self.t_Qn == 0 and self.create_Wn:
            self.Wn.set_weights(self.W, 'Q', self.t_)



# class DQN(nnMDP):
#     # ----------------------------- 🌖 online learning ---------------------- 
#     def online(self, *args):
#         if len(self.buffer) < self.nbatch: return
#         (s, a, rn, sn, dones), inds = self.batch()
#         Qs  = self.W(s)
#         Qn  = self.Wn(sn).detach() if self.create_Wn and self.ep > 2 else self.W(sn).detach()
#         Qn[dones] = 0
#         targets = Qs.clone().detach()
#         targets[inds, a] = self.γ * Qn.max(1).values + rn
#         loss = self.W.fit(Qs, targets)
#         if self.t_Qn and self.t_ % self.t_Qn == 0 and self.create_Wn:
#             self.Wn.set_weights(self.W, 'Q', self.t_)

# ===============================================================================================
class DDQN(DQN):
    # ----------------------------- 🌖 online learning ---------------------- 
    def online(self, *args):
        if len(self.buffer) < self.nbatch: return
        (s, a, rn, sn, dones), inds = self.batch()
        
        Qs  = self.W(s)
        an  = self.W(sn).detach().argmax(1)
        Qn  = self.Wn(sn).detach() if self.create_Wn and self.ep > 2 else self.W(sn).detach()
        
        Qn[dones] = 0
        targets = Qs.clone().detach()
        targets[inds, a] = self.γ * Qn[inds, an] + rn.squeeze(1)
        
        self.W.fit(Qs, targets, exact=True)
        
        if self.t_Qn and self.t_ % self.t_Qn == 0 and self.create_Wn:
            self.Wn.set_weights(self.W, 'Q', self.t_)
# ===============================================================================================

class DuelDQN(DQN):
    def __init__(self, model_class=nnDuelModel, **kw):
        super().__init__(model_class=model_class, **kw)

# ===================================Eligibility Traces with Neural Net !==============================================

class nnTDλ(nnMRP):
    def __init__(self, λ=0.8, clipCNN=False, **kw):
        super().__init__(clipCNN=clipCNN, **kw)
        self.λ = λ

    def step0(self):
        self.z = Trace(torch.zeros_like(p) for p in self.w.parameters())

    def online(self, s, rn, sn, done, *args):
        s, _, rn, sn, done = self.trajectory(-1)

        Vs = self.w(s)
        Vn = self.w(sn).detach()
        δ  = (rn + (1 - done.float()) * self.γ * Vn - Vs).detach().squeeze()
        
        self.z *= self.γ * self.λ
        self.z += self.w.Δ(Vs)           # ∇V(s)
        self.w.update(δ, self.z)         # θ ← θ + α·δ·z
# ===============================================================================================
class nnSarsaλ(nnMDP):
    def __init__(self, λ=0.8, clipCNN=False, **kw):
        super().__init__(clipCNN=clipCNN, **kw)
        self.λ = λ

    def init(self):
        self.step = self.step_an

    def step0(self):
        self.Z = Trace(torch.zeros_like(p) for p in self.W.parameters()) 

    def online(self, s, rn, sn, done, a, an):
        s, a, rn, sn, done = self.trajectory(-1)

        Qs = self.W(s)
        Qn = self.W(sn).detach()

        δ = ((1 - done.float()) * self.γ * Qn[:, an] + rn.squeeze() - Qs[:, a]).detach().squeeze()

        self.Z *= self.γ * self.λ            # z ← γλz
        self.Z += self.W.Δ(Qs[:, a])         # z ← z + ∇Q(s,a)
        self.W.update(δ, self.Z)             # θ ← θ + α·δ·z

class nnActor_Critic(nnPG):
    # -----------------------🌖 online learning -------------------
    def online(self, *args):
        s, a, rn, sn, done = self.trajectory(-1)   # one step only
        Vn, *_ = self.wϴ(sn)
        Vn = Vn.squeeze(1).detach()
        Vn[done] = 0
        Gt = rn + self.γ * Vn                      # one-step return
        a  = a.to(torch.int64)
        self.wϴ.fit(s, a, Gt)
        
# ===============================================================================================
# def AC(base=nnPG, label='AC'):
#     class nnAC_(base):
#         def online(self, *args):
#             if len(self.buffer) < self.nbatch: return
#             (s, a, rn, sn, dones), inds = self.batch()
#             Vn, *_ = self.wϴ(sn); Vn = Vn.squeeze(1).detach()
#             Vn[dones] = 0
#             Gt = rn + self.γ * Vn
#             a  = torch.tensor(np.array(a.tolist()), dtype=torch.float32 if label == 'continuous' else torch.int64)
#             loss = self.wϴ.fit(s, a, Gt)
#     nnAC_.__name__ = f'AC_{label}'
#     return nnAC_

# # ===============================================================================================
# nnAC  = AC(nnPG,  'discrete')
# nnACc = AC(nnPGc, 'continuous')

# # ===============================================================================================
# '''
# nnRollout replaces the replay buffer with a rollout buffer for on-policy methods.
# It collects full trajectories, computes GAE advantages, and supports multiple epochs
# over the same rollout.
# '''
# class nnPGcrollout(nnPGc):
#     def __init__(self, nsteps=128, epochs=4, λ=0.95, **kw):
#         super().__init__(create_w=False, **kw)
#         self.nsteps = nsteps
#         self.epochs = epochs
#         self.λ      = λ
#         self.s    = []
#         self.a    = []
#         self.rn   = []
#         self.done = []
#         self.logπ = []
#         self.Vb   = []

#     def store_(self, s=None, a=None, rn=None, sn=None, an=None, done=None, t=0):
#         s_t  = torch.tensor(s, dtype=torch.float32).unsqueeze(0)
#         a_t  = torch.tensor(a, dtype=torch.float32)
#         with torch.no_grad():
#             V, μ, σ = self.wϴ(s_t)
#             logπ    = Normal(μ, σ).log_prob(a_t).sum().item()
#         self.s.append(s)
#         self.a.append(a)
#         self.rn.append(rn)
#         self.done.append(done)
#         self.logπ.append(logπ)
#         self.Vb.append(V.item())

#     def GAE(self, sn_last, done_last):
#         γ = self.γ
#         with torch.no_grad():
#             Vn_last, _, _ = self.wϴ(torch.tensor(sn_last, dtype=torch.float32).unsqueeze(0))
#             Vn_last = 0 if done_last else Vn_last.item()
#         As = []
#         Gt = []
#         A  = 0
#         Vn = Vn_last
#         for rn, V, done in zip(reversed(self.rn), reversed(self.Vb), reversed(self.done)):
#             δ  = rn + γ * Vn * (1 - done) - V
#             A  = δ + γ * self.λ * A * (1 - done)
#             As.insert(0, A)
#             Gt.insert(0, A + V)
#             Vn = V
#         return torch.tensor(As, dtype=torch.float32), torch.tensor(Gt, dtype=torch.float32)

#     def rollout_batch(self):
#         s        = torch.tensor(np.array(self.s),  dtype=torch.float32)
#         a        = torch.tensor(np.array(self.a),  dtype=torch.float32)
#         logπ_old = torch.tensor(self.logπ,         dtype=torch.float32)
#         return s, a, logπ_old

#     def clear_rollout(self):
#         self.s    = []
#         self.a    = []
#         self.rn   = []
#         self.done = []
#         self.logπ = []
#         self.Vb   = []

# # ===============================================================================================
# '''
# PPO — Proximal Policy Optimisation.
# Inherits Gaussian policy and ϴ network from nnPGc, rollout buffer and GAE from nnRollout.
# The student only needs to see this class.
# '''
# class PPO(nnPGcrollout):
#     def __init__(self, ε_clip=0.2, **kw):
#         super().__init__(**kw)
#         self.ε_clip = ε_clip

#     def online(self, s, a, rn, sn, an, done, t):
#         if len(self.s) < self.nsteps: return
#         As_, Gt_          = self.GAE(sn, done)
#         s_, a_, logπ_old_ = self.rollout_batch()
#         As_ = (As_ - As_.mean()) / (As_.std() + 1e-8)         # normalise advantages
#         ε   = self.ε_clip
#         for _ in range(self.epochs):
#             self.wϴ.train()
#             idx = torch.randperm(self.nsteps)
#             for start in range(0, self.nsteps, self.nbatch):
#                 mb                     = idx[start:start + self.nbatch]
#                 s, a, logπ_old, As, Gt = s_[mb], a_[mb], logπ_old_[mb], As_[mb], Gt_[mb]
#                 self.wϴ.optim.zero_grad()
#                 V, μ, σ  = self.wϴ(s) ; V = V.squeeze(1)
#                 p         = Normal(μ, σ)
#                 logπ      = p.log_prob(a).sum(dim=-1)
#                 r         = (logπ - logπ_old).exp()            # π_new / π_old
#                 L_actor   = torch.min(r * As, torch.clamp(r, 1-ε, 1+ε) * As).mean()
#                 L_critic  = 0.5  * F.mse_loss(V, Gt)
#                 L_entropy = 0.01 * p.entropy().mean()
#                 loss      = -(L_actor - L_critic + L_entropy)
#                 loss.backward()
#                 clip_grad_norm_(self.wϴ.parameters(), max_norm=0.5)
#                 self.wϴ.optim.step()
#         self.clear_rollout()

# =================================================================================================
# usage example
# ppo = PPO(env=env, α=3e-4, γ=0.99, seed=1, episodes=1000, **demoGame,
#           trunk=[(8,4,2),(4,4,4)], nF=64,
#           nsteps=128, epochs=4, λ=0.95,
#           ε_clip=0.2,
#           v0=0, final_bias=True,
#           file_name='PPO_exp').interact()