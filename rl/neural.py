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

'''nnMRP, nnMDP, PG Classes with Neural Net
    games: trunk=[(32, 8, 4), (64, 4, 2), (64, 3, 1)], nF=S
    igrid: trunk=[ (8, 4, 2),  (4, 4, 4)],             nF=S
'''

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
                 clipCNN=False,
                 clipModel=False,
                 β_entropy=0.01,
                 optimiser=None,
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
        
        # clipCNN is redundant and can be replaced by clipModel
        self.clipModel = clipModel 
        self.clipCNN = clipCNN # kept for compatibility, should be removed later
        self.w  = self.create_model('V',  self.model_class) if create_w  else None
        self.wn = self.create_model('Vn', self.model_class) if create_wn else None
        self.optimiser = optimiser
        

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
            α=self.α, 
            αv=getattr(self, 'αv', None), 
            αq=getattr(self, 'αq', None),
            αt=getattr(self, 'αt', None),
            σ=getattr(self, 'σ', 1),
            τ=getattr(self, 'τ', 1.0),
            β_entropy=getattr(self, 'β_entropy', 0.01),
            net_str=net_str, final_bias=self.final_bias,
            clipModel=self.clipModel or self.clipCNN,
            optimiser=self.optimiser,
        )
        if self.model_summary: model.print_model_summary(net_str)
        return model
        

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
            if endbatch: samples.extend(self.slice(endbatch))
        else: samples = self.slice(nbatch)

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
    def __init__(self, ac_model_class=nnACSharedModel, **kw):
        # nnAC_SharedModel returns two parts: the V for the critic and Mu and sigma for the Actor
        # no need to initialise the w independently unless we don't want to share the same 
        # trunk between the actor and the critic
        super().__init__(create_w=False, create_W=False, create_Wn=False, **kw)        
        self.wϴ = self.create_model(net_str='wϴ',  model_class=ac_model_class) # discrete actions
        self.policy = self.softmax

    def init_(self):
        torch.manual_seed(self.seed)
        self.wϴ.load_weights('ϴ') if self.load_weights_ else self.wϴ.init_weights(head1_v0=self.v0, head2_q0=self.q0)
        
        self.V_ = self.V
        # self.Q_ = self.Q
        self.H_ = self.H
    
    def V(self, s=None):
        V, _ = self.wϴ.predict(s if s is not None else self.env.S_(), self.state_dim)
        return V.detach().numpy().squeeze(-1) # necessary for the bas classes

    def H(self, s=None, a=None):
        _, π = self.wϴ.predict(s if s is not None else self.env.S_(), self.state_dim)
        if a is None: return π.detach().numpy()
        return π[a].detach().numpy() # necessary for base classes
        
    def softmax(self, s):
        if self.dτ < 1: self.τ = max(self.τmin, self.τ  *self.dτ)                  # exponential decay
        if self.Tτ > 0: self.τ = max(self.τmin, self.τ0 * (1 - self.t_ / self.Tτ)) # linear      decay
        self.wϴ.τ = self.τ # set τ in wϴ model only if not learned in wϴ

        _, π = self.wϴ.predict(s, self.state_dim)
        π = π.detach().numpy().flatten()              # flatten to 1-d    
        a = choices(range(self.env.nA), weights=π, k=1)[0]
        return a
        
# ===============================================================================================
class nnPGc(PG(nnMDP)):
    def __init__(self, σ=1, σmin=.01, dσ=1, Tσ=0, ac_model_class=nnACcSharedModel, **kw):
        super().__init__(action_dtype=torch.float32, create_w=False, create_W=False, create_Wn=False, **kw)
        self.σ    = σ
        self.σ0   = σ
        self.dσ   = dσ
        self.Tσ   = Tσ
        self.σmin = σmin
        
        self.wϴ = self.create_model(net_str='wϴ',  model_class=ac_model_class) # continuous actions
        self.policy = self.Gaussian

    def init_(self):
        torch.manual_seed(self.seed)
        self.wϴ.load_weights('ϴ') if self.load_weights_ else self.wϴ.init_weights(head1_v0=self.v0, head2_q0=self.q0)
        self.V_ = self.V
        # self.Q_ = self.Q

    def V(self, s=None):
        V, _, _ = self.wϴ.predict(s if s is not None else self.env.S_(), self.state_dim)
        return V.detach().numpy().squeeze(-1)

    def Gaussian(self, s):
        if self.dσ < 1: self.σ = max(self.σmin, self.σ  * self.dσ)
        if self.Tσ > 0: self.σ = max(self.σmin, self.σ0 * (1 - self.t_ / self.Tσ))
        if self.wϴ.σ_head is None: self.wϴ.σ = self.σ # set σ in wϴ model only if not learned in wϴ
        
        _, μ, σ = self.wϴ.predict(s, self.state_dim)
        μ = μ.detach().numpy()
        σ = σ.detach().numpy() if self.wϴ.σ_head is not None  else σ
        a = np.random.normal(μ, σ) # if you change this here, then change it in the linear.py
        a = np.clip(a, self.env.action_space.low, self.env.action_space.high)
        return np.atleast_1d(a)

    # def μ_π(self, s):
    #     _, μ, _ = self.wϴ.predict(s, self.state_dim)
    #     return μ.detach().numpy()

    # def σ_π(self, s):
    #     _, _, σ = self.wϴ.predict(s, self.state_dim)
    #     return σ.detach().numpy()
        
    # def π(self, s, a):
    #     μ, σ = self.μ_π(s), self.σ_π(s)
    #     return np.prod((1.0 / (np.sqrt(2 * np.pi) * σ)) * np.exp(-((a - μ) ** 2) / (2 * σ**2)))

    # def logπ(self, s, a):
    #     μ, σ = self.μ_π(s), self.σ_π(s)
    #     return np.sum(-((a - μ)**2) / (2 * σ**2) - np.log(σ) - .5 * np.log(2 * np.pi))

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
        
# ===============================================================================================
class nnREINFORCE(nnPG):

    def init(self):
        self.nbuffer = self.max_t

    def step0(self):
        self.buffer.clear()

    # ---------------------------- 🌘 offline, MC policy learning: end-of-episode learning ----------------------------
    def offline(self):
        Gt = 0
        γt = self.γ ** self.t

        for t in range(self.t, -1, -1):
            s, a, rn, sn, done = self.trajectory(t=t)

            Gt = self.γ * Gt + rn.squeeze(-1)
            self.wϴ.fit(s, a, Gt, γt=γt)

            γt /= self.γ if self.γ != 0 else 1
# ============================= needs verfying========================================
'''
collect the Gt and samples and then batch update them
'''
class nnREINFORCE__(nnPG):
    def init(self):
        self.nbuffer = self.max_t

    def step0(self):
        self.buffer.clear()

    def offline(self):
        Gt  = 0
        γt  = self.γ ** self.t
        Gts = []
        γts = []
        for t in range(self.t, -1, -1):
            s, a, rn, _, _ = self.trajectory(t=t)
            Gt  = self.γ * Gt + rn.squeeze(-1)
            Gts.insert(0, Gt.clone().detach())
            γts.insert(0, γt)
            γt /= self.γ if self.γ != 0 else 1

        # now update in mini-batches of nbatch steps
        T   = self.t + 1
        idx = torch.randperm(T)
        s_all  = torch.stack([self.trajectory(t=t)[0].squeeze(0) for t in range(T)])
        a_all  = torch.stack([self.trajectory(t=t)[1].squeeze(0) for t in range(T)])
        Gt_all = torch.stack(Gts)
        γt_all = torch.tensor(γts)
        for start in range(0, T, self.nbatch):
            mb = idx[start:start + self.nbatch]
            self.wϴ.fit(s_all[mb], a_all[mb], Gt_all[mb], γt=γt_all[mb])
# ===============================================================================================

# ===============================================================================================
class nnActor_Critic(nnPG):
    
    def step0(self):
        self.γt = 1    
    # ---------------------------- 🌖 online control learning ----------------------------
    def online(self, *args):
        s, a, rn, sn, done = self.trajectory(-1)
        
        Vn, *_ = self.wϴ(sn)
        Vn = Vn.squeeze(-1).detach() 
        Vn[done] = 0
        
        Gt = self.γ * Vn + rn.squeeze(-1) 

        self.wϴ.fit(s, a, Gt,  γt=self.γt)
        # self.γt *= self.γ  # dropped to promote stability

# ===============================================================================================        
def AC(base=nnPG, name='nnActor_Critic'):
    class nnActor_Critic_(base):
        def step0(self):
            self.γt = 1
         # ---------------------------- 🌖 online control learning ----------------------------
        def online(self, *args):
            s, a, rn, sn, done = self.trajectory(-1)
            Vn, *_ = self.wϴ(sn)
            Vn = Vn.squeeze(-1).detach()
            
            Vn[done] = 0
            Gt = self.γ * Vn + rn.squeeze(-1)
            
            self.wϴ.fit(s, a, Gt, γt=self.γt)
            # self.γt *= self.γ  # usually dropped in implmentation
            
    nnActor_Critic_.__name__     = name
    nnActor_Critic_.__qualname__ = name # for pickle to work
    
    return nnActor_Critic_

# nnActor_Critic   = AC(nnPG,  'nnActor_Critic')     # discrete  — softmax already defined above, above kept for reference
nnActor_c_Critic = AC(nnPGc, 'nnActor_c_Critic')   # continuous — Gaussian
# ===============================================================================================
class nnActor_Critic_nSteps(nnPG):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.rndbatch = False
        self.endbatch = 0
        self.nbuffer  = self.nbatch

    def step0(self):
        self.buffer.clear()
        
     # ---------------------------- 🌖 online control learning ----------------------------
    def online(self, *args):
        if len(self.buffer) < self.nbatch: return # ensures that collect enough rollout consecutive samples
        self._update()
    
    # ---------------------------- 🌘 offline control learning ----------------------------
    def offline(self):
        if len(self.buffer) == 0: return # ensures to process remaining samples of the episode, important for sparse rewards
        self._update()

     # ---------------------------- 🌖 update for both online and remainder offline🌘 ------
    def _update(self):
        n = len(self.buffer)
        (s, a, rn, sn, dones), _ = self.batch(nbatch=n, endbatch=0)
        
        Vn, *_ = self.wϴ(sn[-1].unsqueeze(0))
        Vn     = Vn.squeeze().detach()
        Gt     = Vn * (1 - dones[-1].float())
        
        returns = []
        for r, d in zip(reversed(rn.squeeze(-1)), reversed(dones)):
            Gt = r + self.γ * Gt * (1 - d.float())
            returns.insert(0, Gt)
        Gt = torch.stack(returns)
        
        self.wϴ.fit(s, a, Gt)
        self.buffer.clear()        # discard rollout — on-policy, must not reuse
# ===============================================================================================

class nnPPO(nnActor_Critic_nSteps):
    def __init__(self, ε_clip=0.2, epochs=4, λ=0.95, norm_adv=False, **kw):
        super().__init__(ac_model_class=nnACEpochModel, **kw)
        self.ε_clip   = ε_clip
        self.epochs   = epochs
        self.λ        = λ
        self.norm_adv = norm_adv

    def _update(self):
        n = len(self.buffer)
        if n < 2: self.buffer.clear(); return
        (s, a, rn, sn, dones), _ = self.batch(nbatch=n, endbatch=0)
        
        self.wϴ.eval()
        with torch.no_grad():
            V_old, π_old = self.wϴ.Vπ(s)
            logπ_old     = self.wϴ.logπ(π_old, a)
            Vn_all, _    = self.wϴ.Vπ(sn)
            Vn_all       = Vn_all * (1 - dones.float())            
            A, gae       = torch.zeros(n), torch.tensor(0.0)
            for t in reversed(range(n)):
                δ    = rn.squeeze(-1)[t] + self.γ * Vn_all[t] - V_old[t]
                gae  = δ + self.γ * self.λ * (1 - dones[t].float()) * gae
                A[t] = gae
            Gt = (A + V_old).detach()
            if self.norm_adv and A.std() > 1e-6:
                A = ((A - A.mean()) / (A.std() + 1e-8)).detach()
            else:
                A = A.detach()

        epochs  = self.epochs if n == self.nbatch else 1
        mb_size = min(self.nbatch, n)

        self.wϴ.train()
        self.wϴ.fit(s, a, A, Gt, logπ_old, epochs, mb_size, self.ε_clip) # nnACEpochModel.fit
        self.buffer.clear()
# ============================================================================================
'''
A tidy implementation for PPO with class factory, the abov eis left for the discrete case, both eqiv
'''
def PPO(base=nnPG, model=nnACEpochModel, name='nnPPO'):
    class nnPPO_(base):
        def __init__(self, ε_clip=0.2, epochs=4, λ=0.95, **kw):
            super().__init__(ac_model_class=model, **kw)
            self.ε_clip = ε_clip
            self.epochs = epochs
            self.λ      = λ

            self.rndbatch = False
            self.endbatch = 0
            self.nbuffer  = self.nbatch

        def step0(self):
            self.buffer.clear()

        # ---------------------------- 🌖 online control learning ----------------------------
        def online(self, *args):
            if len(self.buffer) < self.nbatch: return   # wait for a full rollout
            self._update()

        # ---------------------------- 🌘 offline: flush remainder at episode end ------------
        def offline(self):
            if len(self.buffer) == 0: return
            self._update()

        # ---------------------------- 🌗 GAE: generalised advantage estimation --------------
        def GAE(self, rn, Vo, Vn, dones):
            """ Generalised Advantage Estimation.   A_t = δ_t + γλ(1 - d_t)·A_{t+1}"""
            n      = len(rn)
            A, gae = torch.zeros(n), torch.tensor(0.0)
            for t in reversed(range(n)):
                δ    = rn[t] + self.γ * Vn[t] - Vo[t]
                gae  = δ + self.γ * self.λ * (1 - dones[t].float()) * gae
                A[t] = gae
            return A

        # ---------------------------- 🌖 PPO update: GAE + clipped epochs -------------------
        def _update(self):
            n = len(self.buffer)
            if n < 2: self.buffer.clear(); return
            (s, a, rn, sn, dones), _ = self.batch(nbatch=n, endbatch=0)
            rn = rn.squeeze(-1)

            self.wϴ.eval()
            with torch.no_grad():
                Vo, πo    = self.wϴ.Vπ(s)
                logπo     = self.wϴ.logπ(πo, a)
                Vn, _     = self.wϴ.Vπ(sn)
                Vn        = Vn * (1 - dones.float())

                A  = self.GAE(rn, Vo, Vn, dones)
                Gt = (A + Vo).detach()
                A  = ((A - A.mean()) / (A.std() + 1e-8)).detach()

            epochs  = self.epochs if n == self.nbatch else 1
            mb_size = min(self.nbatch, n)

            self.wϴ.train()
            self.wϴ.fit(s, a, A, Gt, logπo, epochs, mb_size, self.ε_clip)
            self.buffer.clear()        # discard rollout — on-policy, must not reuse

    nnPPO_.__name__     = name
    nnPPO_.__qualname__ = name         # for pickle
    return nnPPO_


# nnPPO   = PPO(nnPG,  nnACEpochModel,   'nnPPO')      # discrete   — softmax
nnPPO_c = PPO(nnPGc, nnACEpochModel_c, 'nnPPO_c')    # continuous — Gaussian
# ===============================================================================================