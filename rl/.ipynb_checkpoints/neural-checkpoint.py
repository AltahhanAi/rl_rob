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
class nnMRP(MRP):
    def __init__(self,
                 trunk=[(32, 8, 4), (64, 4, 2), (64, 3, 1)],
                 final_zero=False,
                 final_bias=False,
                 nF=512, rndbatch=True,
                 nbuffer=10000, nbatch=32, endbatch=1,
                 save_weights=1000, load_weights=False,
                 create_vN=True,
                 create_vNn=False, t_Vn=1000,
                 action_dtype=torch.int64,
                 model_class=nnModel, # which type of neural network model from nn.py to create
                 **kw):
        print(f'------------------- 易  {self.__class__.__name__} is being set up 易 ---------------------')
        super().__init__(**kw)
        self.store        = True
        self.create_vN    = create_vN
        self.create_vNn   = create_vNn
        self.t_Vn         = t_Vn
        
        self.trunk        = trunk
        self.nF           = nF
        self.final_zero   = final_zero
        self.final_bias   = final_bias
        
        self.action_dtype = action_dtype
        self.model_class  = model_class
        
        if endbatch > nbatch: endbatch = nbatch - 1
        self.endbatch     = endbatch
        self.nbuffer      = nbuffer
        self.nbatch       = nbatch
        self.rndbatch     = rndbatch
        self.buffer       = deque(maxlen=self.nbuffer)
        
        self.load_weights_ = load_weights
        self.save_weights_ = save_weights
        self.t_           = 0
        self.vN  = self.create_model('V',  self.α, self.final_bias) if create_vN  else None
        self.vNn = self.create_model('Vn', self.α, self.final_bias) if create_vNn else None

    def init_(self):
        torch.manual_seed(self.seed)
        self.vN.load_weights('V') if self.load_weights_ else self.vN.init_weights(self.final_zero)
        self.vNn.eval() if self.create_vNn else None
        self.V_ = self.V

    def create_model(self, net_str, α, final_bias):
        self.state_dim  = self.env.reset().shape
        self.action_dim = 1 if net_str == 'V' else self.env.nA
        model = self.model_class(
            inp_dim=self.state_dim, trunk=self.trunk,
            nF=self.nF, out_dim=self.action_dim,
            α=α, net_str=net_str, final_bias=final_bias
        )
        model.print_model_summary(net_str)
        return model

    def V(self, s=None):
        result = self.vN.predict(s if s is not None else self.env.S_(), self.state_dim)
        return result.detach().numpy()

    def Vn(self, sn):
        return self.vNn.predict(sn, self.state_dim) if self.create_vNn else None

    def allocate(self):
        self.buffer = deque(maxlen=self.nbuffer)

    def store_(self, s=None, a=None, rn=None, sn=None, an=None, done=None, t=0):
        self.buffer.append((
            torch.tensor(s,    dtype=torch.float32),
            torch.tensor(a,    dtype=self.action_dtype),
            torch.tensor(rn,   dtype=torch.float32),
            torch.tensor(sn,   dtype=torch.float32),
            torch.tensor(done, dtype=torch.bool)
        ))

    def slice_(self, buffer, nbatch):
        return list(islice(buffer, len(buffer) - nbatch, len(buffer)))

    def batch(self):
        endbatch = self.endbatch
        samples  = sample(self.buffer, self.nbatch - endbatch) if self.rndbatch else self.slice_(self.buffer, self.nbatch)
        samples.extend(self.slice_(self.buffer, self.endbatch))
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
    def __init__(self, create_vN=False, create_qNn=True, t_Qn=1000, **kw):
        super().__init__(create_vN=create_vN, **kw)
        self.create_qNn = create_qNn
        self.t_Qn = t_Qn
        self.qN  = self.create_model('Q',  self.α, self.final_bias)
        self.qNn = self.create_model('Qn', self.α, self.final_bias) if create_qNn else None

    def init_(self):
        torch.manual_seed(self.seed)
        if self.create_vN:
            self.vN.load_weights('V') if self.load_weights_ else self.vN.init_weights(self.final_zero)
        self.qN.load_weights('Q') if self.load_weights_ else self.qN.init_weights(self.final_zero)
        self.qNn.eval() if self.create_qNn else None
        self.V_ = self.V
        self.Q_ = self.Q

    def Q(self, s):
        return self.qN.predict(s, self.state_dim)

    def Qn(self, sn):
        return self.qNn.predict(sn, self.state_dim) if self.create_qNn else None

# ===============================================================================================
class nnPG(PG(nnMDP)):
    def __init__(self, **kw):
        super().__init__( **kw)
        self.vN  = self.create_model('V',  self.αc, self.final_bias)
        
        self.ϴ = nnACModel(inp_dim=self.state_dim, trunk=self.trunk,
                            nF=self.nF, out_dim=self.env.nA,
                            α=self.αa, net_str='ϴ', final_bias=self.final_bias)
        self.policy = self.softmax

    def init_(self):
        torch.manual_seed(self.seed)
        self.ϴ.load_weights('ϴ') if self.load_weights_ else self.ϴ.init_weights(self.final_zero)
        
        self.V_ = self.V
        self.Q_ = self.Q
        self.H_ = self.H

    def V(self, s=None):
        V, _ = self.ϴ.predict(s if s is not None else self.env.S_(), self.state_dim)
        return V.detach().numpy()

    def H(self, s=None, a=None):
        _, π = self.ϴ.predict(s if s is not None else self.env.S_(), self.state_dim)
        if a is None: return π.detach().numpy()
        return π[a].detach().numpy()

    def softmax(self, s):
        _, π = self.ϴ.predict(s, self.state_dim)
        π = π.detach().numpy().flatten()              # flatten to 1-d
        return np.random.choice(self.env.nA, p=π)
        (s, rn,sn, done, a,an) 
        
    # def online(self, s, rn,sn, done, a,an):
    #     if len(self.buffer) < self.nbatch: return
    #     (s, a, rn, sn, dones), inds = self.batch()
    #     Vn, _ = self.ϴ(sn)
    #     Vn    = Vn.squeeze(1).detach()
    #     Vn[dones] = 0
    #     Gt = rn + self.γ * Vn
    #     a  = torch.tensor(np.array(a.tolist()), dtype=torch.int64)
    #     loss = self.ϴ.fit(s, a, Gt)

# ===============================================================================================
class nnPGc(PG(nnMDP)):
    def __init__(self, σ=1, σmin=.01, dσ=1, Tσ=0, **kw):
        super().__init__(action_dtype=torch.float32, **kw)
        self.σ    = σ
        self.σ0   = σ
        self.dσ   = dσ
        self.Tσ   = Tσ
        self.σmin = σmin
        self.ϴ    = nnACcModel(inp_dim=self.state_dim, trunk=self.trunk,
                                nF=self.nF, out_dim=self.env.action_space.shape[0],
                                α=self.α, net_str='ϴ', final_bias=self.final_bias)
        self.policy = self.Gaussian

    def init_(self):
        torch.manual_seed(self.seed)
        self.ϴ.load_weights('ϴ') if self.load_weights_ else self.ϴ.init_weights(self.final_zero)
        self.V_ = self.V
        self.Q_ = self.Q

    def V(self, s=None):
        V, _, _ = self.ϴ.predict(s if s is not None else self.env.S_(), self.state_dim)
        return V.detach().numpy()

    def μ_π(self, s):
        _, μ, _ = self.ϴ.predict(s, self.state_dim)
        return μ.detach().numpy()

    def σ_π(self, s):
        _, _, σ = self.ϴ.predict(s, self.state_dim)
        return σ.detach().numpy()

    def Gaussian(self, s):
        if self.dσ < 1: self.σ = max(self.σmin, self.σ  * self.dσ)
        if self.Tσ > 0: self.σ = max(self.σmin, self.σ0 * (1 - self.t_ / self.Tσ))
        _, μ, σ = self.ϴ.predict(s, self.state_dim)
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

    # def online(self, s, rn,sn, done, a,an):
    #     if len(self.buffer) < self.nbatch: return
    #     (s, a, rn, sn, dones), inds = self.batch()
    #     Vn, _, _ = self.ϴ(sn)
    #     Vn = Vn.squeeze(1).detach()
    #     Vn[dones] = 0
    #     Gt = rn + self.γ * Vn
    #     a  = torch.tensor(np.array(a.tolist()), dtype=torch.float32)
    #     loss = self.ϴ.fit(s, a, Gt)

# ===============================================================================================
class TDN(nnMRP):
    def online(self, *args):
        if len(self.buffer) < self.nbatch: return
        (s, a, rn, sn, dones), inds = self.batch()
        Vs  = self.vN(s)
        Vn  = self.vNn(sn).detach() if self.create_vNn and self.ep > 2 else self.vN(sn).detach()
        Vn[dones] = 0
        targets = self.γ * Vn + rn.unsqueeze(1)
        loss = self.vN.fit(Vs, targets, exact=True)
        if self.t_Vn and self.t_ % self.t_Vn == 0 and self.create_vNn:
            self.vNn.set_weights(self.vN, 'V', self.t_)

# ===============================================================================================
class DQN(nnMDP):
    def online(self, *args):
        if len(self.buffer) < self.nbatch: return
        (s, a, rn, sn, dones), inds = self.batch()
        Qs  = self.qN(s)
        Qn  = self.qNn(sn).detach() if self.create_qNn and self.ep > 2 else self.qN(sn).detach()
        Qn[dones] = 0
        targets = Qs.clone().detach()
        targets[inds, a] = self.γ * Qn.max(1).values + rn
        loss = self.qN.fit(Qs, targets)
        if self.t_Qn and self.t_ % self.t_Qn == 0 and self.create_qNn:
            self.qNn.set_weights(self.qN, 'Q', self.t_)

# ===============================================================================================
class DDQN(DQN):
    def online(self, *args):
        if len(self.buffer) < self.nbatch: return
        (s, a, rn, sn, dones), inds = self.batch()
        Qs  = self.qN(s)
        an  = self.qN(sn).detach().argmax(1)
        Qn  = self.qNn(sn).detach() if self.create_qNn and self.ep > 2 else self.qN(sn).detach()
        Qn[dones] = 0
        targets = Qs.clone().detach()
        targets[inds, a] = self.γ * Qn[inds, an] + rn
        loss = self.qN.fit(Qs, targets, exact=True)
        if self.t_Qn and self.t_ % self.t_Qn == 0 and self.create_qNn:
            self.qNn.set_weights(self.qN, 'Q', self.t_)

# ===============================================================================================
class DuelDQN(DQN):
    def __init__(self, model_class=nnDuelModel, **kw):
        super().__init__(model_class=model_class, **kw)

# ===============================================================================================
def AC(base, label):
    class nnAC_(base):
        def online(self, *args):
            if len(self.buffer) < self.nbatch: return
            (s, a, rn, sn, dones), inds = self.batch()
            Vn, *_ = self.ϴ(sn)
            Vn = Vn.squeeze(1).detach()
            Vn[dones] = 0
            Gt = rn + self.γ * Vn
            a  = torch.tensor(np.array(a.tolist()), dtype=torch.float32 if label == 'continuous' else torch.int64)
            loss = self.ϴ.fit(s, a, Gt)
    nnAC_.__name__ = f'AC_{label}'
    return nnAC_

# ===============================================================================================
nnAC  = AC(nnPG,  'discrete')
nnACc = AC(nnPGc, 'continuous')


# ===============================================================================================
'''
nnRollout replaces the replay buffer with a rollout buffer for on-policy methods.
It collects full trajectories, computes GAE advantages, and supports multiple epochs
over the same rollout.
'''
class nnRollout(nnMRP):
    def __init__(self, nsteps=128, epochs=4, λ=0.95, **kw):
        super().__init__(create_vN=False, **kw)
        self.nsteps = nsteps
        self.epochs = epochs
        self.λ      = λ
        self.s    = []
        self.a    = []
        self.rn   = []
        self.done = []
        self.logπ = []
        self.Vb   = []

    def store_(self, s=None, a=None, rn=None, sn=None, an=None, done=None, t=0):
        s_t  = torch.tensor(s, dtype=torch.float32).unsqueeze(0)
        a_t  = torch.tensor(a, dtype=torch.float32)
        with torch.no_grad():
            V, μ, σ = self.ϴ(s_t)
            logπ    = Normal(μ, σ).log_prob(a_t).sum().item()
        self.s.append(s)
        self.a.append(a)
        self.rn.append(rn)
        self.done.append(done)
        self.logπ.append(logπ)
        self.Vb.append(V.item())

    def GAE(self, sn_last, done_last):
        γ = self.γ
        with torch.no_grad():
            Vn_last, _, _ = self.ϴ(torch.tensor(sn_last, dtype=torch.float32).unsqueeze(0))
            Vn_last = 0 if done_last else Vn_last.item()
        As = []
        Gt = []
        A  = 0
        Vn = Vn_last
        for rn, V, done in zip(reversed(self.rn), reversed(self.Vb), reversed(self.done)):
            δ  = rn + γ * Vn * (1 - done) - V
            A  = δ + γ * self.λ * A * (1 - done)
            As.insert(0, A)
            Gt.insert(0, A + V)
            Vn = V
        return torch.tensor(As, dtype=torch.float32), torch.tensor(Gt, dtype=torch.float32)

    def rollout_batch(self):
        s        = torch.tensor(np.array(self.s),  dtype=torch.float32)
        a        = torch.tensor(np.array(self.a),  dtype=torch.float32)
        logπ_old = torch.tensor(self.logπ,         dtype=torch.float32)
        return s, a, logπ_old

    def clear_rollout(self):
        self.s    = []
        self.a    = []
        self.rn   = []
        self.done = []
        self.logπ = []
        self.Vb   = []

# ===============================================================================================
'''
PPO — Proximal Policy Optimisation.
Inherits Gaussian policy and ϴ network from nnPGc, rollout buffer and GAE from nnRollout.
The student only needs to see this class.
'''
class PPO(nnPGc, nnRollout):
    def __init__(self, ε_clip=0.2, **kw):
        super().__init__(**kw)
        self.ε_clip = ε_clip

    def online(self, s, a, rn, sn, an, done, t):
        if len(self.s) < self.nsteps: return
        As_, Gt_          = self.GAE(sn, done)
        s_, a_, logπ_old_ = self.rollout_batch()
        As_ = (As_ - As_.mean()) / (As_.std() + 1e-8)         # normalise advantages
        ε   = self.ε_clip
        for _ in range(self.epochs):
            self.ϴ.train()
            idx = torch.randperm(self.nsteps)
            for start in range(0, self.nsteps, self.nbatch):
                mb                     = idx[start:start + self.nbatch]
                s, a, logπ_old, As, Gt = s_[mb], a_[mb], logπ_old_[mb], As_[mb], Gt_[mb]
                self.ϴ.optim.zero_grad()
                V, μ, σ  = self.ϴ(s) ; V = V.squeeze(1)
                p         = Normal(μ, σ)
                logπ      = p.log_prob(a).sum(dim=-1)
                r         = (logπ - logπ_old).exp()            # π_new / π_old
                L_actor   = torch.min(r * As, torch.clamp(r, 1-ε, 1+ε) * As).mean()
                L_critic  = 0.5  * F.mse_loss(V, Gt)
                L_entropy = 0.01 * p.entropy().mean()
                loss      = -(L_actor - L_critic + L_entropy)
                loss.backward()
                clip_grad_norm_(self.ϴ.parameters(), max_norm=0.5)
                self.ϴ.optim.step()
        self.clear_rollout()

# =================================================================================================
# usage example
# ppo = PPO(env=env, α=3e-4, γ=0.99, seed=1, episodes=1000, **demoGame,
#           trunk=[(8,4,2),(4,4,4)], nF=64,
#           nsteps=128, epochs=4, λ=0.95,
#           ε_clip=0.2,
#           final_zero=True, final_bias=True,
#           file_name='PPO_exp').interact()